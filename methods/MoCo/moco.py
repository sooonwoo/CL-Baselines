import torch 
import torch.nn as nn 

from methods.base import CLModel

class MoCoModel(CLModel):
    def __init__(self, args):
        super().__init__(args)

        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.temp

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.encoder_k = self.model_generator()

        if self.mlp_layers == 2:
            self.proj_head_k = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )
            self.proj_head_q = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )

        elif self.mlp_layers == 3:
            self.proj_head_k = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )
            self.proj_head_q = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )
        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(128, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, print(self.K, batch_size)  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _forward(self, im_q, im_k):
        # compute query features
        q = self.backbone(im_q)  # queries: NxC
        q = self.proj_head_q(q)
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.proj_head_k(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.criterion(logits, labels)

        return loss, q, k, logits, labels

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        loss_12, q1, k2, logits1, labels2 = self._forward(im_q, im_k)
        loss_21, q2, k1, logits2, labels1= self._forward(im_k, im_q)
        
        loss = loss_12 + loss_21

        # dequeue and enqueue
        self._momentum_update_key_encoder() 
        self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))

        return loss
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output