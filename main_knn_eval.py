import os 
import sys 
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from loaders.dataloader import set_loader
from networks.resnet_org import model_dict 
from networks.resnet_cifar import model_dict as model_dict_cifar

from utils.knn import knn_monitor 
from utils.util import load_model 

parser = argparse.ArgumentParser(description='PyTorch Training')

# dataloader 
parser.add_argument('--data_path', default='../datasets')
parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet-100', 'imagenet', 'coco'])
parser.add_argument('--batch_size', default=512, type=int)                          
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# model 
parser.add_argument('--method', default='simclr', choices=['simclr', 'moco', 'simsiam'])

parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--saved_path', default='none', type=str)

args = parser.parse_args()

# args.temp = 1
args.distributed = False 

def main(args):
    # load model
    if 'cifar' in args.dataset:
        print('CIFAR-variant Resnet is loaded')
        model_fun, feat_dim= model_dict_cifar[args.arch]
    else:
        print('Original Resnet is loaded')
        model_fun, feat_dim = model_dict[args.arch]
    
    encoder = model_fun()
    encoder = load_model(encoder, args.saved_path).cuda()
    # model = set_model(args)
    
    # create data loader
    train_loader, train_sampler, ft_loader, ft_sampler, test_loader, memory_loader = set_loader(args)
    
    # eval 
    knn_acc = knn_monitor(encoder, memory_loader, test_loader, -1, hide_progress=False, classes=args.num_classes, subset=args.dataset=='imagenet-100')
    print('knn acc: {:.3f}'.format(knn_acc))

if __name__ == '__main__':
    main(args)