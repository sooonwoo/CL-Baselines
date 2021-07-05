import os
import random
import numpy as np

from PIL import Image, ImageFilter
from functools import partial

import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .coco import COCODataset


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def set_loader(args):
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        args.size = 32
        args.num_classes = 10
        args.save_freq = 100
    
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        args.size = 32
        args.num_classes = 100
        args.save_freq = 100
    
    elif 'imagenet' in args.dataset: 
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.size = 224
        args.save_freq = 1
        if args.dataset == 'imagenet-100':
            args.num_classes = 100
        else:
            args.num_classes = 1000
    
    elif args.dataset == 'coco':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.size = 224
        args.save_freq = 1
        args.save_freq = 1
        args.num_classes = -1
    else:
        raise ValueError(args.dataset)

    normalize = transforms.Normalize(mean=mean, std=std)


    ####################### Define Transforms #######################
    if 'cifar' in args.dataset: 
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])

        ft_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # ImageNet, COCO 
    else:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # Gaussian Blur is added 
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        
        ft_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_transform =transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    ####################### Define Datasets #######################
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_path,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)

        ft_dataset = datasets.CIFAR10(root=args.data_path,
                                         transform=ft_transform,
                                         download=False)
        test_dataset = datasets.CIFAR10(root=args.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
        memory_dataset = datasets.CIFAR10(root=args.data_path,
                                       train=True,
                                       transform=test_transform,
                                       download=False)
    
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_path,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        ft_dataset = datasets.CIFAR100(root=args.data_path,
                                         transform=ft_transform,
                                         download=False)
        test_dataset = datasets.CIFAR100(root=args.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
        memory_dataset = datasets.CIFAR100(root=args.data_path,
                                       train=True,
                                       transform=test_transform,
                                       download=False)
   
    elif 'imagenet' in args.dataset:
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
                                            transform=TwoCropTransform(train_transform))

        ft_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
                                         transform=ft_transform,
                                        )
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"),
                                       transform=test_transform,
                                        )
        memory_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
                                       transform=test_transform,
                                        )
        if args.dataset == 'imagenet-100':
            class_list = np.loadtxt("./imagenet100.txt", dtype=str)
    
            idx = [i for i, f in enumerate(train_dataset.imgs) if f[0].split('/')[-2] in class_list]
            train_dataset = torch.utils.data.Subset(train_dataset, idx)
            
            idx = [i for i, f in enumerate(ft_dataset.imgs) if f[0].split('/')[-2] in class_list]
            ft_dataset = torch.utils.data.Subset(ft_dataset, idx)

            for i, c in enumerate(class_list):
                for j in idx:
                    if ft_dataset.dataset.imgs[j][0].split('/')[-2] == c:
                        ft_dataset.dataset.imgs[j] = (ft_dataset.dataset.imgs[j][0] , i)
                        ft_dataset.dataset.targets[j] = i

            idx = [i for i, f in enumerate(memory_dataset.imgs) if f[0].split('/')[-2] in class_list]
            memory_dataset = torch.utils.data.Subset(memory_dataset, idx)
            
            for i, c in enumerate(class_list):
                for j in idx:
                    if memory_dataset.dataset.imgs[j][0].split('/')[-2] == c:
                        memory_dataset.dataset.imgs[j] = (memory_dataset.dataset.imgs[j][0] , i)
                        memory_dataset.dataset.targets[j] = i

            idx = [i for i, f in enumerate(test_dataset.imgs) if f[0].split('/')[-2] in class_list]
            test_dataset = torch.utils.data.Subset(test_dataset, idx)
 
            for i, c in enumerate(class_list):
                for j in idx:
                    if test_dataset.dataset.imgs[j][0].split('/')[-2] == c:
                        test_dataset.dataset.imgs[j] = (test_dataset.dataset.imgs[j][0], i)
                        test_dataset.dataset.targets[j] = i
    elif args.dataset == 'coco':
        train_dataset = COCODataset(root = os.path.join(args.data_path,'/coco/images/train2017'),
                            annFile = os.path.join(args.data_path,'coco/annotations/instances_train2017.json'),
                            transform = TwoCropTransform(train_transform))
        # For coco dataset, linear-eval/fine-tuning is deprecated.
        ft_dataset = train_dataset                            

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        ft_sampler = torch.utils.data.distributed.DistributedSampler(ft_dataset)
    else:
        train_sampler = None 
        ft_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    ft_loader = torch.utils.data.DataLoader(
        ft_dataset, batch_size=args.eval_batch_size, shuffle=(ft_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=ft_sampler)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, train_sampler, ft_loader, ft_sampler, test_loader, memory_loader
