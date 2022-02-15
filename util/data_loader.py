import os

import numpy as np
import torch as t
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def DDP_load_data(cfg):
    assert cfg.dataset == 'imagenet' or cfg.dataset == 'imagenet1000'
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    inception_normalize = tv.transforms.Normalize(mean=[0.5,0.5,0.5],
                                                 std=[0.5,0.5,0.5])
    if cfg.dataset == 'imagenet1000':
        traindir = os.path.join(cfg.path, 'train_img')
        valdir = os.path.join(cfg.path, 'val_img')
    else:
        traindir = os.path.join(cfg.path, 'train')
        valdir = os.path.join(cfg.path, 'val')
    print("Train dir:", traindir)
    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    print("Train_sampler initing...")
    train_sampler = t.utils.data.distributed.DistributedSampler(train_set)
    print("Train_sampler init successed")

    train_loader = t.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)
    
    test_loader = t.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)

    val_loader = test_loader
    return train_loader, val_loader, test_loader, train_sampler