#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# @Author: levi
# Copyright (c) 2020 
# Created by Levi levio123@163.com
# --------------------------------------------------------
import torch
import os.path as osp
from cfg.voc import cfg
from yolo.yolov1 import create_yolov1
from data_utils.build import make_dist_voc_loader
from torch.cuda import amp


def train_step(epochs, model, train_loader, test_loader, optim, classes, device='cuda'):
    scaler = amp.GradScaler(enabled=True)
    for epoch in range(epochs):
        print('epoch =', epoch)
        for idx, (img, gt_info) in enumerate(train_loader):
            optim.zero_grad()
            img = img.to(device)
            loss_dict = model(img, gt_info)
            loss = sum(l for l in loss_dict.values())
            print(int(loss), end=' ')
            scaler.scale(loss).backward()
            optim.step()
        torch.save(model.state_dict(), 'best_model.pth')


def train():
    train_cfg = cfg['train_cfg']
    model_cfg = cfg['model_cfg']
    model_name = model_cfg['model_type']
    epochs = train_cfg['epochs']
    classes = train_cfg['classes']
    lr = train_cfg['lr']
    bs = train_cfg['batch_size']
    device = train_cfg['device']
    out_dir = train_cfg['out_dir']
    resume = train_cfg['resume']
    use_sgd = train_cfg['use_sgd']
    mile = train_cfg['milestone']
    gamma = train_cfg['gamma']
    train_root = train_cfg['dataroot']
    img_size = train_cfg['img_size']

    model = create_yolov1(model_cfg)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr[0],
                                 weight_decay=5e-5)

    train_loader = make_dist_voc_loader(osp.join(train_root, 'train.txt'),
                                        img_size=img_size,
                                        batch_size=bs,
                                        train=True,
                                        )
    test_loader = make_dist_voc_loader(osp.join(train_root, 'voc2007_test.txt'),
                                       img_size=img_size,
                                       batch_size=16,
                                       train=False,
                                       )

    train_step(epochs, model, train_loader, test_loader, optimizer, classes, device=device)


if __name__ == '__main__':
    train()

