# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/11 23:39:19
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import sys
sys.path.append('.')
import config
import dataset
import module

import time,os
import torch.utils.data as tud
import torch as th
import torch.nn as nn
import cv2
import numpy as np

net = module.yolov3.YOLOV3(img_size=256)

fdict = th.load('./weights/20230713_005639_temp.pth',map_location='cpu')
net.load_state_dict(fdict['weights'])
net.train()
ds = dataset.cocoDataSet(dataType='val',output_size=[256,256])
for di in range(10):
    d = ds[di]
    x = d[0].reshape(3,256,256)
    x=d[0]
    x = np.array(x,dtype='uint8').transpose(1,2,0)
    # x=cv2.resize(x,[160,160])
    x=th.tensor(x.transpose(2,0,1),dtype=th.float).view(1,3,256,256)
    # for i in d[1:]:
    #     y=i[:,:,:,4]
    #     idx = th.where(y>0.8)
    #     print(idx)
    #     print(y[idx])

    print('--------------------')
    ts=d[1:]
    
    img = np.array(d[0],dtype='uint8').transpose(1,2,0)
    ps = net(x)
    print('-----------:',ts[1][11,5,2].shape,ps[1].shape)
    print('-----------:',ts[1][11,5,2,:5],ps[1][0,11,5,2,:5])
    for sz,p in enumerate(ps):
        t=d[1:][sz].reshape(p.shape[0],p.shape[1],p.shape[2],p.shape[3],p.shape[4])
        c=p[:,:,:,:,4]
        tc=t[:,:,:,:,4]
        
        c=nn.functional.sigmoid(c)
        idx = th.where(c>0.5)
        if (il:=len(idx[0]))>0:
            cwhs=p[idx][:,:4]
            ycwhs=t[idx][:,:4]
            ones = p[idx][:,5:]
            # print(cwhs)
            for j in range(il):
                print(c[idx[0][j],idx[1][j],idx[2][j],idx[3][j]].detach(),tc[idx[0][j],idx[1][j],idx[2][j],idx[3][j]])
                cwh=np.array(cwhs[j,:].detach()).flatten()
                ycwh = np.array(ycwhs[j,:]).flatten()
                ps_ind = int(idx[3][j])
                psc = np.array(ds.cfg['pre_scale'][sz][ps_ind])/416
                xywh = dataset.utils.tcwh2xywh(cwh,sz,int(idx[2][j]),int(idx[2][j]),psc[0],psc[1],[5,10,20])
                yxywh = dataset.utils.tcwh2xywh(ycwh,sz,int(idx[2][j]),int(idx[1][j]),psc[0],psc[1],[5,10,20])
                onehot = ones[j].flatten()
                print(xywh,yxywh)
                cl = onehot.argmax()
                print(cl,ds.cfg['category'][cl])
                dataset.utils.drawLabel(img,[[cl,xywh],[cl,yxywh]])
