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

net = module.yolov3.YOLOV3(img_size=160)

fdict = th.load('weights/20230712_164429_temp.pth',map_location='cpu')
net.load_state_dict(fdict['weights'])
net.eval()
ds = dataset.cocoDataSet(dataType='val')
for di in range(10):
    d = ds[di]
    x = d[0].reshape(1,3,160,160)
    # for i in d[1:]:
    #     y=i[:,:,:,4]
    #     idx = th.where(y>0.8)
    #     print(idx)
    #     print(y[idx])

    print('--------------------')

    img = np.array(d[0],dtype='uint8').transpose(1,2,0)

    ps = net(x)
    for sz,p in enumerate(ps):
        c=p[:,:,:,:,4]
        c=nn.functional.sigmoid(c)
        idx = th.where(c>0.9)
        if (il:=len(idx[0]))>0:
            cwhs=p[idx][:,:4]
            print(cwhs)
            for j in range(il):
                cwh=np.array(cwhs[j,:].detach()).flatten()
                ps_ind = int(idx[3][j])
                psc = np.array(ds.cfg['pre_scale'][sz][ps_ind])/416
                xywh = dataset.utils.tcwh2xywh(cwh,sz,int(idx[1][j]),int(idx[2][j]),psc[0],psc[1],[5,10,20])
                dataset.utils.drawLabel(img,[[0,xywh]])
