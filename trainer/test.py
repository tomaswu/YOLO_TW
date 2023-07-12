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

net = module.yolov3.YOLOV3()

fdict = th.load('weights/20230712_004322_20_temp.pth',map_location='cpu')
net.load_state_dict(fdict['weights'])
net.eval()
ds = dataset.cocoDataSet(dataType='val')
d = ds[0]
x = d[0].reshape(1,3,416,416)
for i in d[1:]:
    y=i[:,:,:,4]
    idx = th.where(y>0.8)
    print(idx)
    print(y[idx])

print('--------------------')

ps = net(x)
for p in ps:

    c=p[:,:,:,:,4]
    c=nn.functional.sigmoid(c)
    idx = th.where(c>0.1)
    print(idx,c[idx])

img = np.array(d[0],dtype='uint8').transpose(1,2,0)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()