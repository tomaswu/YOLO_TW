# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/06/29 11:14:38
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import numpy as np
import cv2
import time

def TimeTest(f):
    def fc(*args,**kwargs):
        t0=time.time()
        r=f(*args,**kwargs)
        print(f'TIMETEST::{f.__name__} spent:{(time.time()-t0)*1000:.3f} ms')
        return r
    return fc

def padding(img,output_size,gray=False):
    '''img: ndarray dims=row,col,rgb
       output_size:w,h'''
    w,h=output_size
    if gray and len(img.shape)!=2:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if (not gray) and len(img.shape)==2:
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    csize = img.shape
    if len(csize)==2:
        c=0
    elif len(csize)==3:
        c=csize[2]
    else:
        raise Exception("img channels must be 1 or 3")
    
    ch,cw=csize[:2]
    
    if cw>w or ch>h:
        if cw>ch:
            k=w/cw
        else:
            k=h/ch
    else:
        k=1

    nw = int(cw*k)
    nh = int(ch*k)

    if c==0:
        nimg = np.zeros([h,w],dtype=np.uint8)
    else:
        nimg = np.zeros([h,w,c],dtype=np.uint8)

    rimg = cv2.resize(img,[nw,nh])
    nimg[:nh,:nw]=rimg
    return nimg,k

def drawLabel(img,label,t=0):
    if len(img.shape)==2:
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for i in label:
        idx,b=i
        if (b<1).all():
            b=np.array(b)
            b[0]=b[0]*img.shape[1]
            b[1]=b[1]*img.shape[0]
            b[2]=b[2]*img.shape[1]
            b[3]=b[3]*img.shape[0]
        b=[int(j) for j in b]
        cv2.rectangle(img,b,[0,255,0],1,cv2.LINE_AA)
        cv2.putText(img,f'{idx}',(b[0],b[1]),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0,0,255],1)
    cv2.imshow('test',img)
    cv2.waitKey(t)

def iouByBbox(b1,b2):
    """calulate iou of two bbox

    Args:
        b1 (array): bbox with x,y,w,h
        b2 (array): bbox with x,y,w,h

    Returns:
        flaot: iou of b1 and b2
    """    
    #A
    x1,y1,wa,ha=b1
    x2=x1+wa
    y2=y1+ha
    #B
    x3,y3,wb,hb=b2
    x4=x3+wb
    y4=y3+hb

    xs=[x1,x2,x3,x4]
    ys=[y1,y2,y3,y4]
    xs.sort()
    ys.sort()
    if (x1==xs[0] and x2==xs[1]) or (x1==xs[3] and x2==xs[4]) or (y1==ys[0] and y2==ys[1]) or (y1==ys[3] and y2==ys[4]):
        i=0
    else:
        i=(xs[1]-xs[2])*(ys[1]-ys[2])
    u=wa*ha+wb*hb-i
    return i/u

def t_cwh2anchors_cwh(t,pre_scale):
    tx,ty,tw,th=t
    cx,cy,pw,ph=pre_scale
    return np.array([sigmod(tx)+cx,sigmod(ty)+cy,pw*np.exp(tw),ph*np.exp(th)])

def anchors_cwh2t_cwh(achors_xywh,pre_scale):
    cx,cy,pw,ph=pre_scale
    x,y,w,h=achors_xywh
    # print(x-cx,y-cy)
    return np.array([x-cx,y-cy,np.log(w/pw),np.log(h/ph)])  # not using sigmod_T ,add it to loss func

def cwh2xywh(cwh):
    cx,cy,w,h=cwh
    return np.array([cx-w/2,cy-h/2,w,h])

def xywh2cwh(xywh):
    x,y,w,h=xywh
    return np.array([x+w/2,y+h/2,w,h])

def tcwh2xywh(tcwh,size,xind,yind,pw,ph):
    """decode t_center_w_h to xywh
    Args:
        tcwh (tensor): list of tcwh(yolo)
        size (int): one of [0,1,2],means small,middle,large
        xind (ind): index of x gird
        yind (ind): index of y gird
        pw (float): normed prescale width
        ph (float): normed prescale height
    Returns:
        tensor of normed xywh 
    """    
    gs=[13,26,52]
    g=gs[size]
    p=[xind/g,yind/g,pw*g,ph*g]
    a=t_cwh2anchors_cwh(tcwh,p)
    return cwh2xywh(a)

def onehot_smooth(onehot,N_classes=-1,hyper_parameter = 0.01):
    if N_classes<=0:
        N_classes=len(onehot.flatten())
    osm = onehot * (1 - hyper_parameter) + hyper_parameter * 1.0 / N_classes
    return osm

def sigmod(x):
    return 1/(1+np.exp(-x))

def sigmod_T(y):
    # if y==1:
    #     y=1-1e-28
    return np.log(y/(1-y))

if __name__=='__main__':
    # import matplotlib.pyplot as plt
    # x=np.arange(100)
    # onehot = np.zeros(100)
    # onehot[77]=1
    # os = onehot_smooth(onehot)
    # plt.figure()
    # # plt.plot(x,onehot,marker='o')
    # plt.plot(x,os)
    # plt.show()

    # import torch as th
    # t_cwh=th.tensor([0.00955168,  0.01432752, -3.51035784, -3.71039719]*3)
    # b = tcwh2xywh(t_cwh,2,23,15,156/416,198/416)
    # print(b)

    print(onehot_smooth(0,2))



    


