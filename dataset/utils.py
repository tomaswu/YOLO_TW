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
        b=[int(j) for j in b]
        cv2.rectangle(img,b,[0,255,0],1,cv2.LINE_AA)
        cv2.putText(img,f'{idx}',(b[0],b[1]),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0,0,255],1)
    cv2.imshow('test',img)
    cv2.waitKey(t)

def iouByBbox(b1,b2):
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


if __name__=='__main__':
    print(iouByBbox([5,5,3,3],[6,7,5,5]))




    


