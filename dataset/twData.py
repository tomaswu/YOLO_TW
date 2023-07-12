# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/01/13 10:17:17
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch.utils.data as data
import PIL.Image as Image
import torch as th
import os,json
import numpy as np
from pycocotools.coco import COCO
try:
    from . import config
    from . import utils
except:
    import config
    import utils

class cocoDataSet(data.Dataset):
    def __init__(self,dataType = 'val',classified_num=80) -> None:
        super().__init__()
        
        self.classified_num=classified_num
        self.dataType=dataType
        self.cfg=config.load()
        iz = self.cfg['output_size'][0]
        self.gs=np.array([iz//32,iz//16,iz//8])
        af = os.path.join(self.cfg['coco_dir'], f'annotations/instances_{dataType}{self.cocoYear}.json')
        self.coco=COCO(af)
        self.datasetSize=len(self.coco.dataset['images'])

    def __getitem__(self,index):
        imgIdx = self.coco.dataset['images'][index]['id']
        fname = self.coco.loadImgs(imgIdx)[0]['file_name']
        fpath = os.path.join(self.cfg['coco_dir'],f'{self.dataType}{self.cocoYear}',fname)
        img = Image.open(fpath)
        img,k=utils.padding(np.array(img),self.cfg['output_size'],self.cfg['output_gray'])
        if self.cfg['output_gray']:
            img.reshape(img.shape[0],img.shape[1],1)
        annIdx = self.coco.getAnnIds(imgIdx)
        ann=self.coco.loadAnns(annIdx)
        label=[]
        for i in ann:
            cid=i['category_id']
            cname = self.coco.cats[cid]['name']
            if cname not in self.cfg['category']:
                print('skip cname:{cname}')
                continue
            tidx = self.cfg['category'].index(cname)
            bbox =i['bbox'] if k==1 else [tmp*k for tmp in i['bbox']]
            label.append([tidx]+list(np.array(bbox)/self.cfg['output_size'][0]))
        ylabel=self.create_ylabel(label)
        imgs=th.tensor(img.transpose(2,0,1),dtype=th.float)
        label_large,label_middle,label_small = [th.tensor(i,dtype=th.float) for i in ylabel]
        # labels = th.tensor(label,dtype=float).reshape(-1,5)
        return imgs,label_large,label_middle,label_small#, labels# reutrn 12,26,52

    def __len__(self):
        return self.datasetSize

    @property
    def cocoYear(self):
        return int(self.cfg['coco_dir'][-4:])
    
    # @utils.TimeTest
    def create_ylabel(self,bboxes):
        w = self.cfg['output_size'][0]
        gs=self.gs
        strides = (w/gs).astype(int)
        # print('strides',strides)
        # * lms_dect * [xind,yind,pre_scale,confidence+tbox+onehot]
        isobj = utils.onehot_smooth(1.0,2)
        isnoobj = utils.onehot_smooth(0.0,2)
        labels = [np.zeros([gs[i],gs[i],3,1+4+self.classified_num],dtype=np.float32)+isnoobj for i in range(strides.shape[0])]
        for b in bboxes:
            #onehot
            onehot = np.zeros(self.classified_num,dtype=np.float32)
            onehot[int(b[0])] = 1
            onehot = utils.onehot_smooth(onehot,self.classified_num)
            best_dectind=0
            best_xind=0
            best_yind=0
            best_pre_scale_ind=0
            iou_history=0
            t_cwh=None
            #对应应该选为目标的cx,cy
            xywh=np.array(b[1:])
            cx,cy = utils.xywh2cwh(xywh)[:2]
            for dect_size,g in enumerate(gs):    
                for prescale_idx in range(len(self.cfg['pre_scale'][0])):
                    pw,ph = np.array(self.cfg['pre_scale'][dect_size][prescale_idx])/416
                    p_xywh=utils.cwh2xywh([cx,cy,pw,ph])
                    iou=utils.iouByBbox(xywh,p_xywh)
                    if iou>iou_history:
                        dectind=dect_size
                        xind=int(cx*g)
                        yind=int(cy*g)
                        pre_scale_ind=prescale_idx
                        if labels[dectind][xind,yind,pre_scale_ind,4]!=isobj:
                            best_dectind=dectind
                            best_xind=xind
                            best_yind=yind
                            best_pre_scale_ind=pre_scale_ind
                            iou_history=iou
                            ixypwh = [xind/gs[dectind],yind/gs[dectind],pw*gs[dectind],ph*gs[dectind]]
                            t_cwh = utils.anchors_cwh2t_cwh(utils.xywh2cwh(xywh),ixypwh)
            if t_cwh is not None:
                labels[best_dectind][best_xind,best_yind,best_pre_scale_ind,4]=isobj
                labels[best_dectind][best_xind,best_yind,best_pre_scale_ind,0:4]=t_cwh
                # print(t_cwh,best_xind,best_yind,best_pre_scale_ind,best_dectind)
            else:
                print('miss a label',b)
        return labels


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    d = cocoDataSet()

    img,ys,ym,yl= d[0]
    t_cwh=[-3.70307667, -3.87953649, -2.5584004,  -2.61853048]
    dect = 1
    pre_scale = 2
    x_ind=10
    y_ind=6
    s = np.array(d.cfg['pre_scale'][dect][pre_scale])/416
    b = utils.tcwh2xywh(t_cwh,dect,x_ind,y_ind,s[0],s[1],[8,16,32])
    print(b)
    utils.drawLabel(np.array(img,dtype='uint8').transpose(1,2,0),[[0,b]])

    # loader = data.DataLoader(d,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
    # i=0
    # for da in loader:
    #     print('da:',i,da[0].shape)
    #     i+=1