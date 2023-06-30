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
import config
import utils
from pycocotools.coco import COCO

class cocoDataSet(data.Dataset):
    def __init__(self,dataType = 'val',gird_size=[13,26,52],classified_num=80) -> None:
        super().__init__()
        self.gs=np.array(gird_size)
        self.classified_num=classified_num
        self.dataType=dataType
        self.cfg=config.load()
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
            label.append([tidx,np.array(bbox)/self.cfg['output_size'][0]])
        ylabel=self.create_ylabel(label)
        return img.transpose(2,0,1),ylabel[0],ylabel[1],ylabel[2]

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
        labels = [np.zeros([gs[i],gs[i],3,1+4+self.classified_num],dtype=np.float32) for i in range(strides.shape[0])]
        for b in bboxes:
            #onehot
            onehot = np.zeros(self.classified_num,dtype=np.float32)
            onehot[b[0]] = 1
            best_dectind=0
            best_xind=0
            best_yind=0
            best_pre_scale_ind=0
            iou_history=0
            t_cwh=None
            #对应应该选为目标的cx,cy
            xywh=np.array(b[1])
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
                        if labels[dectind][xind,yind,pre_scale_ind,0]!=1.0:
                            best_dectind=dectind
                            best_xind=xind
                            best_yind=yind
                            best_pre_scale_ind=pre_scale_ind
                            iou_history=iou
                            ixypwh = [xind/gs[dectind],yind/gs[dectind],pw*gs[dectind],ph*gs[dectind]]
                            t_cwh = utils.anchors_cwh2t_cwh(utils.xywh2cwh(xywh),ixypwh)
            if t_cwh is not None:
                labels[best_dectind][best_xind,best_yind,best_pre_scale_ind,0]=1.0
                labels[best_dectind][best_xind,best_yind,best_pre_scale_ind,1:5]=t_cwh
            else:
                print('miss a label',b)
        return labels


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    d = cocoDataSet()

    # img,ys,ym,yl = d[55]
    # t_cwh=[0.00955168,  0.01432752, -3.51035784, -3.71039719]
    # b = utils.tcwh2xywh(t_cwh,2,23,15,156/416,198/416)
    # utils.drawLabel(img.transpose(1,2,0),[[0,b]])

    loader = data.DataLoader(d,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
    i=0
    for da in loader:
        print('da:',i,da[1].shape)
        i+=1