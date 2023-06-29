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
            label.append([tidx,bbox])
        return img.transpose(2,0,1),label

    def __len__(self):
        return len(self.imageInfos)

    @property
    def cocoYear(self):
        return int(self.cfg['coco_dir'][-4:])
    
    def create_label(self,bboxes):
        w = self.cfg['output_size'][0]
        strides = w/self.gs
        labels = [np.zeros([strides[i],strides[i],3,1+4+self.classified_num],dtype=np.float32) for i in range(strides.shape[0])]
        for b in bboxes:
            #onehot
            onehot = np.zeros(self.classified_num,dtype=np.float32)
            onehot[b[0]] = 1
            xywh=np.array(b[1])/strides
            ...
            



if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    d = cocoDataSet()
    img,lb = d[4]
    utils.drawLabel(img.transpose(1,2,0),lb)
    # loader = data.DataLoader(d,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

    # for da in loader:
    #     print(da)