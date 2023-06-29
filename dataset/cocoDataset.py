# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/06/29 13:33:42
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import numpy as np
import pycocotools.coco as coco
import config
import os

dataType = 'val2014'
cfg=config.load()

af = os.path.join(cfg['coco_dir'], f'annotations/instances_{dataType}.json')

c=coco.COCO(af)

print(c.dataset.keys())
id=c.dataset['images'][0]['id']

info=c.loadImgs(id)
aid = c.getAnnIds(id)
am=[]
for i,j in enumerate(c.cats.keys()):
    am.append(c.cats[j]['name']) 
print(am)