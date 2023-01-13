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

class cocoDataSet(data.Dataset):
    def __init__(self,id='train') -> None:
        super().__init__()
        self.__id=id
        self.imageInfos = self.setImgPath()

    def __getitem__(self,index):
        imgBaseName = self.imageInfos[index]['file_name']
        imgFilePath = os.path.join(self.cocoDir,imgBaseName)
        img = Image.open(imgFilePath)
        labelFilePath = os.path.join(self.cocoLabelDir,imgBaseName.split('.')[0]+'.txt')
        if(os.path.isfile(labelFilePath)):
            with open(labelFilePath,'r') as f:
                label=[]
                for i in f.readlines():
                    m=i.split(' ')
                    tmp=[float(m[k]) for k in range(5)]

                label=np.array(label)
        else:
            label=[]
        return np.array(img),label,imgBaseName

    def __len__(self):
        return len(self.imageInfos)

    @property
    def cocoRoot(self):
        name = os.popen("hostname").read()
        match name:
            case 'DESKTOP-0QRHBN4\n':
                return r'E:\coco2014'
            case _:
                raise Exception("coco path not set!!")

    @property
    def cocoYear(self):
        return int(self.cocoRoot[-4:])

    @property
    def cocoDir(self):
        if self.__id=='train':
            dr = f'train{self.cocoYear}'
        elif self.__id=='test' or self.__id=='val':
            dr = f'val{self.cocoYear}'
        else:
            raise Exception('unknow dataset name!!!')
        return os.path.join(self.cocoRoot,dr)
    
    @property
    def cocoLabelDir(self):
        if self.__id=='train':
            dr = f'train{self.cocoYear}'
        elif self.__id=='test' or self.__id=='val':
            dr = f'val{self.cocoYear}'
        else:
            raise Exception('unknow dataset name!!!')
        return os.path.abspath(f'{self.cocoRoot}/labels/{dr}')

    def setImgPath(self):
        if self.__id=='train':
            dr = f'train{self.cocoYear}'
        elif self.__id=='test' or self.__id=='val':
            dr = f'val{self.cocoYear}'
        else:
            raise Exception('unknow dataset name!!!')
        file = self.cocoRoot+f'/annotations/captions_{dr}.json'
        with open(file,'r') as f:
            fdr = json.load(f)
        return fdr['images']

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    d = cocoDataSet()
    img,lb,name = d[4]
    img=Image.fromarray(img)
    img.show()

    loader = test_loader = data.DataLoader(d,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

    for da in loader:
        print(da)