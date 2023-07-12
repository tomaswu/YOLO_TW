# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/10 11:56:09
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

import numpy as np
import cv2
import time,os
import torch.utils.data as tud
import torch as th
import torch.nn as nn
import torch.utils.tensorboard as tb

def log(*args,**kwargs):
    print(*args,**kwargs)

class Trainer():
    def __init__(self) -> None:
        self.cfg = config.load()
        self.device = self._setDevice()
        log(f'set device {self.cfg["device"]},using {self.device}!!!')
        self._setData()
        self._setModule()
        self._loadWeight()

        # 打开tensorboard
        # cmd: tensorboard --logdir=logs --port=6007
        self.summary_writer = tb.SummaryWriter(log_dir='./logs')

    def _setDevice(self):
        match self.cfg['device']:
            case 'cuda':
                if th.cuda.is_available():
                    return  th.device('cuda')
                else:
                    raise Exception('cant find cuda device.')
            case 'mps':
                if th.has_mps:
                    return th.device('mps')
                else:
                    raise Exception('cant find mps device.')
            case 'auto':
                if th.cuda.is_available():
                    return  th.device('cuda')
                elif th.has_mps:
                    return th.device('mps')
                else:
                    return th.device('cpu')
            case _:
                return th.device('cpu')

    def _setData(self):
        log('loading data...')
        self.train_data = dataset.twData.cocoDataSet('val')
        # self.val_data = dataset.twData.cocoDataSet('val')
        # log(f'using data: train {len(self.train_data)} val:{len(self.val_data)}')

    def _setModule(self):
        self.net = module.yolov3.YOLOV3(img_size=self.train_data.cfg['output_size'][0])
        self.loss_fn = module.loss.LOSS()
        lr = self._getConfigLr()
        self.optimer = th.optim.Adam(self.net.parameters(),lr=lr)
        log(f'using learning rate:{lr}')
    
    def _getConfigLr(self):
        return float(self.cfg['lr'])
    
    def _saveWeight(self,comment:str=''):
        if self.cfg['save_file']=='auto':
            fname=time.strftime('%Y%m%d_%H%M%S')+f'_{comment}.pth'
        else:
            fname=self.cfg['save_file']
        fpath = os.path.join('./weights',fname)
        log(fpath)
        th.save({'weights':self.net.state_dict()},fpath)
    
    def _loadWeight(self):
        path = self.cfg['load_file']
        if os.path.isfile(path):
            fdict = th.load(path,map_location='cpu')
            self.net.load_state_dict(fdict['weights'])
            log(f'loaded module {path}!!')

    def train(self):
        epoch_count = self.cfg['total_epoch']
        bz = self.cfg['batch_size']
        self.td_loder = tud.DataLoader(dataset=self.train_data,batch_size=bz,shuffle=False,num_workers=0,drop_last=False)
        dl = len(self.train_data)
        self.net.to(self.device)
        self.loss_fn.to(self.device)
        self.net.train()
        t0=time.time()
        for epoch in range(epoch_count):
            for count,data in enumerate(self.td_loder):
                x,y_l,y_m,y_s = data
                # img = np.array(x[0],dtype='uint8').transpose(1,2,0)
                # cv2.imshow('test',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                x=x.to(self.device)
                y_l=y_l.to(self.device)
                y_m=y_m.to(self.device)
                y_s=y_s.to(self.device)
                for i in range(200):
                    self.net.zero_grad()
                    p_l,p_m,p_s = self.net(x)
                    loss_l,loss_l_box,loss_l_obj,loss_l_cls = self.loss_fn(p_l,y_l)
                    loss_m,loss_m_box,loss_m_obj,loss_m_cls = self.loss_fn(p_m,y_m)
                    loss_s,loss_s_box,loss_s_obj,loss_s_cls = self.loss_fn(p_s,y_s)
                    loss = loss_l + loss_m + loss_s
                    loss.backward()
                    self.optimer.step()
                    print(f'----------{time.time()-t0:.3f}_epoch&count:{epoch}_{count}_{i}_{(count+1)*bz}/{dl}-----------')
                    print(f'loss:{loss:<8.5f} loss13:{loss_l:<8.2f} loss26:{loss_m:<8.2f} loss52:{loss_s:<8.2f}')
                    print(f'box:{loss_l_box:<8.2f} {loss_m_box:<8.2f} {loss_s_box:<8.2f}')
                    print(f'obj:{loss_l_obj:<8.2f} {loss_m_obj:<8.2f} {loss_s_obj:<8.2f}')
                    print(f'cls:{loss_l_cls:<8.2f} {loss_m_cls:<8.2f} {loss_s_cls:<8.2f}')
                    t0=time.time()
                self._saveWeight('temp')
                return
                self.summary_writer.add_scalar('loss-imgs',bz*(count+1)*(epoch+1),loss)
                self.summary_writer.add_scalar('loss_large-imgs',bz*(count+1)*(epoch+1),loss_l)
                self.summary_writer.add_scalar('loss_middle-imgs',bz*(count+1)*(epoch+1),loss_m)
                self.summary_writer.add_scalar('loss_small-imgs',bz*(count+1)*(epoch+1),loss_s)
                if count >0 and count%self.cfg['save_per_count']==0:
                    self._saveWeight(f'{epoch_count}')


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import shutil
    try:
        shutil.rmtree('./logs')
    except:
        ...
    tr = Trainer()
    tr.train()