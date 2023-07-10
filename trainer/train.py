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
        # self.train_data = dataset.twData.cocoDataSet('train')
        self.val_data = dataset.twData.cocoDataSet('val')
        # log(f'using data: train {len(self.train_data)} val:{len(self.val_data)}')

    def _setModule(self):
        self.net = module.darknet53.DARKNET53()
        self.loss_fn = module.loss.LOSS()
        lr = self._getConfigLr()
        self.optimer = th.optim.Adam(self.net.parameters(),lr=lr)
        log(f'using learning rate:{lr}')
    
    def _getConfigLr(self):
        return float(self.cfg['lr'])
    
    def _saveWeight(self):
        ...
    
    def _loadWeight(self):
        ...

    def train(self):
        epoch_count = self.cfg['total_epoch']
        self.td_loder = tud.DataLoader(dataset=self.val_data,batch_size=2,shuffle=True,num_workers=0,drop_last=False)
        for epoch in range(epoch_count):
            for count,data in enumerate(self.td_loder):
                x,y13,y26,y52 = data
                x=x.to(self.device)
                y13=y13.to(self.device)
                y26=y26.to(self.device)
                y52=y52.to(self.device)
                self.net.zero_grad()
                p13,p26,p52 = self.net(x)
                loss13 = self.loss_fn(p13,y13)
                loss26= self.loss_fn(p26,y26)
                loss52 = self.loss_fn(p52,y52)
                loss = loss13 + loss26 + loss52
                loss.backward()
                self.optimer.step()
                self.summary_writer.add_scalar('loss')
                print(f'epoch&count:{epoch}_{count},loss:{loss:.5f}')
                # return



if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    tr = Trainer()
    tr.train()