# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/06 11:14:05
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import sys
sys.path.append('.')

import torch as th
import torch.nn as nn
import dataset.utils as dutils
import dataset.config as dconfig
import numpy as np

class HEAD(nn.Module):
    def __init__(self,cof_thershold=0.5):
        super().__init__()
        self.cfg=dconfig.load()
        self.cof_thershold=cof_thershold
    
    def forward(self,p):
        small,middle,large=p
        s=self.decode(small)
        m=self.decode(middle)
        l=self.decode(large)
        return p,th.tensor(s+m+l).reshape(-1,6)
        
    def decode(self,p):
        """decode prediction results
        Args:
            p (tensor): 13x13x3x85
        """
        dect_size=int(np.log(p.shape[0]/13)/np.log(2))      
        idx=th.where(p[:,:,:,0]>self.cof_thershold)
        r=[]
        for i in range(len(idx[0])):
            prescale = self.cfg['pre_scale'][dect_size][idx[2][i]]
            pw=prescale[0]/416
            ph=prescale[1]/416
            x_ind=idx[0][i]
            y_ind=idx[1][i]
            tcwh=p[x_ind,y_ind,idx[2][i],1:5]
            xywh = dutils.tcwh2xywh(tcwh,dect_size,x_ind,y_ind,pw,ph)
            confidence = p[x_ind,y_ind,idx[2][i],0]
            lix=int(p[x_ind,y_ind,idx[2][i],5:].argmax())
            r.append(list(xywh)+[lix]+[confidence])
        return r

if __name__=='__main__':
    p1=th.zeros([52,52,3,85])
    t_cwh=th.tensor([0.00955168,  0.01432752, -3.51035784, -3.71039719])
    p1[23,15,2,0]=0.87
    p1[23,15,2,1:5]=t_cwh
    p1[23,15,2,66]=0.8
    head=HEAD()
    r=head([p1]*3)[1]
    print(r)
