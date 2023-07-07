# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/07 13:57:56
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class LOSS(nn.Module):
    def __init__(self,l_coord=1,l_noobj=0.2,l_class=1):
        super().__init__()
        self.l_coord=l_coord
        self.l_noobj = l_noobj
        self.l_class = l_class
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.meanBCE = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self,pred,target):
        """cal loss of box
        Args:
            pred (tensor): b x [tx,ty,tw,th]
            target (tensor): x hat
        """
        assert len(pred.shape)==5,'pret input size error.'
        assert len(target.shape)==5,'target input size error.'
        assert pred.shape==target.shape,'shape of pred and target must be same.'
        batch_size = pred.shape[0]
        total_loss = self.lbox(pred,target)+self.lobj(pred,target)+self.lcls(pred,target)
        loss=total_loss/batch_size
        return loss
    
    def se(self,x,y):
        return th.pow(x-y,2)
    
    def lbox(self,pred,target):
        # ! tx=pred[:,:,:,:,:0] 0,1,2,3
        p_x=pred[:,:,:,:,0]        
        p_y=pred[:,:,:,:,1]        
        p_w=pred[:,:,:,:,2]        
        p_h=pred[:,:,:,:,3]
        t_x=target[:,:,:,:,0]        
        t_y=target[:,:,:,:,1]        
        t_w=target[:,:,:,:,2]        
        t_h=target[:,:,:,:,3]
        sex = self.se(p_x,t_y)
        sey = self.se(p_y,t_y)
        sew = self.se(th.sqrt(p_w),th.sqrt(t_w))
        seh = self.se(th.sqrt(p_h),th.sqrt(t_h))
        total_se = sex+sey+sew+seh
        I_obj = th.round(target[:,:,:,:,4])
        loss_vec = I_obj*total_se
        print(I_obj.max(),sew[0,0,0,0],sex[0,0,0,0])
        loss = th.sum(loss_vec)
        return loss

    def lobj(self,pred,target):
        p_c = pred[:,:,:,:,4]
        t_c = target[:,:,:,:,4]
        I_obj = th.round(t_c)
        I_noobj = 1-I_obj
        bce_all = self.BCE(p_c,t_c)
        loss_pobj = th.sum(I_obj*bce_all)
        loss_nobj = th.sum(self.l_noobj *I_noobj*bce_all)
        return loss_pobj+loss_nobj
    
    def lcls(self,pred,target):
        p_p = pred[:,:,:,:,5:]
        t_p = target[:,:,:,:,5:]
        bce_all = self.BCE(p_p,t_p)
        bce_sum = th.sum(bce_all,4)
        t_c = target[:,:,:,:,4]
        I_obj = th.round(t_c)
        loss_cls = th.sum(I_obj*bce_sum)
        return loss_cls


if __name__=='__main__':
    import sys
    sys.path.append('./')
    import dataset.utils as utils
    #* test lbox
    x=th.zeros(2,2,2,3,85)+0.65
    y=x-0.3
    y[0,0,0,0,4] = 0.8
    lf=LOSS(l_noobj=1)
    # print(lf.lbox(x,y))
    
    #* test lobj
    x[0,0,0,0,4] = utils.sigmod_T(0.8)
    # print(lf.lobj(x,y))

    #* test lcls
    x[0,0,0,0,5:] = th.zeros(80)+utils.sigmod_T(1e-3)
    y[0,0,0,0,5:] = th.zeros(80)+1e-3
    y[0,0,0,0,55] = 0.95
    x[0,0,0,0,55] = utils.sigmod_T(0.95)
    print(x[0,0,0,0])
    print(y[0,0,0,0])
    print(lf.lcls(x,y))