# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/06 16:29:47
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''


import torch as th
import torch.nn as nn

try:
    import layer
    import head

except:
    from .. import layer
    from .. import head

class DARKNET53(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st
        self.cv_1_0 = layer.Convolutional(3,32,3,1,1,'bn','leaky')
        self.cv_1_1 = layer.Convolutional(32,64,3,2,1,'bn','leaky')
        # b1*1
        self.b_1 = nn.Sequential(
            layer.Convolutional(32,32,1,1,1,'bn','leaky'),
            layer.Convolutional(32,64,3,1,1,'bn','leaky'),
            layer.Residual_block(64,64,32)
        )

        # 2nd
        self.cv_2_0 = layer.Convolutional(64,128,3,2,1,'bn','leaky')
        self.b_2=nn.Sequential(
            layer.Convolutional(64,64,1,1,1,'bn','leaky'),
            layer.Convolutional(64,128,3,1,1,'bn','leaky'),
            layer.Residual_block(128,64,32)
        )

        # 3rd
        self.cv_3_0 = layer.Convolutional(64,256,1,1,1,'bn','leaky')