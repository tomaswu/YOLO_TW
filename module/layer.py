# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/04 14:05:22
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """Mish activate
    f(x) = x*tanh(ln(1+e^x))
    """    
    def __init__(self):
        super(Mish).__init__()
    def forward(self, x):
        x = x * (th.tanh(F.softplus(x)))
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "mish": Mish,# it can be replaced by nn.Mish
    "swish":Swish}

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class Residual_block(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium):
        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0,
                                     norm="bn", activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1,
                                     norm="bn", activate="leaky")
    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r
        return out