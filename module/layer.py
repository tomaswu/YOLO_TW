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
    
# 下采样层：既改变通道数，也改变形状
class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.Conv2d(out_channels,in_channels,3,1,1),
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.Conv2d(out_channels,in_channels,3,1,1),
            nn.Conv2d(in_channels,out_channels,1,1,0)
        )

    def forward(self, x):
        return self.conv(x)


# 采样层 
class SampleLayer(nn.Module):
    def __init__(self,scale_factor=2):
        super(SampleLayer, self).__init__()
        self.scale_factor=scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
    

class DectLayer(nn.Module):
    def __init__(self,in_channels,out_channels,N_classes=80):
        super().__init__()
        self.dect = nn.Sequential(
            Convolutional(in_channels,out_channels,3,1,1,'bn','leaky'),
            nn.Conv2d(out_channels,(N_classes+5)*3,1,1,0)
        )
    def forward(self,x):
        return self.dect(x)
    
class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSetLayer, self).__init__()
        self.sub_module = nn.Sequential(
            # 讲解中卷积神经网络的的5个卷积
            Convolutional(in_channels, out_channels, 1, 1, 0,'bn','leaky'),
            Convolutional(out_channels, in_channels, 3, 1, 1,'bn','leaky'),

            Convolutional(in_channels, out_channels, 1, 1, 0,'bn','leaky'),
            Convolutional(out_channels, in_channels, 3, 1, 1,'bn','leaky'),

            Convolutional(in_channels, out_channels, 1, 1, 0,'bn','leaky')
        )

    def forward(self, x):
        return self.sub_module(x)


if __name__=='__main__':
    a=th.rand(8,3,52,52)
    dl=DownSampleLayer(3,12)
    d=dl(a)
    print(d.shape)
    u=SampleLayer(0.5)(d)
    print(u.shape)