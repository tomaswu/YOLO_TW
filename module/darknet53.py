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
        self.cv_1_0 = layer.Convolutional(3,32,3,1,1,'bn','leaky') # 416x416x32
        self.cv_1_1 = layer.Convolutional(32,64,3,2,1,'bn','leaky') # 208x208x64
        # b1*1
        self.b_1 = nn.Sequential(
            layer.Convolutional(64,32,1,1,0,'bn','leaky'), # 208
            layer.Convolutional(32,64,3,1,1,'bn','leaky'), # 208
            layer.Residual_block(64,64,32) # 208
        )

        # 2nd
        self.cv_2_0 = layer.Convolutional(64,128,3,2,1,'bn','leaky') # 104
        self._b_2=nn.Sequential(
            layer.Convolutional(128,64,1,1,0,'bn','leaky'), # 104
            layer.Convolutional(64,128,3,1,1,'bn','leaky'), #104
            layer.Residual_block(128,128,64) # 104
        )
        self.b_2 = nn.Sequential(*([self._b_2]*2))

        # 3rd
        self.cv_3_0 = layer.Convolutional(128,256,3,2,1,'bn','leaky') # 52x52x256
        self._b_3 = nn.Sequential(
            layer.Convolutional(256,128,1,1,0,'bn','leaky'),
            layer.Convolutional(128,256,3,1,1,'bn','leaky'),
            layer.Residual_block(256,256,128) # 52
        )
        self.b_3 = nn.Sequential(*([self._b_3]*8)) #52x52x256  small cat

        # 4th
        self.cv_4_0 = layer.Convolutional(256,512,3,2,1,'bn','leaky') # 26
        self._b_4 = nn.Sequential(
            layer.Convolutional(512,256,1,1,0,'bn','leaky'),
            layer.Convolutional(256,512,3,1,1,'bn','leaky'),
            layer.Residual_block(512,512,256) # 26
        )
        self.b_4 = nn.Sequential(*([self._b_4]*8)) #26x26x512 middle cat

        # 5th
        self.cv_5_0 = layer.Convolutional(512,1024,3,2,1,'bn','leaky') # 13x13x1024 
        self._b_5 = nn.Sequential(
            layer.Convolutional(1024,512,1,1,0,'bn','leaky'),
            layer.Convolutional(512,1024,3,1,1),
            layer.Residual_block(1024,1024,512)
        )
        self.b_5 = nn.Sequential(*([self._b_5]*4)) #13x13x1024 large concatenate

        # predect
        self.cs1 = layer.ConvolutionalSetLayer(1024,512) # 512x13x13
        self.dect13 = layer.DectLayer(512,1024)  ##output 13x13

        self.p_c_13 = layer.Convolutional(512,256,1,1,0)
        self.up13 = layer.SampleLayer(2) #26x26x256 after cat b_4 26x26x768
        self.cs2 = layer.ConvolutionalSetLayer(768,512) #26x26x512
        self.dect26 = layer.DectLayer(512,1024) ##! output 26x26
        
        self.p_c_26 = layer.Convolutional(512,256,1,1,0) # 26x26x256
        self.up26 = layer.SampleLayer(2) #52x52x256 after cat b_3 52x52x512
        
        self.cs3 = layer.ConvolutionalSetLayer(512,256)
        self.dect52 = layer.DectLayer(256,512)

        # self.avgpool = nn.AvgPool2d(3,1,1)
        # self.connected = nn.Conv2d(512,1000,3,1,1)
        # self.softmax = nn.Softmax(1000)


    def forward(self,x):
        x_1_0 = self.cv_1_0(x)
        x_1_1 = self.cv_1_1(x_1_0)
        x_1_2 = self.b_1(x_1_1)

        x_2_0 = self.cv_2_0(x_1_2)
        x_2_1 = self.b_2(x_2_0)

        x_3_0 = self.cv_3_0(x_2_1)
        x_3_1 = self.b_3(x_3_0)  # small 0

        x_4_0 = self.cv_4_0 (x_3_1)
        x_4_1 = self.b_4(x_4_0) # middle 0

        x_5_0 = self.cv_5_0(x_4_1)
        x_5_1 = self.b_5(x_5_0) # large

        x_13_0 = self.cs1(x_5_1)
        p13 = self.dect13(x_13_0)

        x_13_1 = self.p_c_13(x_13_0)
        x_13_2 = self.up13(x_13_1)
        x_13_3 = th.cat([x_13_2,x_4_1],1)

        x_26_0 = self.cs2(x_13_3)
        p26 = self.dect26(x_26_0)

        x_26_1 = self.p_c_26(x_26_0)
        x_26_2 = self.up26(x_26_1)
        x_26_3 = th.cat([x_26_2,x_3_1],1)
        
        x_52_0 = self.cs3(x_26_3)
        p52 = self.dect52(x_52_0)

        return p13.view(-1,13,13,3,85),p26.view(-1,26,26,3,85),p52.view(-1,52,52,3,85) 
    

if __name__=='__main__':
    x=th.rand(3,3,416,416)
    net=DARKNET53()
    p=net(x)
    print(p[1].shape)