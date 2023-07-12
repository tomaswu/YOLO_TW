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
    from . import layer
    from . import head

class YOLOV3(nn.Module):
    def __init__(self,img_size=256):
        super().__init__()
        assert img_size%32==0,'img_size must be 32*N'
        self.img_size = img_size
        # pp
        self.pp = layer.Convolutional(3,32,3,1,1,'bn','leaky') # sizex32 size = img_size
        # 1st
        self.ro1 = layer.ResOperator(32,64,1) # size/2 x64
        # 2nd
        self.ro2 = layer.ResOperator(64,128,2) #size/4 x128
        # 3rd
        self.ro3 = layer.ResOperator(128,256,8) #size/8 x256  small cat
        # 4th
        self.ro4 = layer.ResOperator(256,512,8) #size/16 x512 middle cat
        # 5th
        self.ro5 = layer.ResOperator(512,1024,4) #size/32 x1024 large concatenate
        # ==================end darknet 53================================
        # predect
        self.cs1 = layer.ConvolutionalSetLayer(1024,512) # 512x size/32
        self.dect_large = layer.DectLayer(512,1024)  ##output size/32 x255

        self.p_c_large = layer.Convolutional(512,256,1,1,0)
        self.up_large = layer.SampleLayer(2) #size/16 x256 after cat b_4 size/16 x768
        self.cs2 = layer.ConvolutionalSetLayer(768,512) #size/8 x512
        self.dect_middle = layer.DectLayer(512,1024) ##! output middlexmiddle
        
        self.p_c_middle = layer.Convolutional(512,256,1,1,0) # size/8 x256
        self.up_middle = layer.SampleLayer(2) #size/4 x256 after cat b_3 largexlargex512
        
        self.cs3 = layer.ConvolutionalSetLayer(512,256)
        self.dect_small = layer.DectLayer(256,512) # size/8 x255

    def forward(self,x):
        xpp = self.pp(x)
        x1 = self.ro1(xpp)
        x2 = self.ro2(x1)
        x3 = self.ro3(x2)  # small 0
        x4 = self.ro4(x3) # middle 0
        x5 = self.ro5(x4) # large

        x_large_0 = self.cs1(x5)
        p_large = self.dect_large(x_large_0)

        x_large_1 = self.p_c_large(x_large_0)
        x_large_2 = self.up_large(x_large_1)
        x_large_3 = th.cat([x_large_2,x4],1)

        x_middle_0 = self.cs2(x_large_3)
        p_middle = self.dect_middle(x_middle_0)

        x_middle_1 = self.p_c_middle(x_middle_0)
        x_middle_2 = self.up_middle(x_middle_1)
        x_middle_3 = th.cat([x_middle_2,x3],1)
        
        x_small_0 = self.cs3(x_middle_3)
        p_small = self.dect_small(x_small_0)

        ls,ms,ss = self.img_size//32,self.img_size//16,self.img_size//8

        return p_large.view(-1,ls,ls,3,85),p_middle.view(-1,ms,ms,3,85),p_small.view(-1,ss,ss,3,85)
    

if __name__=='__main__':
    x=th.rand(3,3,256,256)
    net=YOLOV3()
    p=net(x)
    print(p[1].shape)