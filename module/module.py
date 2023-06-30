# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/06/30 17:23:32
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import torch.nn as nn

class backbone(nn.Module):
    def __init__(self,mcfg) -> None:
        super().__init__()
        self.mcfg = mcfg #* module config
