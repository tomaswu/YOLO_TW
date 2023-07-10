# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/06 11:19:36
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import importlib

all=[]

# COCO = importlib.import_module(f'{__name__}.twData').cocoDataSet
# utils = importlib.import_module(f'{__name__}.utils')
# config = importlib.import_module(f'{__name__}.config')
from .twData import *
from . import utils
from . import config

