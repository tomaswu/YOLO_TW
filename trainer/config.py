# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/06/29 10:39:50
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

import yaml
import os

fdir = os.path.dirname(__file__)

def load(fpath=''):
    if not fpath:
        fpath = os.path.join(fdir,'config.yaml')
    if os.path.isfile(fpath):
        with open(fpath,'rb') as f:
            d=yaml.load(f,yaml.FullLoader)
            if d['name']=='training config':
                return d
            raise Exception('config name error!')
    raise Exception('config path not exist!')

def dump(dic=None,fpath=''):
    if not fpath:
        fpath = os.path.join(fdir,'config.yaml')
    if isinstance(dic,dict):
        with open(fpath,'w') as t:
            yaml.dump_all(dic,t)
    else:
        raise TypeError('content must be dict!!')

if __name__=='__main__':
    c=load()
    print(c)
                