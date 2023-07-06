# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/06 11:21:50
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''
import sys
sys.path.append('.')
import dataset

if __name__=='__main__':
    data = dataset.COCO()
    print(data[0])