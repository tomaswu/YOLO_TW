# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/07/11 09:13:38
    @Author  :   Tomas
    @Version :   1.0
    @Contact :   tomaswu@qq.com
    Desc     :    
'''

def precision(t_p,p_all):
    """cal precison
    Args:
        t_p (float): positive count of prediction
        p_all (float): count of prediction
    Returns:
        float: precision
    """    
    return t_p/p_all

def recall(t_p,t_all):
    """cal recall
    Args:
        t_p (float): positive count of prediction
        t_all (float): count of all positive samples
    Returns:
        float: recall
    """    
    return t_p/t_all

def map(pr,re):
    return pr*re