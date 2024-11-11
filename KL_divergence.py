#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   KL_divergence.py
@Time    :   2024/06/18 14:39:56
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   KL散度
'''

import numpy as np

def kl_divergence(p, q):
    """
    计算两个离散概率分布p和q之间的KL散度。
    
    参数:
    p -- 第一个概率分布，一个包含非负值的NumPy数组，其元素之和为1。
    q -- 第二个概率分布，一个包含非负值的NumPy数组，其元素之和为1。
    
    返回:
    KL散度的值。
    """
    # 确保p和q是有效的概率分布
    assert np.isclose(np.sum(p), 1), "概率分布p的元素之和必须为1"
    assert np.isclose(np.sum(q), 1), "概率分布q的元素之和必须为1"
    # 计算KL散度
    kl = np.sum(p * np.log(p / q))
    return kl

if __name__ == '__main__':
    p = np.array([0.1, 0.2, 0.7])  # 真实分布
    q = np.array([0.4, 0.4, 0.2])  # 近似分布

    kl_div = kl_divergence(p, q)
    print(f"KL散度(P || Q): {kl_div}")
    