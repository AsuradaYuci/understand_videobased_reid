from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch


class RandomIdentitySampler(object):  # 随机为每个id选择一定数目的样本,形成一个批次
    """
    Randomly sample N identities, then for each identity,  首先随机采样N个身份,然后对每个身份,随机选择K个例子,根据这种策略
    randomly sample K instances, therefore batch size is N*K.   形成一个N*K批次.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.  一个包含样本的参数
        num_instances (int): number of instances per identity.  每个身份的样本数
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source  # 数据集
        self.num_instances = num_instances  # 默认给每个身份选择4帧图片
        self.index_dic = defaultdict(list)  # 索引字典的初始化,采用defaultdict()方法
        # defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值,[]
        for index, (_, pid, _) in enumerate(data_source):  # 将数据集中的行人id和索引放进字典中
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # 行人id是字典的key值
        self.num_identities = len(self.pids)  # 行人的个数是行人id的长度

    def __iter__(self):  # 迭代器
        indices = torch.randperm(self.num_identities)  # torch.randperm(n),给定参数n,返回一个从0到n-1的随机整数排列.
        ret = []  # 初始化个列表
        for i in indices:  # 遍历数据集中的行人身份
            pid = self.pids[i]  # 行人的身份
            t = self.index_dic[pid]  # 获得字典中,行人身份对应的样本数
            replace = False if len(t) >= self.num_instances else True
            # 如果t>=4,则replace = False 如果t < 4,replace = True ,即样本数小于4,则复制前面的样本
            t = np.random.choice(t, size=self.num_instances, replace=replace)  # np.random.choice(a,size,replace,p)
            # 从t中选择size个样本,replace = true的话有可能会出现重复的样本,就是将前面抽出来的样本重新放回去.
            ret.extend(t)  # 在list的末尾一次性追加另一个序列中的多个值,即用新列表扩展原来的列表.
        return iter(ret)  # iter()函数,迭代器.

    def __len__(self):
        return self.num_identities * self.num_instances  # 一个批次的长度
