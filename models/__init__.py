from __future__ import absolute_import

from .ResNet import *  # 导入模块，每次使用模块中的函数，直接使用函数就可以了
# 是把ResNet模块中所有函数都导入进来; 注：相当于导入的是一个文件夹中所有文件，所有函数都是绝对路径。
__factory = {
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'resnet50rnn': ResNet50RNN,
}  # 字典


def get_names():  # 获得相应模型的名称(key)
    return __factory.keys()


def init_model(name, *args, **kwargs):  # 初始化模型
    if name not in __factory.keys():  # 如果模型名称不在模型字典中,报错,keyerror	映射中没有这个键
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)  # 返回字典中相应key对应的value,这里是模型的函数名
