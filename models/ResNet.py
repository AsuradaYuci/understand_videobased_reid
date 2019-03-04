from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN']
# 用 __all__ 在模块级别暴露接口,1.提供了哪些是公开接口的约定
# 2.控制 from xxx import * 的行为,import * 就只会导入 __all__ 列出的成员。
# 3.list类型,以字面量的形式显式写出来.同时应该写在所有 import 语句下面


class ResNet50TP(nn.Module):  # 定义时间池化网络模型
    def __init__(self, num_classes, loss={'xent'}, **kwargs):  # **kwargs表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现
        super(ResNet50TP, self).__init__()
        # 首先找到ResNet50TP的父类(nn.Module),然后把类ResNet50TP的对象转换为父类的对象
        self.loss = loss  # 损失函数为标签平滑正则化交叉熵损失函数
        resnet50 = torchvision.models.resnet50(pretrained=True)  # 调用封装好的resnet50模型,
        # pretrained=True 调用一个在imagenet上预训练过的模型
        self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 快速构建基本网络,选择resnet50的子网络,去掉原模型的最后两层
        # nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
        self.feat_dim = 2048  # 输出的特征维度为2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)  # 分类器
        # nn.Linear(x,y)一种线性变换y = Ax + b

    def forward(self, x):  # 前向传播函数定义,输入为x,5维的tensor (batch_size,seq_len，channels,width,height,)
        b = x.size(0)  # batch_size
        t = x.size(1)  # seq_len
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))  # b*t= 32 × 4 =128 图片总数
        # view()会将原有数据重新分配为一个新的张量
        x = self.base(x)  # x输入到基本网络中
        x = F.avg_pool2d(x, x.size()[2:])  # avg_pool2d(x, x.size(2), x.size(3), x.size(4))
        # 2d平均池化操作,input tensor (minibatch x in_channels x iH x iW)
        x = x.view(b, t, -1)  # -1 表示维度从其他维度推断出来的
        # 将数据重新分配
        x = x.permute(0, 2, 1)
        # 调整数据的维度,将1,2维数据互换
        f = F.avg_pool1d(x, t)  # 1d的平均池化
        f = f.view(b, self.feat_dim)  # b为行人的身份数,也是矩阵的行数,矩阵有2048列
        if not self.training:  # 如果是测试,返回特征向量
            return f
        y = self.classifier(f)  # 返回分类器输出结果

        if self.loss == {'xent'}:  # 如果损失函数是标签平滑正则化softmax交叉熵损失函数
            return y
        elif self.loss == {'xent', 'htri'}:  # 损失函数是交叉熵损失函数+难样本挖掘三元组损失
            return y, f
        elif self.loss == {'cent'}:  # 中心损失
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):  # 时间注意力模型
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        # 注意力的生成,通过softmax or sigmoid
        self.feat_dim = 2048  # feature dimension 特征维度为2048
        self.middle_dim = 256  # middle layer dimension  中间层的维度为256
        self.classifier = nn.Linear(self.feat_dim, num_classes)  # 线性输出
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, (7, 4))  # 输入为2048,滤波器为256
        # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x):
        b = x.size(0)  # b = 32
        t = x.size(1)  # t = 4
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))  # 128x3x224x112
        x = self.base(x)  # 将图片输入到模型中  128x2048x7x4
        a = F.relu(self.attention_conv(x))  # 激励函数 128x256x1x1
        a = a.view(b, t, self.middle_dim)  # 将原有数据重新分配为一个新的张量  (b,t,256) => 32x4x256
        a = a.permute(0, 2, 1)  # 调整数据的维度,将1,2维数据互换  32x256x4
        a = F.relu(self.attention_tconv(a))  # 激励函数 32x1x4
        a = a.view(b, t)  # 将原有数据重新分配为一个新的张量  (b,t) => 32x4
        x = F.avg_pool2d(x, x.size()[2:])  # avg_pool2d(x, x.size(3), x.size(4)) 2d的平均池化128x2048x1x1 
        # a = 注意力分数
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)  # dim = 1,在维度1计算softmax
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)  # p =1 标准公式中的指数值, dim = 1 在维度一进行归一化操作
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)  # 将x数据重新进行分配 32x4x2048
        a = torch.unsqueeze(a, -1)  # 在a所在维度上增加1  (b,t,1) 32x4x1
        a = a.expand(b, t, self.feat_dim)  # 指定单个维度扩大为更大的尺寸,(b,t,2048) 32x4x2048
        att_x = torch.mul(x, a)  # 用标量值a乘以输入x的每个元素，并返回一个新的结果张量。32x4x2048
        att_x = torch.sum(att_x, 1)  # 返回输入张量给定维度上每行的和。返回1列数  32x2048
        
        f = att_x.view(b, self.feat_dim)  # 将att_x数据重新进行分配,(b,2048)  32x2048
        if not self.training:
            return f
        y = self.classifier(f)  # 线性输出,根据特征向量f分类  32x625

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)  # 输出层
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,  # 每个小段视频的特征维度
            hidden_size=self.hidden_dim,  # LSTM的隐藏单元
            num_layers=1,  # 有几层lstm层
            batch_first=True  # 输入和输出会是以batch_size为第一维度的特征集
                            )

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)  # x输入到基本网络中
        x = F.avg_pool2d(x, x.size()[2:])  # 进行2d的平均池化
        x = x.view(b, t, -1)  # 对x的数据重新进行分配,  输入x的形状为(batch,time_step,input_size)
        # lstm有两个隐藏层状态,h_n是分线,h_c是主线
        output, (h_n, h_c) = self.lstm(x)  # 输出为(batch,time_step,output_size)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)  # 关于时间步的1d平均池化
        f = f.view(b, self.hidden_dim)  # 对数据重新进行分配,(b,512)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
