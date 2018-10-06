import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding 3*3*3卷积块    https://blog.csdn.net/weicao1990/article/details/80283443
    # 对d帧h*w的彩色RGB图像进行卷积,即假设输入的大小为,(d,w,h),通道数为c,卷积核(滤波器)为f,即滤波器的维度为f*f*f*c,卷积核的数目为1,
    # 输出  (d-f+1)*(w-f+1)*(h-f+1)*1
    return nn.Conv3d(
        in_planes,  # 输入平面,输入信号的通道
        out_planes,  # 输出平面,卷积产生的通道
        kernel_size=3,  # 卷积核大小为3
        stride=stride,  # 卷积步长为1
        padding=1,  # 填充,输入的每一条边补充0的层数   如果padding不是0，会在输入的每一边添加相应数目0
        bias=False  # 不添加偏置
    )


def downsample_basic_block(x, planes, stride):  # 下采样基本块,进行了一次平均池化
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)  # 3d平均池化  输入大小:(N,C,D_in,H_in,W_in)
    # D_out = (D_in-1)/(stride+1)  H_out = (H_in-1)/(stride+1)  W_out = (W_in-1)/(stride+1)  输出:(N,C,D_out,H_out,W_out)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()  # 用0填充该tensor
    if isinstance(out.data, torch.cuda.FloatTensor):  # isinstance(object, classinfo)
        # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
        zero_pads = zero_pads.cuda()  # 转换成GPU张量

    out = Variable(torch.cat([out.data, zero_pads], dim=1))  # 将out.data, zero_pads按列放在一起

    return out


class BasicBlock(nn.Module):  # resnet的基本块,都是3*3*3卷积
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)  # 第一次3*3*3卷积
        self.bn1 = nn.BatchNorm3d(planes)  # 对小批量(mini-batch)4d数据组成的5d输入进行批标准化(Batch Normalization)操作
        # (N, C, D, H, W)
        self.relu = nn.ReLU(inplace=True)  # 激励函数{ReLU}(x)= max(0, x)  inplace-选择是否进行覆盖运算

        self.conv2 = conv3x3x3(planes, planes)  # 第二次 3*3*3卷积
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample  # 跳跃链接
        self.stride = stride

    def forward(self, x):  # 前向传播,朴素残差模块(不带bottleneck)
        residual = x  # 残差

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # H(x)=F(x)+x
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # bottleneck残差模块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # planes = 64
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)  # planes = 256
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_height,
                 sample_width,
                 sample_duration,  # 样本的持续时间,图片的数目 d
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64  # 输入为64个通道
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(  # part1.第一次卷积,
            3,  # 输入的通道为3  (d,h,w)
            64,  # 输出的通道为64
            kernel_size=7,  # 卷积核(滤波器的尺寸)  7*7*7
            stride=(1, 2, 2),  # 步长
            padding=(3, 3, 3),  # padding的深度为3
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # part2: 最大池化.滤波器为3*3*3, 步长为2, 填充的深度为1
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)  # 构建一个bottleneck残差模块,
        # 输入通道数为64,  输出通道数变为256  这一层构建layers[0]=3个卷积块

        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        # 输入通道数为128,  输出通道数变为512  这一层构建layers[1]=4个卷积块

        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        # 输入通道数为256,  输出通道数变为1024  这一层构建layers[2]=6个卷积块

        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        # 输入通道数为512,  输出通道数变为2048  这一层构建layers[3]=3个卷积块

        last_duration = int(math.ceil(sample_duration / 16.0))  # 确定最后平均池化的滤波器大小
        last_height = int(math.ceil(sample_height / 32.0))
        last_width = int(math.ceil(sample_width / 32.0))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_height, last_width), stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层,分类器

        for m in self.modules():  # 遍历网络中的所有模型
            if isinstance(m, nn.Conv3d):  # 如果模型的类型与nn.Conv3d一样,其权重初始化为
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        # block = bottleneck 残差模块   planes = 输入的通道数
        # blocks = 要构建几个残差块
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 步长不等于1,或者通道数不等于当前的通道乘以4
            if shortcut_type == 'A':  # 跳跃链接方式"A",进行平均池化
                downsample = partial(  # 偏函数Partial(第一个参数 = 一个函数,第二部分=可变参数,第三个参数)
                    downsample_basic_block,  # 将所要承载的函数作为partial()函数的第一个参数，
                    planes=planes * block.expansion,  # 原函数的各个参数依次作为partial()函数后续的参数，除非使用关键字参数.
                    stride=stride)  # 这样得到一个新的函数,对原来的downsample_basic_block函数进行了扩展
            else:  # 跳跃链接的方式"B",  进行一次卷积操作+批量归一化
                downsample = nn.Sequential(
                    nn.Conv3d(  # 一次1*1*1  256个通道的卷积
                        self.inplanes,
                        planes * block.expansion,  # 256
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)  # 批量归一化操作
                )

        layers = []  # 建立一个层的列表

        layers.append(block(self.inplanes, planes, stride, downsample))  # 将残差块内容放入layers列表中,
        # 第一个残差块是带有降采样的跳跃链接
        self.inplanes = planes * block.expansion  # 输入的通道数*4
        for i in range(1, blocks):  # blocks = layers[]  从第二个开始遍历layers列表  根据[n]判断要建立几个基本残差块
            layers.append(block(self.inplanes, planes))  # 从第二个残差块开始,跳跃链接没有降采样

        return nn.Sequential(*layers)  # 返回构建好的残差块
    
    def load_matched_state_dict(self, state_dict):  # 从预训练模型中加载匹配的状态字典,即模型参数
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            param = param.data
            print("loading "+name)
            own_state[name].copy_(param)

    def forward(self, x):  # 整个Resnet的前向传播
        # default size is (b, s, c, w, h), s for seq_len, c for channel
        # convert for 3d cnn, (b, c, s, w, h)
        x = x.permute(0,2,1,3,4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return y, x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

