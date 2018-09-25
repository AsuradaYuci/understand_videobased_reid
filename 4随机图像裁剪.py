from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    设定一个概率p,首先将图像尺寸扩大到(1 + 1/8),然后执行随机裁剪
    Args:
        height (int): target height.  目标图像的高度
        width (int): target width.  目标图像的宽度
        p (float): probability of performing this transformation. Default: 0.5.  执行改变图像大小,进行裁剪的概率
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation  # 插值方式:双线性插值

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:  # random.random()返回随机生成的一个实数,它在[0,1)范围内.
            return img.resize((self.width, self.height), self.interpolation)  # 不改变图像的大小,((图像的宽, 高), 双线性插值)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        # 先将图像尺寸扩大到原来的1.125倍, round函数:返回浮点数的四舍五入值
        resized_img = img.resize((new_width, new_height), self.interpolation)  # 改变图像的大小,采用双线性插值方法
        x_maxrange = new_width - self.width  # 宽度的余量
        y_maxrange = new_height - self.height  # 高度的余量
        # random.uniform(x,y) 随机生成一个实数,它在[x,y)范围内.
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))  # 随机确定左上顶点
        box = (x1, y1, x1 + self.width, y1 + self.height)  # 裁剪图片的区域范围(左,上,右,下)
        # Python中规定左上角为(0,0)的坐标点,最后的两个数字必须比前面两个大
        croped_img = resized_img.crop(box)  # 进行图像裁剪

        return croped_img


if __name__ == '__main__':
    pass
