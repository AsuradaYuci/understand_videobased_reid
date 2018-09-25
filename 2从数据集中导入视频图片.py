from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random


def read_image(img_path):
    """Keep reading image until succeed.一直进行读入图片操作，直到成功。
    This can avoid IOError incurred by heavy IO process."""
    got_img = False  # 读取图像的标志，初始为False
    if not os.path.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            # 从图像路径读取图片，并将图片转换为RGB格式
            img = Image.open(img_path).convert('RGB')
            got_img = True  # 成功读取图片，则读取图像的标志置为TRUE
        except IOError:  # 读取图片出错，报错
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.基于视频reid的数据集。
    Note batch data has shape (batch, seq_len, channel, height, width).
    注意，一个批次的数据的形状为（batch，序列长度，通道数，图片的高度，图片的宽度）
    """
    # 三种采样方法，平等的，随机，所有
    sample_methods = ['evenly', 'random', 'all']

    # 初始化数据集
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # 数据集长度

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]  # 从数据集中图片的索引（名称）获得图片路径，行人的身份，摄像头的id
        num = len(img_paths)

        # 采样方式：1.随机采用
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            随机从num个视频帧中采样seq_len长度的连续帧，如果num小于seq_len，则复制它们，使之等于seq_len。在训练阶段采用这种采样策略。
            """
            frame_indices = list(range(num))  # 视频帧的索引0-num
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)  # 随机范围的终止
            begin_index = random.randint(0, rand_end)  # 开始的索引值
            end_index = min(begin_index + self.seq_len, len(frame_indices))  # 结束的索引值

            indices = frame_indices[begin_index:end_index]  # 最终的索引范围

            for index in indices:  # 遍历索引
                if len(indices) >= self.seq_len:  # 如果索引的长度大于等于序列长度
                    break
                indices.append(index)  # 向列表末尾增加索引
            indices = np.array(indices)  # 将列表转换成数组

            imgs = []  # 图像列表
            for index in indices:  # 遍历新的索引
                index = int(index)  # 首先保证索引是整数
                img_path = img_paths[index]  # 从数据集中获得图像的路径,并结合索引,确定图像的具体路径
                img = read_image(img_path)  # 根据路径读取图片
                if self.transform is not None:  # 执行随机裁剪
                    img = self.transform(img)
                img = img.unsqueeze(0)  # unsqueeze(0)函数,增加维度,在第0维增加1个维度
                imgs.append(img)  # 向图像列表末尾增加图像索引.存储这些索引,保证时间顺序
            imgs = torch.cat(imgs, dim=0)  # cat([a,b],dim),若dim=0,则将a,b按行放在一起. 若dim=1,则a,b按列放在一起
            # imgs = imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to 
            be set to 1.
            将视频中的所有帧采样到一系列的clips，每个clip包含seq_len个帧,批次大小需要设置为1
            This sampling strategy is used in test phase. 在测试阶段采用密集采样策略.
            """
            cur_index = 0  # 初始索引为0
            frame_indices = list(range(num))  # 视频帧的索引范围
            indices_list = []
            while num - cur_index > self.seq_len:  # 如果视频帧的数目大于seq_len
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                # 将帧的索引[0,seq_len],添加到原来帧的索引列表的末尾
                cur_index += self.seq_len  # 当前索引值=0+seq_len

            last_seq = frame_indices[cur_index:]  # 最后留下的序列为当前索引值后面的列表值
            for index in last_seq:  # 遍历剩下的索引值
                if len(last_seq) >= self.seq_len:  # 如果剩下的索引值长度大于每个身份要采样的序列长度
                    break
                last_seq.append(index)  # 将索引加到列表后面
            indices_list.append(last_seq)  # 存储这些索引,保证时间顺序

            imgs_list = []
            for indices in indices_list:  # 遍历所有索引中的视频段的索引,即有几个视频段
                imgs = []
                for index in indices:  # 在每个视频段中,遍历每一帧的索引
                    index = int(index)
                    img_path = img_paths[index]  # 确定图像的路径
                    img = read_image(img_path)  # 读图
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                # imgs = imgs.permute(1,0,2,3)
                imgs_list.append(imgs)  # 将每个视频段的图像,存在一个总的list中
            imgs_array = torch.stack(imgs_list)  # 沿着一个新维度对输入张量序列进行连接。
            return imgs_array, pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
