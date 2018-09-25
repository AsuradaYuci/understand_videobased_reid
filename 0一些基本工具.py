from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import torch


def mkdir_if_missing(directory):  # 如果文件不存在,就新建一个文件
    if not osp.exists(directory):  # 如果不存在该文件
        try:
            os.makedirs(directory)  # 建立这个文件
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.     计算并存储平均值和当前值.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, val=0, avg=0, sum=0, count=0):
        self.val = val
        self.avg = avg
        self.sum = sum
        self.count = count
        # self.reset()

    def reset(self):  # 复位
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # 更新各个值的状态
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):  # 保存checkpoint文件
    mkdir_if_missing(osp.dirname(fpath))  # 如果当前路径不存在该名字的文件,就新建它
    torch.save(state, fpath)  # 将模型的状态,保存在fpath文件中
    if is_best:  # 如果这个模型的结果是最好的.
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))
        # shutil.copyfile(src, dst)：复制文件内容（不包含元数据）从src到dst。    dst必须是完整的目标文件名.


class Logger(object):
    """
    Write console output to external text file.     将控制台输出写入外部文本文件。日志输出
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout  # 控制台输出
        self.file = None
        if fpath is not None:  # 如果fpath存在,就建立这个文件
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')  # 打开这个文件,打开文件的模式是写入.

    def __del__(self):  # 关闭文件
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):  # 退出文件
        self.close()

    def write(self, msg):  # 将信息写入文件
        self.console.write(msg)  # = sys.stdout.write(msg)  打印信息
        if self.file is not None:
            self.file.write(msg)  # = open(fpath,'w').write(msg)  向checkpoint文件中写入信息.

    def flush(self):  # flush()函数:刷新stdout,每隔一秒输出,在屏幕上可以实时看到输出信息.
        self.console.flush()
        if self.file is not None:
            self.file.flush()  # 刷新缓冲区，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
            os.fsync(self.file.fileno())  # fileno()返回一个整型的文件描述符(file descriptor FD 整型)，可用于底层操作系统的I/O 操作
            # fsync()强制将文件描述符为fd的文件写入硬盘

    def close(self):
        self.console.close()  # 关闭控制台输出
        if self.file is not None:
            self.file.close()  # 关闭文件


def read_json(fpath):
    with open(fpath, 'r') as f:  # 打开fpath文件,以只读模式,作为f
        obj = json.load(f)  # 读取json信息,取得指定文件中内容,参数为要读取的文件对象
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))  # 先建立文件
    with open(fpath, 'w') as f:  # 打开fpath文件,以写入模式,作为f
        json.dump(obj, f, indent=4, separators=(',', ': '))  # 存入到指定文件,第一个参数为要存入的内容,第二个为文件的对象
        # indent=4,换行且按照indent的数值显示前面的空白分行显示
        # separators：分隔符，这表示dictionary内keys之间用“,”隔开，而KEY和value之间用“：”隔开。
