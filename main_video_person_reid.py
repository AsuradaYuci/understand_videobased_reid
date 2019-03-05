from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler

# 命令行参数
parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets   数据集
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options  优化选择
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture  网络结构
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, "
                                                                         "resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str,
                    default='/home/ying/Desktop/Video-Person-ReID-master/resnet-50-kinetics.pth',
                    help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

# 解析命令行参数
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices  # 在代码中指定需要使用的GPU
    use_gpu = torch.cuda.is_available()  # 查看当前环境是否支持CUDA,支持返回true，不支持返回false
    if args.use_cpu:
        use_gpu = False

    if not args.evaluate:  # 如果不是评估，那就是训练，输出训练日志；否则输出测试日志。
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))  # 打印所有参数

    if use_gpu:  # 如果使用gpu，输出选定的gpu，
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True  # 在程序刚开始加这条语句可以提升一点训练速度,没什么额外开销
        torch.cuda.manual_seed_all(args.seed)  # 为GPU设置种子用于生成随机数，以使得结果是确定的
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)  # 初始化数据集,从data_manager.py文件中加载。

    # import transforms as T.
    # T.Compose=一起组合几个变换。
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),  # 以一个概率进行，首先将图像大小增加到（1 + 1/8），然后执行随机裁剪。
        T.RandomHorizontalFlip(),  # 以给定的概率(0.5)随机水平翻转给定的PIL图像。
        T.ToTensor(),  # 将``PIL Image``或``numpy.ndarray``转换为张量。
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 用平均值和标准偏差归一化张量图像。
        # input[channel] = (input[channel] - mean[channel]) / std[channel]
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),  # 将输入PIL图像的大小调整为给定大小。
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    pin_memory = True if use_gpu else False

    # DataLoader数据加载器。 组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
    trainloader = DataLoader(
        # VideoDataset:基于视频的person reid的数据集.(训练的数据集，视频序列长度，采样方法：随机，进行数据增强）
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random', transform=transform_train),
        # 随机抽样N个身份，然后对于每个身份，随机抽样K个实例，因此批量大小为N * K.
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch,  # 训练的批次大小
        num_workers=args.workers,  # 多进程的数目
        pin_memory=pin_memory,
        drop_last=True,
    )  # 如果数据集大小不能被批量大小整除，则设置为“True”以删除最后一个不完整的批次。

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch,
        shuffle=False,  # 设置为“True”以使数据在每个时期重新洗牌（默认值：False）。
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,  # 如果“False”和数据集的大小不能被批量大小整除，那么最后一批将更小。
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))  # 模型的初始化

    if args.arch == 'resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width,
                                  sample_height=args.height, sample_duration=args.seq_len)
        # 如果不存在预训练模型，则报错
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        # 导入预训练的模型
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}  # 状态字典,从checkpoint文件中加载参数
        for key in checkpoint['state_dict']:
            if 'fc' in key:
                continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    # 损失函数：xent：softmax交叉熵损失函数。htri：三元组损失函数。
    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)
    # 优化器：adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # stepsize，逐步减少学习率（> 0表示已启用）
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
        # lr_scheduler学习率计划，StepLR,将每个参数组的学习速率设置为每个步长时期由gamma衰减的初始lr.
    start_epoch = args.start_epoch  # 手动时期编号（重启时有用）

    if use_gpu:
        model = nn.DataParallel(model).cuda()  # 多GPU训练
        # DataParallel是torch.nn下的一个类，需要制定的参数是module（可以多gpu运行的类函数）和input（数据集）

    if args.evaluate:  # 这里的evaluate没有意义，应该添加代码导入保存的checkpoint，再test
        print("Evaluate only")  # 进行评估
        test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()  # 开始的时间
    best_rank1 = -np.inf  # 初始化，负无穷
    if args.arch == 'resnet503d':  # 如果模型为resnet503d,
        torch.backends.cudnn.benchmark = False

    for epoch in range(start_epoch, args.max_epoch):  # epoch,从开始到最大，进行训练。
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        
        if args.stepsize > 0:
            scheduler.step()

        # 如果运行一次评估的需要的epoch数大于0，并且当前epoch+1能整除这个epoch数，或者等于最大epoch数。那么就进行一次评估。
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1  # 比较，大于则返回true，否则返回false。
            if is_best:
                best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
                # 函数static_dict()用于返回包含模块所有状态的字典，包括参数和缓存。
            else:
                state_dict = model.state_dict()
            # 保存checkpoint文件
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
    # 经过的时间
    elapsed = round(time.time() - start_time)  # round() 方法返回浮点数x的四舍五入值
    elapsed = str(datetime.timedelta(seconds=elapsed))  # 对象代表两个时间之间的时间差,
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):

    model.train()  # 选择训练的数据集
    losses = AverageMeter()  # 计算和保存当前值和平均值。

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):  # trainloader，121行，
        # 从trainloader中获得批次的索引，图像和行人的id。
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()  # 将数据转到gpu上
        imgs, pids = Variable(imgs), Variable(pids)  # 将imgs, pids装进Variable中
        outputs, features = model(imgs)  # 喂给模型图片
        if args.htri_only:
            # only use hard triplet loss to train the network，只使用三元组损失训练网络
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss 三元组损失加交叉熵损失函数
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            loss = xent_loss + htri_loss
        optimizer.zero_grad()  # 将所有参数的梯度置为0
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 进行adam优化
        losses.update(loss.data[0], pids.size(0))  # 参数更新

        if (batch_idx+1) % args.print_freq == 0:  # 输出的频率，多少批次输出一次
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):

    model.eval()  # 模型的评估

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)  # 将imgs装进Variable中,
        # volatile=True的节点不会求导，即使requires_grad=True，也不会进行反向传播，对于不需要反向传播的情景(inference，测试推断)，
        # 该参数可以实现一定速度的提升，并节省一半的显存，因为其不需要保存梯度

        b, n, s, c, h, w = imgs.size()  # b=1, n=batchs, s=图片的长度
        assert(b == 1)  # 断言函数
        imgs = imgs.view(b*n, s, c, h, w)
        features = model(imgs)  # 喂给模型图片，获得特征
        features = features.view(n, -1)  # view()函数作用是将一个多行的Tensor,拼接成一行。
        features = torch.mean(features, 0)  # 取平均值
        features = features.data.cpu()
        qf.append(features)  # 向列表尾部追加一个新元素，序列特征
        q_pids.extend(pids)   # 向列表尾部追加一个列表,人的身份person id
        q_camids.extend(camids)  # 摄像机的id
    qf = torch.stack(qf)  # 堆叠
    q_pids = np.asarray(q_pids)  # 将列表转换为数组
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []  # 图库，gallery
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s, c, h, w)
        assert(b == 1)
        features = model(imgs)
        features = features.view(n, -1)
        if pool == 'avg':  # 采用平均池化还是最大池化
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    # 计算距离矩阵
    m, n = [gf.size(0), qf.size(0)]  # 矩阵的行
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
