from __future__ import print_function, absolute_import
import numpy as np
import copy


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    # 评估函数,参数有(距离矩阵,查询行人的id,图库行人的id,查询的摄像头,图库的摄像头,最大秩)
    num_q, num_g = distmat.shape  # num_q查询的行人数=距离矩阵的行, num_g图库的行人数=距离矩阵的列
    if num_g < max_rank:  # 图库的行人数 < 最大秩
        max_rank = num_g  # 更改最大秩 = 图库的行人数  ,说明图库的样本数太小了
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)  # top_k
    # 将矩阵distmat按照axis排序，并返回排序后的下标
    matches = (g_pids[indices] == q_pids[:, np.newaxis])  # q_pids[:, np.newaxis] 在原有维度后面加一个维度 (n,1)
    # 匹配,相同返回true,不同返回false,
    matches = matches.astype(np.int32)  # 将bool类型的true false转换成int类型的1, 0

    # compute cmc curve for each query  对于每个查询计算它的cmc曲线
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):  # 遍历每个查询的行人
        # get query pid and camid
        q_pid = q_pids[q_idx]  # 获得每个查询的行人身份
        q_camid = q_camids[q_idx]  # 获得对应的摄像头id

        # remove gallery samples that have the same pid and camid with query 从图库样本中删除与查询集中有相同身份和摄像头id
        order = indices[q_idx]  # 查询集索引
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)  # 相等为1,不等为0
        keep = np.invert(remove)  # np.invert() 位非,对每一位取反  ,则删去了相等的,变成去除相等的为0 ,保留不等的为1

        # compute cmc curve  累积匹配曲线
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        # 二进制向量，值为1的位置是正确的匹配
        if not np.any(orig_cmc):  # np.any  相当于或运算.如果可迭代对象orig_cmc中任意存在每一个元素为True则返回True,
            # this condition is true when query identity does not appear in gallery
            # 当查询的身份未出现在图库中时，此条件为真
            continue

        cmc = orig_cmc.cumsum()  # 返回累加和,不改变数据形状
        cmc[cmc > 1] = 1  # cmc最大为1,超过1的置为1

        all_cmc.append(cmc[:max_rank])  # 将前max_rank的cmc添加到all_cmc列表的后面   rank1到rank50
        num_valid_q += 1.  # 有效的查询身份+1

        # compute average precision  计算平均精度
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()  # 所有元素求和
        tmp_cmc = orig_cmc.cumsum()  # 累加和,不改变数据形状
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]  # enumerate() 函数,返回数据下标和数据(i,x)
        # 计算top_i的cmc ..x / (i+1.)
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc  # np.asarray(tmp_cmc),数据类型转换为数组,  只保留正确的匹配 , 错误的值为0
        AP = tmp_cmc.sum() / num_rel  # 平均精度 = 正确匹配的元素求和  除以原来所有元素的总和
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
