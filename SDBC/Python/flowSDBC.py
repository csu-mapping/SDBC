# coding=utf-8
# copyright(c) 2021 Central South University. all rights reserved.
# jianbo.tang@csu.edu.cn
# 2024/10/15

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Arial'
import numpy as np
import time
from flowread import *
from flowplot import *


def flowSDBC(flows, flow_times, spatial_R, angle_T):
    MIN_CLUSTER_MEMBER_NUMS = 5
    dist_mode = 2
    print('Running flowSDBC clustering algorithm...')

    # Step.1 构建邻接矩阵nnmat
    print('Step1. constructing the spatio-temporal neighborhoods of each flow...')
    n = flows.shape[0]
    nnmat = np.zeros((n, n), dtype=bool)
    if flow_times.size > 0:
        for i in range(n):
            for j in range(i + 1, n):
                d, r = flow_distance(flows[i, :], flows[j, :], dist_mode)
                t = (flow_times[i, 0] <= flow_times[j, 1]) and (flow_times[i, 1] >= flow_times[j, 0])
                if d <= spatial_R and r <= angle_T and t:
                    nnmat[i, j] = True
                    nnmat[j, i] = True
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d, r = flow_distance(flows[i, :], flows[j, :], dist_mode)
                if d <= spatial_R and r <= angle_T:
                    nnmat[i, j] = True
                    nnmat[j, i] = True
    print('done')

    # 计算流密度
    print('Step2. computing the density of each flow...')
    flow_density = np.zeros(n)
    spatialneighbors = [np.where(nnmat[i, :])[0] for i in range(n)]
    for i in range(n):
        flow_density[i] = np.sum(nnmat[i, :])
    print('done')

    # Step.2 计算每个流的Gi值，并确定种子流
    print('Step3. identifying the seed flows with Getis-Ord Gi statistis...')
    npts = flows.shape[0]
    data = flow_density
    #data 存储统计Gi值
    data = (data - np.mean(data)) / np.std(data, ddof=0)
    lamada = np.mean(data) - 2 * np.std(data, ddof=0)
    K = np.sum(nnmat, axis=1)
    K = np.sqrt(K * (npts - K) / (npts - 1))
    K[K < 1] = np.finfo(np.float32).eps
    G = nnmat @ data + data
    G = G / K
    seeds = np.where(G >= lamada)[0]

    if len(seeds) < 1:
        IDX = list()
        print('Warning: not any seed flows and clusters detected in the data.')
        return IDX, seeds, flow_density, spatialneighbors
    print('done')

    # 3
    print('Step4. statistic-constrianted density-growthing clustering ...')
    #初始化一个热点（hotspot）列表
    hotspots = [{'Members': [i],
                 'Gi': data[i],
                 'IsSeed': 0,
                 'InnerDis': None} for i in range(npts)]

    for i in seeds:
        hotspots[i]['IsSeed'] = 1

    ########## PLOT SEED FLOWS ##########
    # Figure 1
    fig = plt.figure(1, figsize=(7, 6))
    hold('on')
    IDX = np.zeros(npts)
    IDX[seeds] = 1
    # IDX[[hotspot['IsSeed'] for hotspot in hotspots]] = 1
    flowplot(flows, IDX, 2)
    plt.title('Seed flows', fontsize=13, fontweight='bold')
    draw_now(fig)
    ######################################
   
    count = 0
    nclust = len(hotspots)
    #迭代聚类
    fig = plt.figure(2, figsize=(7, 6))
    hold('on')
    fig.show()
    plt.pause(0.5)
    while nclust > 1:
        I, J, gstat = searching(data, nnmat, hotspots, lamada)
        if I ==-1 or J == -1:
            break
        hotspots, nnmat = updating(nnmat, hotspots, I, J, gstat)
        nclust = len(hotspots)
        count = count + 1
        ########## PLOT CURRENT CLUSTERS ##########
        # Conditional plotting
        if count % 5 == 0:
            plt.cla()
            IX = get_idx(hotspots, npts, 2)
            # 获取当前聚类中心的索引
            flowplot(flows, IX, 2)
            plt.title(f'Clustering (iteration={count})', fontsize=13, fontweight='bold')
            draw_now(fig)
        ###########################################
    del nnmat, data
    print('done')

    print('Step5. merging of neighboring clusters ...')
    for i in range(len(hotspots)):
        dmax, _ = cluster_inner_distance(flows, hotspots[i]['Members'], dist_mode)
        hotspots[i]['InnerDis'] = dmax
    
    n = len(hotspots) + 1
    while len(hotspots) < n:
        n = len(hotspots)
        hotspots, _ = cluster_merge(flows, hotspots, dist_mode)
        ############# MERGE CLUSTERS ###############
        count = count + 1
       # 创建一个图形对象
        plt.cla()
        IX = get_idx(hotspots, npts, 2)
        flowplot(flows, IX, 2)
        plt.title(f'Clustering (iteration={count})', fontsize=13, fontweight='bold')
        draw_now(fig)
        ############################################
    print('done')

    print('Step6. showing the detected candidate clusters ...')
    mnums = np.array([len(h['Members']) for h in hotspots])
    hotspots = [h for h in hotspots if len(h['Members']) >= MIN_CLUSTER_MEMBER_NUMS]

    IDX = np.zeros(npts, dtype=int)
    mnums = np.zeros(npts, dtype=int)
    for i in range(len(hotspots)):
        IDX[hotspots[i]['Members']] = i + 1
        mnums[i] = len(hotspots[i]['Members'])

    I = np.argsort(mnums)[::-1]
    is_seed = np.zeros(npts, dtype=bool)
    is_seed[seeds] = True
    for i in I:
        cid = i + 1
        # print('cluster', cid, 'size:', mnums[i])
        nn = np.array([neighbor for idx in np.where(IDX == cid)[0] for neighbor in spatialneighbors[idx]])
        if(mnums[i] == 0):
            continue
        ind = is_seed[nn] & (IDX[nn] < 0)
        if np.any(ind):
            IDX[nn[ind]] = cid
    del mnums

    ############### FINAL CLUSTERS #############
    plt.cla()
    flowplot(flows, IDX, 2)
    plt.title('Detected candidate clusters', fontsize=13, fontweight='bold')
    draw_now(fig)
    ############################################
    print('done')
    return IDX, seeds, flow_density, spatialneighbors


# searching
#在给定的数据集中找到具有最大统计值的聚类组合
def searching(data, nnmat, hotspots, lamada):
    n = len(data)
    seeds = [i for i, hotspot in enumerate(hotspots) if hotspot['IsSeed']]
    clusternums = len(seeds)
    I = -1
    J = -1
    max_gstat = -999999.0  #存储当前找到的最大统计值
    for i in range(clusternums):
        id = seeds[i]
        A = hotspots[id]['Members']
        gstatA = hotspots[id]['Gi']
        neighbor = np.nonzero(nnmat[id, :])[0]
        #包含了与 id 热点相邻的所有节点的索引，返回一个包含非零元素索引的元组，通过 [0] 提取第一个元素（即行索引）
        if neighbor.size == 0:
            continue  #跳过当前循环的剩余部分，直接进入下一次循环迭代

        for j in range(len(neighbor)):
            nid = neighbor[j]
            B = hotspots[nid]['Members']
            gstatB = hotspots[nid]['Gi']
            # #判断两个聚类的统计值是否大于阈值(自己加)
            # if gstatA < lamada or gstatB < lamada:
            #     continue
            #将A和B合并
            current_member = np.concatenate((A, B))
            k = len(current_member)

            # measuring index
            current_stat = np.sum(data[current_member]) / np.sqrt(k * (n - k) / (n - 1))
            # 更新聚类的统计值
            hotspots[I]['Gi'] = current_stat
            hotspots[J]['Gi'] = current_stat
            if (current_stat > lamada) and (current_stat > max_gstat):
                max_gstat = current_stat
                I = id
                J = nid
    return I, J, max_gstat


# updating
def updating(nnmat, hotspots, I, J, gstat):
    hotspots[I]['Members'] += hotspots[J]['Members']
    hotspots[I]['Gi'] = gstat
    hotspots[I]['IsSeed'] = True
    del hotspots[J]

    if J!=-1:
        nnmat[I, :] = nnmat[I, :] | nnmat[J, :]
    else:
        nnmat[I, :] = np.any(nnmat[np.array([I, J]), :], axis=0)

    nnmat[:, I] = nnmat[I, :].T
    nnmat[I, I] = False

    nnmat = np.delete(nnmat, J, axis=0)
    nnmat = np.delete(nnmat, J, axis=1)
    return hotspots, nnmat


# flow_distance
def flow_distance(flow_i, flow_j, dist_mode):
    df = flow_j[:4] - flow_i[:4]
    df = df ** 2
    d = np.inf
    if dist_mode == 1:
        d = np.sqrt(max([df[0] + df[1], df[2] + df[3]]))
    elif dist_mode == 2:
        d = (np.sqrt(df[0] + df[1]) + np.sqrt(df[2] + df[3])) / 2

    v1 = flow_i[2:4] - flow_i[0:2]
    v2 = flow_j[2:4] - flow_j[0:2]
    r = (v1[0] * v2[0] + v1[1] * v2[1]) / (np.linalg.norm(v1) * np.linalg.norm(v2) + np.finfo(float).eps)
    r = 180/np.pi*np.arccos(r)
    return d, r


# fdij
#计算一个数据集中不同样本之间的最大内聚距离和平均内聚距离。函数的输入参数包括三个：flows（数据集）、dist_mode（距离计算模式）。
def fdij(flow_i, flow_j, dist_mode):
    if dist_mode == 1:
        #计算两个向量在x和y方向上的欧几里得距离
        df = flow_j[:4] - flow_i[:4]
        df = df ** 2
        d = np.sqrt(max([df[0] + df[1], df[2] + df[3]]))
        return d
    elif dist_mode == 2:
        #计算两个向量在x和y方向上的欧几里得距离的平均值
        df = flow_j[:4] - flow_i[:4]
        df = df ** 2
        d = (np.sqrt(df[0] + df[1]) + np.sqrt(df[2] + df[3])) / 2
        return d
    else:
        df = flow_j[:4] - flow_i[:4]
        df = df ** 2
        do = df[0] + df[1]
        dd = df[2] + df[3]
        L1 = np.linalg.norm(flow_i[2:4] - flow_i[0:2])
        L2 = np.linalg.norm(flow_j[2:4] - flow_j[0:2])
        d = np.sqrt((do + dd) / (L1 * L2 + np.finfo(float).eps))
        return d


# cluster_inner_distance
#计算一个数据集中不同样本之间的最大内聚距离和平均内聚距离。
def cluster_inner_distance(flows, ids, dist_mode):
    dmax = -np.inf
    davg = 0
    n = len(ids)
    count = 0
    for i in range(n - 1):
        flow_i = flows[ids[i], :]
        for j in range(i + 1, n):
            flow_j = flows[ids[j], :]
            d = fdij(flow_i, flow_j, dist_mode)
            dmax = max(dmax, d)
            davg += d
            count += 1
    if count == 0:
        davg = 0  # 或者设置为其他的默认值
    else:
        davg = davg / count
    return dmax, davg


# cluster_direction_similarity
def cluster_direction_similarity(flows, c1_ids, c2_ids):
    v1 = np.mean(flows[c1_ids, :], axis=0)
    v1 = v1[2:4] - v1[0:2]
    v2 = np.mean(flows[c2_ids, :], axis=0)
    v2 = v2[2:4] - v2[0:2]
    dr = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + np.finfo(float).eps)
    dr = 180/np.pi*np.arccos(dr)
    return dr


# cluster_between_distance
def cluster_between_distance(flows, c1_ids, c2_ids, dist_mode):
    n1 = len(c1_ids)
    n2 = len(c2_ids)
    dmin = np.inf
    davg = 0
    count = 0
    for i in range(n1):
        flow_i = flows[c1_ids[i], :]
        for j in range(n2):
            flow_j = flows[c2_ids[j], :]
            d = fdij(flow_i, flow_j, dist_mode)
            dmin = min(dmin, d)
            davg += d
            count += 1
    davg = davg / count
    return dmin, davg

# cluster_merge
def cluster_merge(flows, hotspots, dist_mode):
    
    num_hotspots = len(hotspots)
    # print(num_hotspots)
    mnums = np.array([len(hotspot['Members']) for hotspot in hotspots])
    # mnums = []
    # for i in range(num_hotspots):
    #     # print(list_data[i])
    #     members_length = len(list_data[i]['Members'])
    #     print(f"Hotspot {i} has {members_length} members")
    #     mnums.append(members_length)
    ind = np.argsort(mnums)[::-1] #排序,并返回排序后的索引
    # print(ind)
    flag = False
    II = -1
    JJ = -1
    max_angle = 30  # degree
    for i in range(len(ind) - 1):
        I = ind[i]
        for j in range(i + 1, len(ind)):
            J = ind[j]
            A = hotspots[I]['Members']
            B = hotspots[J]['Members']
            if len(A)>0 and len(B)>0:
                dmin, davg = cluster_between_distance(flows, A, B, dist_mode)
                dr = cluster_direction_similarity(flows, A, B)
                if (davg <= 1.2 * min(hotspots[I]['InnerDis'], hotspots[J]['InnerDis'])) and (dr <= max_angle):
                    II = I
                    JJ = J
                    flag = True
                    break
            else:
                continue
        if flag:
            break
    if II != -1 and JJ != -1:
        hotspots[II]['Members'] = np.unique(np.concatenate((hotspots[II]['Members'], hotspots[JJ]['Members'])))
        dmax, davg = cluster_inner_distance(flows, hotspots[II]['Members'], dist_mode)
        hotspots[II]['InnerDis'] = davg
        hotspots.pop(JJ)
        st = True
    else:
        st = False
    return hotspots, st


# get_idx
def get_idx(hotspots, n, minpts):
    IDX = np.zeros(n, dtype=int)
    for i in range(len(hotspots)):      
        if len(hotspots[i]['Members']) >= minpts:
            IDX[hotspots[i]['Members']] = i + 1
    return IDX

def save_result(fname, labels):
    if len(labels) < 1:
        print("Error in 'save_result()' : Cluster labels is empty")
        return
    np.savetxt(fname, labels, fmt="%d")


# main
if __name__ == '__main__':
    flow_data_file = 'Data/SD1.csv'
    [flows, flowtimes, groundtruth_labels] = flowread(flow_data_file)

    spatial_R = 300
    angle_Thita  = 30
    [flows, IDX, seeds, flow_density, spatialneighbors] = flowSDBC(flows, flowtimes, spatial_R=spatial_R, angle_T=angle_Thita)
    cluster_result_file = 'Result/flow_labels.txt'
    save_result(cluster_result_file, IDX)


