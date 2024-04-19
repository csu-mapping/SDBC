from flowclass import *
from Clustersclass import *
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm


# Read OD flow
def read_flow(sf, nf):
    flows = list()
    file1 = open(sf, 'r')
    file2 = open(nf, 'r')
    while True:
        str_line1 = file1.readline()
        if not str_line1:
            break
        str_line2 = file2.readline()
        str_lines1 = str_line1.split(',')
        str_lines2 = str_line2.split(',')
        f_id = int(str_lines1[0])
        ox = float(str_lines1[1])
        oy = float(str_lines1[2])
        #otime = float(str_lines1[3])
        dx = float(str_lines2[1])
        dy = float(str_lines2[2])
        #dtime = float(str_lines2[3])
        #new_flow = OD_Flow(f_id, ox, oy, otime, dx, dy, dtime)
        new_flow = OD_Flow(f_id, ox, oy, dx, dy)
        flows.append(new_flow)
    return flows

def FDij(flow_1, flow_2):
    dis_O = (flow_1.OY - flow_2.OY) ** 2 + (flow_1.OX - flow_2.OX) ** 2
    dis_D = (flow_1.DY - flow_2.DY) ** 2 + (flow_1.DX - flow_2.DX) ** 2
    Li = math.sqrt((flow_1.OX - flow_1.DX) ** 2 + (flow_1.OY - flow_1.DY) ** 2)
    Lj = math.sqrt((flow_2.OX - flow_2.DX) ** 2 + (flow_2.OY - flow_2.DY) ** 2)
    try:
        fdij = math.sqrt((dis_O + dis_D) / (Li * Lj))
    except ZeroDivisionError:
        fdij = 100000
    return fdij

# Directional similarity
def calc_angel(flow_1, flow_2):
    """计算两条流之间夹角的sin"""
    d_ox = flow_1.OX - flow_2.OX
    d_oy = flow_1.OY - flow_2.OY
    d_dx = flow_1.DX - flow_2.DX
    d_dy = flow_1.DY - flow_2.DY
    angle1 = math.atan2(d_oy, d_ox)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(d_dy, d_dx)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        angle = abs(angle1 - angle2)
    else:
        angle = abs(angle1) + abs(angle2)
        if angle > 180:
            angle = 360 - angle
    return angle

# Euclidean distance formula Distance metric
def flow_distance(flow_1, flow_2):
    d_ox = flow_1.OX - flow_2.OX
    d_oy = flow_1.OY - flow_2.OY
    d_dx = flow_1.DX - flow_2.DX
    d_dy = flow_1.DY - flow_2.DY
    dis = math.sqrt(d_ox ** 2 + d_oy ** 2) + math.sqrt(d_dx ** 2 + d_dy ** 2)
    return dis/2


# Spatial neighborhood and density measure of OD flow
def flow_neigh(flows, k_distance, eps):
    """
    :param flows: 流数据的集合
    :param r: 距离阈值
    :param sin: 角度阈值

    """
    num = len(flows)
    S = np.zeros(shape=(num, num))  # 初始化 关系矩阵 w 为 n*n的矩阵
    for i in tqdm(range(num)):
        for j in range(i + 1, num, 1):
            distance = flow_distance(flows[i], flows[j])
            angle = calc_angel(flows[i], flows[j])
            if distance < k_distance and angle < eps:
                flows[i].neighs.append(j)
                flows[j].neighs.append(i)
                wij = 1
                S[i][j] = wij
                S[j, i] = S[i, j]
    for l in flows:
        density = len(l.neighs)
        l.density = density
    return flows, S


# Calculate the Gi index
def compute_G(flows, S):
    num = len(S)
    density = np.zeros(shape=(num))
    for i in range(num):
        density[i] = np.sum(S[i, :])
    s = np.std(density)
    z_mean = np.mean(density)
    # density 进行归一化
    nor_density = (density - z_mean) / s
    for i in range(num):
        t_fenzi = np.sum(nor_density * S[i, :])
        #fenmu = (num*density[i]-density[i]**2)/(num-1)
        fenmu = math.sqrt(density[i] * (num - density[i]) / (num - 1))
        flows[i].nor_density = nor_density[i]
        if fenmu == 0:
            flows[i].Gi = 0.0
        else:
            flows[i].Gi = t_fenzi/fenmu
    return flows


# Find hot-pot represent_flows
def find_hotpot(flows, λ):
    represent_flows = list()
    represent_id = list()
    for flow in flows:
        if flow.Gi > λ:
            represent_flows.append([flow.f_id, flow, flow.neighs, flow.density, flow.Gi])      # 0-flow_id; 1-flow_neighbors; 3-flow_Gi
            represent_id.append(flow.f_id)
    return represent_flows


# Initialize the cluster
def identify_cluster(flows):
    """
    初始的clusters C={C1,C2,C3.....,CM}
    C:member 成员 指包含哪些 represent flows
    C:neighbor 邻域有哪些flows
    C:Gi*
    """
    clusters = list()
    m = len(flows)
    for i in range(m):
        C_id = i
        C_member = flows[i].f_id  # flow 实体 member的id
        C_neighs = flows[i].neighs    # cluster neighbors index
        C_GR = flows[i].Gi
        C_Density = flows[i].nor_density
        new_cluster = Cluster(C_id)
        new_cluster.members.append(C_member)
        new_cluster.neighs.append(C_neighs)
        new_cluster.GR = C_GR
        new_cluster.nor_density = C_Density
        clusters.append(new_cluster)
    return clusters

# Find the global largest GR and start merging
def search_max_GR(clusters, represent_flows,  n):
    """
    :param clusters: Ω
    :param represent_flows:  q
    :param represent_id:  id
    :param s: s
    :param z_mean: z_mean
    :param n: length of flow_data
    :return:
    """

    cluster_nums = len(represent_flows)
    I = 0
    J = 0
    max_GR = -999999.0

    for i in tqdm(range(cluster_nums)):
        cur_id = represent_flows[i][0]
        cur_cluster = clusters[cur_id]
        cur_member = cur_cluster.members  # flow_member 流的index
        cur_GR = cur_cluster.GR


        if len(cur_cluster.neighs) == 0:
            continue
        cur_neighbors = cur_cluster.neighs[0]  # index

        m = len(cur_neighbors)
        # 判断neighbors是否为空
        if m == 0:
            continue

        for j in range(m):
            new_density = list()
            nid = cur_neighbors[j]
            ncluster = clusters[nid]
            nmember = ncluster.members
            if len(ncluster.neighs) == 0:
                continue

            nGR = ncluster.GR
            #n_density = ncluster.nor_density
            new_member = [cur_member, nmember]  # id 索引
            new_members = [int(x) for item in new_member for x in item]
            k = len(new_member)
            for i in range(k):
                density = clusters[i].nor_density
                new_density.append(density)
            #print(new_members)
            #print(new_density)
            max_GR_AB = max(nGR, cur_GR)

            current_GR = sum(new_density) / math.sqrt(k * (n - k) / (n - 1))
            if current_GR > max_GR and current_GR > max_GR:
                max_GR = current_GR
                I = cur_id  # 当前的聚类的id
                J = nid  # 当前要合并的flow的id
    return I, J, max_GR



# Update clusters
def update_cluster(I, J, max_Gi, clusters, flows):
    clusters[I].members.extend(clusters[J].members)
    #clusters[I].members = [clusters[I].members, clusters[J].members]
    #clusters[I].members = [int(x) for item in list(clusters[I].members) for x in item]
    clusters[I].neighs[0].extend(clusters[J].neighs[0])
    #clusters[I].neighs = [int(x) for item in list(clusters[I].neighs) for x in item]
    clusters[I].neighs = [list((set(clusters[I].neighs[0])-(set(clusters[I].members))))]
    clusters[I].GR = max_Gi
    clusters[J].neighs.clear()
    return clusters

# Calculate the central flow of the clusters
def cluster_compute(clusters, flows):
    final_cluster = list()
    for cluster in clusters:
        if len(cluster.members) > 3 and len(cluster.neighs) > 0:
          sum_OX = 0
          sum_OY = 0
          sum_DX = 0
          sum_DY = 0
          for ID in cluster.members:
             sum_OX = flows[ID].OX + sum_OX
             sum_DX = flows[ID].DX + sum_DX
             sum_OY = flows[ID].OY + sum_OY
             sum_DY = flows[ID].DY + sum_DY
          ave_OX = sum_OX / len(cluster.members)
          ave_DX = sum_DX / len(cluster.members)
          ave_OY = sum_OY / len(cluster.members)
          ave_DY = sum_DY / len(cluster.members)
          cluster.OX = ave_OX
          cluster.DX = ave_DX
          cluster.DY = ave_DY
          cluster.OY = ave_OY
          cluster.members.extend(cluster.neighs[0])
          final_cluster.append(cluster)
    return final_cluster

# Compute similarities between flows
def FDij(flow_1, flow_2):
    dis_O = (flow_1.OY - flow_2.OY) ** 2 + (flow_1.OX - flow_2.OX) ** 2
    dis_D = (flow_1.DY - flow_2.DY) ** 2 + (flow_1.DX - flow_2.DX) ** 2
    Li = math.sqrt((flow_1.OX - flow_1.DX) ** 2 + (flow_1.OY - flow_1.DY) ** 2)
    Lj = math.sqrt((flow_2.OX - flow_2.DX) ** 2 + (flow_2.OY - flow_2.DY) ** 2)
    try:
        fdij = math.sqrt((dis_O + dis_D) / (Li * Lj))
    except ZeroDivisionError:
        fdij = 100000
    return fdij

# Merge clusters
def merge_final_cluster(final_cluster):
    num = len(final_cluster)
    last_cluster = list()
    for i in tqdm(range(0, num, 1)):
        for j in range(i + 1, num, 1):
            fdij = FDij(final_cluster[i], final_cluster[j])
            if fdij < 0.15:
                final_cluster[i].members.extend(final_cluster[j].members)
                final_cluster[j].members.clear()
    return final_cluster

























