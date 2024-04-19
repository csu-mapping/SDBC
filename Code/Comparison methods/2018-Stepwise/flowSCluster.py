# -*- coding: utf-8 -*-
# Zhaoyuxin 对比实验 改
# 2024-02-01

import os
import math
import csv
import psycopg2
import time
import sys
import heapq
from math import sqrt
from flowClass import *
import time



#
# # 读取OD点坐标
# def readData(fileName):
#     data = []
#     with open(fileName, 'r') as f:
#         while True:
#             line = f.readline()
#             if line:
#                 sl = line.split(',')
#                 if len(sl) > 1:
#                     d = [float(sl[1]), float(sl[2]), float(sl[3]), float(sl[4])]
#                     data.append(d)
#             else:
#                 break
#
#     return data

# 读取OD流文件
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


# 计算第i个数据中点的k个近邻点,返回近邻点序号列表
def KNN(i, k):
    conn = psycopg2.connect(database="flow clustering", user="postgres", password="123", host='localhost', port="5432")
    cur = conn.cursor()

    cur.execute('select tgid, midpnt <-> (select midpnt from taxi_odt where tgid = ' + str(
        i) + ') dist from taxi_odt order by dist limit ' + str(k + 1) + ';')
    results = cur.fetchall()
    n = []
    for row in results:
        if row[0] != i:
            n.append(row[0])

    conn.commit()
    cur.close()
    conn.close()
    return n

# 假设points是一个坐标列表，每个坐标形如(x, y)
#points = [(x1, y1), (x2, y2), ..., (xn, yn)]


# 欧式距离公式 距离度量
def flow_distance(flow_1, flow_2):
    d_ox = flow_1.OX - flow_2.OX
    d_oy = flow_1.OY - flow_2.OY
    d_dx = flow_1.DX - flow_2.DX
    d_dy = flow_1.DY - flow_2.DY
    dis = sqrt(d_ox ** 2 + d_oy ** 2) + sqrt(d_dx ** 2 + d_dy ** 2)
    return dis/2


# 计算第i条的k个近邻点
def Knn(data, i, k):
    distances = []
    for index in range(len(data)):
        if index != i:
            dist = flow_distance(data[i], data[index])
            # 使用堆来保持距离最小的k个点
            if len(distances) < k:
                heapq.heappush(distances, (-dist, index))
            else:
                heapq.heappushpop(distances, (-dist, index))

    # 取出最近的k个点的索引
    neighbors = [index for _, index in distances]
    return neighbors



# 计算cluster的中心流坐标
def calcClusterFlow(c, data):
    ox = 0
    oy = 0
    dx = 0
    dy = 0
    for k in c:
        ox += data[k].OX
        oy += data[k].OY
        dx += data[k].DX
        dy += data[k].DY
    d = float(len(c))

    ox /= d
    oy /= d
    dx /= d
    dy /= d
    return ox, oy, dx, dy


def flowSim(vi, vj, alpha):
    leni = math.sqrt((vi[0] ** 2 + vi[1] ** 2))
    lenj = math.sqrt((vj[0] ** 2 + vj[1] ** 2))
    dv = math.sqrt((vi[0] - vj[0]) ** 2 + (vi[1] - vj[1]) ** 2)
    if leni > lenj:
        return dv / (alpha * leni)
    else:
        return dv / (alpha * lenj)


# 计算clusterID为ci和cj的两个类的相似性
def clusterSim(ci, cj, data, alpha):
    oix, oiy, dix, diy = calcClusterFlow(ci, data)
    ojx, ojy, djx, djy = calcClusterFlow(cj, data)

    vi = [dix - oix, diy - oiy]
    vj = [djx - ojx, djy - ojy]
    return flowSim(vi, vj, alpha)


# 合并相似度高的类
def merge(c, ci, cj, l):
    # 保留小数字的clusterID
    if ci > cj:
        ci, cj = cj, ci

    for lid in c[cj]:
        l[lid] = ci
        c[ci].append(lid)
    c.pop(cj)


# 输出带类标签的OD数据到csv格式文件
def outputSLabeledData(filename, data, l):
    rf = open(filename, 'w', newline='')
    sheet = csv.writer(rf)
    sheet.writerow(['id', 'x1', 'y1', 'x2', 'y2','cluster'])
    for i in range(len(data)):
        r = [i]
        #r.extend(data[i])
        r.append(data[i].OX)
        r.append(data[i].OY)
        r.append(data[i].DX)
        r.append(data[i].DY)
        # r.append(let[i])
        # r.append(w[i])
        r.append(l[i])
        sheet.writerow(r)
    rf.close()


# 输出空间类数据，包括clusterID，类中心流坐标，包含的流的个数
def outputSClusterData(filename, data, c):
    rf = open(filename, 'w', newline='')
    sheet = csv.writer(rf)
    sheet.writerow(['clusterID', 'ox', 'oy', 'dx', 'dy', 'flownum'])
    for i in c.keys():
        if len(c[i]) > 0:
            ox, oy, dx, dy = calcClusterFlow(c[i], data)
            sheet.writerow([i, ox, oy, dx, dy, len(c[i])])
    rf.close()


if __name__ == '__main__':
    print('Running ', sys.argv[0])
    startTime = time.clock()

    # 空间聚类参数
    alpha = 0.55  # 边界圆尺度系数
    K = 25  # 近邻数
    path = './sim experiment'
    sf = path + '/sim_7o.txt'
    nf = path + '/sim_7d.txt'


    # dataFile = path + 'taxi data(May 13)_processed.csv'
    #dataFile = path + '/sim_7.csv'
    ldataFile = 's_ld(May 13) ' + str(K) + ' ' + str(alpha) + '.csv'
    clusterFile = 's_c(May 13) ' + str(K) + ' ' + str(alpha) + '.csv'

    #print('file: ', dataFile)
    print('alpha =', alpha, '; k =', K)

    # ----------------------------初始化------------------------------------
    print('\ninitialize...')

    #data = readData(dataFile)
    data = read_flow(sf,nf)
    dataLen = len(data)
    #Point = []   # 数据点point集合
    #for i in range(dataLen):


    c = {}  # 类集合
    l = []  # 数据标签集合

    # ----------------------------空间聚类----------------------------------
    # 初始化时第i类只包括第i个数据，第i个数据的数据标签为第i类
    for i in range(dataLen):
        c[i] = [i]  # 类编号(整数编号)，包含的流编号
        l.append(i)  # 流的类标签

    print('start clustering...')
    st = time.perf_counter()
    for i in range(dataLen):
        if i % 5000 == 0:
            et = time.clock()
            print(i, '%.2f' % ((et - st) / 60.0), 'mins')
            st = et

        #knn = KNN(i, K)  # 计算k近邻点
        knn = Knn(data,i,K)

        for j in knn:
            if l[i] != l[j]:  # 如果第i条流和第j条流不属于同一类
                if not (clusterSim(c[l[i]], c[l[j]], data, alpha) > 1):
                    merge(c, l[i], l[j], l)

    if os.path.exists(ldataFile):
        os.remove(ldataFile)
    if os.path.exists(clusterFile):
        os.remove(clusterFile)

    #outputSLabeledData(ldataFile, data, l, lst, let, w)
    outputSLabeledData(ldataFile, data, l)
    outputSClusterData(clusterFile, data, c)

    endTime = time.perf_counter()
    print('Total running time: %.2f' % ((endTime - startTime) / 3600.0), 'hours')
