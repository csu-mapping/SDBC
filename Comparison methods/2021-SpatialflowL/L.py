
# --coding: utf-8--
"""
@time: 2023/7/29 15:28
@author: Zhao Yu xin 复现

模拟的数据给映射到[0,1]范围里弄的

@file: L.py
@paper: 2023-ST L Function

"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import math

from scipy.spatial import ConvexHull




"""
This python file is used to calculate the global SpatialflowL

Input flow data:
OX: X coordinate of O point
OY: Y coordinate of O point
DX: X coordinate of D point
DY: Y coordinate of D point

Output the result of global STflowL
r: the spatial scale
L: the global SaptialflowL value

"""

def getRandomPointInCircle(num, radius, centerx, centery):
    xx = []
    yy = []
    for i in range(num):
        while True:
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            if (x ** 2) + (y ** 2) <= (radius ** 2):
                xx.append(x + centerx)
                yy.append(y + centery)
                break
    return np.array(xx), np.array(yy)


def dis_max(ox1, oy1, dx1, dy1, ox2, oy2, dx2, dy2):
    dO = math.sqrt((ox2 - ox1) ** 2 + (oy2 - oy1) ** 2)
    dD = math.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2)
    return max(dO, dD)



def min_enclosing_polygon_area(points):
    # Calculate the convex hull of the points
    hull = ConvexHull(points)
    # Calculate the area of the convex hull
    area = hull.area


    return area


def disMatrix(df_in_points, df_all_points):
    disM = []
    N_df_all = len(df_all_points)
    for index1, row1 in df_in_points.iterrows():
        row1_ox = row1['ox'] * np.ones([1, N_df_all])
        row1_oy = row1['oy'] * np.ones([1, N_df_all])
        row1_dx = row1['dx'] * np.ones([1, N_df_all])
        row1_dy = row1['dy'] * np.ones([1, N_df_all])
        all_ox = (df_all_points['OX'].values).reshape(1, N_df_all)
        all_oy = (df_all_points['OY'].values).reshape(1, N_df_all)
        all_dx = (df_all_points['DX'].values).reshape(1, N_df_all)
        all_dy = (df_all_points['DY'].values).reshape(1, N_df_all)

        o_dis = pow((row1_ox - all_ox) ** 2 + (row1_oy - all_oy) ** 2, 1 / 2)
        d_dis = pow((row1_dx - all_dx) ** 2 + (row1_dy - all_dy) ** 2, 1 / 2)
        dis = np.vstack((o_dis, d_dis))
        disM1 = np.max(dis, axis=0)
        disM.append(disM1)
    return disM

def LocalST_K(df_flows,df_all_flows,lamda,r_scale,result_path):

    N_df_all = len(df_all_flows)
    result = []
    for index1, row1 in df_flows.iterrows():
        print(index1)
        # ----空间距离----
        row1_ox = row1['ox'] * np.ones([1, N_df_all])
        row1_oy = row1['oy'] * np.ones([1, N_df_all])
        row1_dx = row1['dx'] * np.ones([1, N_df_all])
        row1_dy = row1['dy'] * np.ones([1, N_df_all])
        all_ox = (df_all_flows['OX'].values).reshape(1, N_df_all)
        all_oy = (df_all_flows['OY'].values).reshape(1, N_df_all)
        all_dx = (df_all_flows['DX'].values).reshape(1, N_df_all)
        all_dy = (df_all_flows['DY'].values).reshape(1, N_df_all)
        o_dis = pow((row1_ox-all_ox)**2+(row1_oy-all_oy)**2,1/2)
        d_dis = pow((row1_dx-all_dx)**2+(row1_dy-all_dy)**2,1/2)
        dis = np.vstack((o_dis,d_dis))
        disM1 = np.max(dis, axis=0)


        for r in r_scale:

            disMtemp = np.where(disM1 <= r, 1, 0)
            count = np.sum(disMtemp == 1) - 1
            Local_L = pow(count/(lamda*math.pi*math.pi), 1/4)-r
            Local_Lr4 = count / (lamda * math.pi * math.pi) - pow(r, 4)
            result.append([int(row1['ID']), r, count, Local_L, Local_Lr4])
    df_result = pd.DataFrame(result, columns=['ID', 'r',  'count', 'LocalL', 'Local_Lr4'])
    df_result.to_csv(result_path, index=False, header=True)
    return df_result

def draw_KL(r, localK, localL, step):
    # 将r_values, K_values, Lr4_values转换为np.array格式
    r_array = np.array(r)
    K_array = np.array(localK)
    L_array = np.array(localL)

    s = 'sim' + str(step)

    plt.figure()
    plt.title(s , fontsize=16)
    plt.plot(r_array, np.pi ** 2 * r_array ** 4, '-^r', markersize=3)
    plt.plot(r_array, K_array, '-')
    # 添加横坐标和纵坐标标注
    plt.xlabel('Distance r',fontsize=14)
    plt.ylabel('K(r)',fontsize=14)
    # 设置坐标刻度的字体大小
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()

    plt.figure()
    plt.plot([0, max(r)], [0, 0], '-r')
    plt.plot(r_array, L_array, '-')
    plt.axis([0, max(r), -0.05, 0.05])
    plt.xlabel('Distance r',fontsize=14)
    plt.ylabel('L(r)',fontsize=14)
    # 设置坐标刻度的字体大小
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title(s, fontsize=16)
    plt.show()



def main():

    for step in range(1, 8):
        # for step in range(1, 200):
        print(step)
        df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_' + str(
            step) + '.csv'
        df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')  # Input the flows within study domain
        df_all_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_' + str(
            step) + '.csv'
        df_all_flows = pd.read_csv(df_all_flows_path, encoding='utf-8')  # Input the flows within the buffer (for avoiding edge effects)
        global_L_result_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/Global_L/Global_L_'+ str(
            step) + '.csv'   # Ouput global_L values results (a flie).


        disM = np.array(disMatrix(df_in_flows, df_all_flows))


        lamda = len(df_in_flows) / (((1 * 1) ** 2)) # Set flow density manually

        # points_list_O = list(zip(df_in_flows['ox'], df_in_flows['oy']))  # Convert DataFrame to points_list
        # points_list_D = list(zip(df_in_flows['dx'], df_in_flows['dy']))
        # #print(points_list_O)
        #
        #
        # O_area = min_enclosing_polygon_area(points_list_O)
        # D_area = min_enclosing_polygon_area(points_list_D)
        #
        # print("Area of the minimum bounding polygonO:", O_area)
        # print("Area of the minimum bounding polygonO:", D_area)
        # lamda = len(df_in_flows) / (O_area * D_area)

        result = []
        r_scale = np.linspace(0.01, 0.6, 60)

        for r in np.linspace(0.01, 0.6, 60):
            print(r)
            disMtemp = np.where(disM <= r, 1, 0)

            count = np.sum(disMtemp == 1) - len(df_in_flows)
            localK = (count / len(df_in_flows)) / lamda
            localL = (localK / (math.pi ** 2)) ** (1 / 4) - r
            Lr4 = localK / pow(math.pi, 2) - pow(r, 4)

            result.append([r, localK, localL, Lr4])


        df_result = pd.DataFrame(result, columns=['r', 'localK', 'localL', 'Lr4'])
        r = df_result['r'].values
        localK = df_result['localK'].values
        localL = df_result['localL'].values
        df_result.to_csv(global_L_result_path, index=False, header=True)
        draw_KL(r, localK, localL,step)

def test():
    for step in range(1, 8):
        # for step in range(1, 200):
        print(step)
        df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows/flows_in_' + str(
            step) + '.csv'
        df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')  # Input the flows within study domain
        df_all_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/buffer_edge/flows_all_' + str(
            step) + '.csv'
        df_all_flows = pd.read_csv(df_all_flows_path, encoding='utf-8')  # Input the flows within the buffer (for avoiding edge effects)
        global_L_result_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/Global_L/Global_L_'+ str(
            step) + '.csv'   # Ouput global_L values results (a flie).


        points_list_O = list(zip(df_in_flows['ox'], df_in_flows['oy']))  # Convert DataFrame to points_list
        points_list_D = list(zip(df_in_flows['dx'], df_in_flows['dy']))
        #print(points_list_O)

        O_area = min_enclosing_polygon_area(points_list_O)
        D_area = min_enclosing_polygon_area(points_list_D)

        print("Area of the minimum bounding polygonO:", O_area)
        print("Area of the minimum bounding polygonO:", D_area)

        disM = np.array(disMatrix(df_in_flows, df_all_flows))

        #lamda = len(df_in_flows) / (((1 * 1) ** 2)) # Set flow density manually
        lamda = len(df_in_flows) / (O_area * D_area)
        #lamda = lamda * 1000000000
        r_scale = np.linspace(1000, 8000, 100)
        LocalST_K(df_in_flows, df_all_flows, lamda, r_scale, global_L_result_path)

def test2():
    xv = np.array([0, 0, 1, 1, 0])
    yv = np.array([0, 1, 1, 0, 0])
    o_area = np.vstack((xv, yv)).T
    d_area = np.vstack((xv, yv)).T

    scale = np.linspace(0.01, 0.6, 60)

    df_in_flows_arr = []
    df_all_flows_arr = []
    num = 1000  # number of flows
    n_in = 0
    while n_in <= num:
        ox = random.uniform(-0.6, 1.6)
        oy = random.uniform(-0.6, 1.6)
        dx = random.uniform(-0.6, 1.6)
        dy = random.uniform(-0.6, 1.6)

        if (ox >= 0) & (ox <= 1) & (oy >= 0) & (oy <= 1) & (dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1) :
            df_in_flows_arr.append([ox, oy, dx, dy])
            n_in = n_in + 1
        df_all_flows_arr.append([ox, oy, dx, dy])


    # 计算o_area和d_area的面积
    area_o = abs(ConvexHull(o_area).volume)
    area_d = abs(ConvexHull(d_area).volume)

    print("Area of o_area:", area_o)
    print("Area of d_area:", area_d)

    df_in_flows = pd.DataFrame(df_in_flows_arr, columns=['ox', 'oy', 'dx', 'dy'])

    df_all_flows = pd.DataFrame(df_all_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])
    df_in_flows['ID'] = df_in_flows.index
    df_all_flows['ID'] = df_all_flows.index
    disM = np.array(disMatrix(df_in_flows, df_all_flows))

    lamda = len(df_in_flows) / (((1 * 1) ** 2)) # Set flow density manually

    result = []

    for r in np.linspace(0.01, 0.6, 60):
        #print(r)
        disMtemp = np.where(disM <= r, 1, 0)

        count = np.sum(disMtemp == 1) - len(df_in_flows)
        localK = (count / len(df_in_flows)) / lamda
        localL = (localK / (math.pi ** 2)) ** (1 / 4) - r
        Lr4 = localK / pow(math.pi, 2) - pow(r, 4)

        result.append([r, localK, localL, Lr4])

    df_result = pd.DataFrame(result, columns=['r', 'localK', 'localL','Lr4'])
    r = df_result['r'].values
    localK = df_result['localK'].values
    localL = df_result['localL'].values
    print(max(localL))

    draw_KL(r, localK, localL)

def round2():
    step = 'sim7_R3'
    df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_' + str(
                step) + '.csv'
    df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')  # Input the flows within study domain
    df_all_flows_path ='/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_' + str(
            step) + '.csv'
    df_all_flows = pd.read_csv(df_all_flows_path,
                               encoding='utf-8')  # Input the flows within the buffer (for avoiding edge effects)
    global_L_result_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/Global_L/Global_L_7_round2.csv'  # Ouput global_L values results (a flie).

    disM = np.array(disMatrix(df_in_flows, df_all_flows))

    lamda = len(df_in_flows) / (((1 * 1) ** 2))  # Set flow density manually

    # points_list_O = list(zip(df_in_flows['ox'], df_in_flows['oy']))  # Convert DataFrame to points_list
    # points_list_D = list(zip(df_in_flows['dx'], df_in_flows['dy']))
    # #print(points_list_O)
    #
    #
    # O_area = min_enclosing_polygon_area(points_list_O)
    # D_area = min_enclosing_polygon_area(points_list_D)
    #
    # print("Area of the minimum bounding polygonO:", O_area)
    # print("Area of the minimum bounding polygonO:", D_area)
    # lamda = len(df_in_flows) / (O_area * D_area)

    result = []
    r_scale = np.linspace(0.01, 0.6, 60)

    for r in np.linspace(0.01, 0.6, 60):
        print(r)
        disMtemp = np.where(disM <= r, 1, 0)

        count = np.sum(disMtemp == 1) - len(df_in_flows)
        localK = (count / len(df_in_flows)) / lamda
        localL = (localK / (math.pi ** 2)) ** (1 / 4) - r
        Lr4 = localK / pow(math.pi, 2) - pow(r, 4)

        result.append([r, localK, localL, Lr4])

    df_result = pd.DataFrame(result, columns=['r', 'localK', 'localL', 'Lr4'])
    r = df_result['r'].values
    localK = df_result['localK'].values
    localL = df_result['localL'].values
    df_result.to_csv(global_L_result_path, index=False, header=True)
    draw_KL(r, localK, localL,step='7_round3')




if __name__ == '__main__':
    #main()


    round2()
    #test()
    #test2()



