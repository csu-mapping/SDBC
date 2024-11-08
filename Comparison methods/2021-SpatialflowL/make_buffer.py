# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
import math
from sklearn.preprocessing import MinMaxScaler


"""
This python file is used to generate buffer of research region.

Input :

num: number of flows in each run
step: number of simulations

Output:

df_result: the global ST-flowL results in each run

"""

# def geo_shift(min_val, max_val):
#     # 该函数将地理坐标映射到[0,1]内
#
#     # 假设coords是包含地理坐标的NumPy数组，其中每一行是一个坐标点，第一列是X坐标，第二列是Y坐标
#     # 示例数据，用来说明方法
#     coords = np.array([[10, 20],
#                        [15, 25],
#                        [30, 40],
#                        [20, 35]])
#
#     # 找到最小值和最大值
#     min_val = np.min(coords, axis=0)
#     max_val = np.max(coords, axis=0)
#
#     x_min_boundary = df_in_flows['ox'].min()
#     x_max_boundary = df_in_flows['ox'].max()
#
#     # Min-Max缩放，将坐标映射到[0, 1]之间
#
#     x_ave = (x_max_boundary - x_min_boundary) / 2
#     y_min_boundary = min(df_in_flows['oy'].min(), df_in_flows['dy'].min())
#     y_max_boundary = max(df_in_flows['oy'].max(), df_in_flows['dy'].max())
#     y_ave = (y_max_boundary - y_min_boundary) / 2
#
#
#     coords_scaled = (coords - min_val) / (max_val - min_val)
#
#     print("原始坐标数据：")
#     print(coords)
#     print("\n缩放后的坐标数据：")
#     print(coords_scaled)
#



#  坐标缩放[0,1]内
def normalize_coordinates(coordinates):

    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)

    normalized_coords = (coordinates - min_coords) / (max_coords - min_coords)


    print("\n缩放后的坐标数据：")
    print(normalized_coords)

    return normalized_coords


def old_version():
    for step in range(1, 8):
        # for step in range(1, 200):
        print(step)
        df_in_flows_arr = []
        df_all_flows_arr = []
        # num = 400
        df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows/flows_in_' + str(
            step) + '.csv'
        df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')
        n_in = 0
        num = len(df_in_flows) * 4  # number of flows

        x_min_boundary = min(df_in_flows['ox'].min(), df_in_flows['dx'].min())
        x_max_boundary = max(df_in_flows['ox'].max(), df_in_flows['dx'].max())

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_in_flows[['ox', 'oy', 'dx', 'dy']] = scaler.fit_transform(df_in_flows[['ox', 'oy', 'dx', 'dy']])

        x_ave = (x_max_boundary - x_min_boundary) / 2
        y_min_boundary = min(df_in_flows['oy'].min(), df_in_flows['dy'].min())
        y_max_boundary = max(df_in_flows['oy'].max(), df_in_flows['dy'].max())
        y_ave = (y_max_boundary - y_min_boundary) / 2

        while n_in <= num:

            ox = random.uniform(x_min_boundary - x_ave, x_max_boundary + x_ave)
            oy = random.uniform(y_min_boundary - x_ave, y_max_boundary + x_ave)
            dx = random.uniform(x_min_boundary - y_ave, x_max_boundary + y_ave)
            dy = random.uniform(y_min_boundary - y_ave, y_max_boundary + y_ave)
            n_in = n_in + 1
            if (ox >= x_min_boundary) & (ox <= x_max_boundary) & (oy >= y_min_boundary) & (oy <= y_max_boundary) & (
                    dx >= x_min_boundary) & (dx <= x_max_boundary) & (dy >= y_min_boundary) & (dy <= y_max_boundary):
                df_in_flows_arr.append([ox, oy, dx, dy])
            else:
                df_all_flows_arr.append([ox, oy, dx, dy])

        # df_in_flows = pd.DataFrame(df_in_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])

        df_all_flows = pd.DataFrame(df_all_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_all_flows[['OX', 'OY', 'DX', 'DY']] = scaler.fit_transform(df_all_flows[['OX', 'OY', 'DX', 'DY']])

        df_all_flows['ID'] = df_all_flows.index
        print(len(df_all_flows))
        df_all_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/buffer_edge/flows_all_' + str(
                step) + '.csv', index=False, header=True)


def new_version():
    for step in range(1, 8):
        # for step in range(1, 200):
        print(step)
        df_in_flows_arr = []
        df_all_flows_arr = []
        # num = 400
        df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows/flows_in_' + str(
            step) + '.csv'
        df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')
        num = len(df_in_flows) # number of flows

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_in_flows[['ox', 'oy', 'dx', 'dy']] = scaler.fit_transform(df_in_flows[['ox', 'oy', 'dx', 'dy']])
        df_in_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_' + str(
                step) + '.csv', index=False, header=True)

        density_in = len(df_in_flows) / ((1*1)**2)


        while (num / ((1.4*1.4)**2)) < density_in :
            ox = random.uniform(-0.2, 1.2)
            oy = random.uniform(-0.2, 1.2)
            dx = random.uniform(-0.2, 1.2)
            dy = random.uniform(-0.2, 1.2)

            if (ox >= 0) & (ox <= 1) & (oy >= 0) & (oy <= 1) & (dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1):
                df_in_flows_arr.append([ox, oy, dx, dy]) # 这个是随机的 flows_in

            else:
                df_all_flows_arr.append([ox, oy, dx, dy])
                num = num+1
                density_all = num / ((1.4*1.4)**2)
        print(density_in)
        print(density_all)

        # df_in_flows = pd.DataFrame(df_in_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])

        df_all_flows = pd.DataFrame(df_all_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_all_flows[['OX', 'OY', 'DX', 'DY']] = scaler.fit_transform(df_all_flows[['OX', 'OY', 'DX', 'DY']])

        df_all_flows['ID'] = df_all_flows.index
        print(len(df_all_flows))
        df_all_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_' + str(
                step) + '.csv', index=False, header=True)






def new0827():
    for step in range(1, 8):
        # for step in range(1, 200):
        print(step)
        df_in_flows_arr = []
        df_all_flows_arr = []
        # num = 400
        df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows/flows_in_' + str(
            step) + '.csv'
        df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')
        num = len(df_in_flows) # number of flows

        # 提取DataFrame的某一列为NumPy数组
        ox = df_in_flows['ox'].values
        oy = df_in_flows['oy'].values
        ox = normalize_coordinates(ox)
        oy = normalize_coordinates(oy)
        dx = df_in_flows['dx'].values
        dy = df_in_flows['dy'].values
        dx = normalize_coordinates(dx)
        dy = normalize_coordinates(dy)
        print(ox)
        df_in_flows['ox'] = ox
        df_in_flows['oy'] = oy
        df_in_flows['dx'] = dx
        df_in_flows['dy'] = dy
        print(df_in_flows)

        df_in_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_' + str(
                step) + '.csv', index=False, header=True)

        density_in = len(df_in_flows) / ((1 * 1) ** 2)

        while (num / ((1.4 * 1.4) ** 2)) < density_in:
            ox = random.uniform(-0.2, 1.2)
            oy = random.uniform(-0.2, 1.2)
            dx = random.uniform(-0.2, 1.2)
            dy = random.uniform(-0.2, 1.2)

            if (ox >= 0) & (ox <= 1) & (oy >= 0) & (oy <= 1) & (dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1):
                df_in_flows_arr.append([ox, oy, dx, dy])  # 这个是随机的 flows_in

            else:
                df_all_flows_arr.append([ox, oy, dx, dy])
                num = num + 1
                density_all = num / ((1.4 * 1.4) ** 2)
        print(density_in)
        print(density_all)

        # df_in_flows = pd.DataFrame(df_in_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])

        df_all_flows = pd.DataFrame(df_all_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_all_flows[['OX', 'OY', 'DX', 'DY']] = scaler.fit_transform(df_all_flows[['OX', 'OY', 'DX', 'DY']])

        df_all_flows['ID'] = df_all_flows.index
        print(len(df_all_flows))
        df_all_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_' + str(
                step) + '.csv', index=False, header=True)

def round2():
    step = 'sim7_R3'
    df_in_flows_arr = []
    df_all_flows_arr = []
    # num = 400
    df_in_flows_path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/SpatialGlobalL/SpatialflowL_cluster/7_del_3.csv'
    df_in_flows = pd.read_csv(df_in_flows_path, encoding='utf-8')
    num = len(df_in_flows)  # number of flows

    # 提取DataFrame的某一列为NumPy数组
    ox = df_in_flows['ox'].values
    oy = df_in_flows['oy'].values
    ox = normalize_coordinates(ox)
    oy = normalize_coordinates(oy)
    dx = df_in_flows['dx'].values
    dy = df_in_flows['dy'].values
    dx = normalize_coordinates(dx)
    dy = normalize_coordinates(dy)
    print(ox)
    df_in_flows['ox'] = ox
    df_in_flows['oy'] = oy
    df_in_flows['dx'] = dx
    df_in_flows['dy'] = dy
    print(df_in_flows)

    df_in_flows.to_csv(
            '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_' + str(
                step) + '.csv', index=False, header=True)

    density_in = len(df_in_flows) / ((1 * 1) ** 2)

    while (num / ((1.4 * 1.4) ** 2)) < density_in:
        ox = random.uniform(-0.2, 1.2)
        oy = random.uniform(-0.2, 1.2)
        dx = random.uniform(-0.2, 1.2)
        dy = random.uniform(-0.2, 1.2)

        if (ox >= 0) & (ox <= 1) & (oy >= 0) & (oy <= 1) & (dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1):
            df_in_flows_arr.append([ox, oy, dx, dy])  # 这个是随机的 flows_in

        else:
            df_all_flows_arr.append([ox, oy, dx, dy])
            num = num + 1
            density_all = num / ((1.4 * 1.4) ** 2)
    print(density_in)
    print(density_all)

    # df_in_flows = pd.DataFrame(df_in_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])

    df_all_flows = pd.DataFrame(df_all_flows_arr, columns=['OX', 'OY', 'DX', 'DY'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_all_flows[['OX', 'OY', 'DX', 'DY']] = scaler.fit_transform(df_all_flows[['OX', 'OY', 'DX', 'DY']])

    df_all_flows['ID'] = df_all_flows.index
    print(len(df_all_flows))
    df_all_flows.to_csv(
        '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_' + str(
            step) + '.csv', index=False, header=True)


if __name__ == '__main__':
    #new_version()
    #new0827()
    round2()


