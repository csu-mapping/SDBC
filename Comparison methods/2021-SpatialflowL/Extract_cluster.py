# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
import math

"""
This python file is used to extract the dominant cluster

Input flow data:
OX: X coordinate of O point
OY: Y coordinate of O point
DX: X coordinate of D point
DY: Y coordinate of D point


Output the dominant cluster extraction result:
ID: the ID of flow
obs_type: if value=1, the flow is extracted as the dominant cluster, if not, the flow is considered as the noise flow.
OX: X coordinate of O point
OY: Y coordinate of O point
DX: X coordinate of D point
DY: Y coordinate of D point

"""

def LocalST_K(df_flows,df_all_flows,lamda,r_scale):

    N_df_all = len(df_all_flows)
    result=[]
    for index1, row1 in df_flows.iterrows():
        print(index1)

        row1_ox=row1['OX']*np.ones([1,N_df_all])
        row1_oy = row1['OY'] * np.ones([1, N_df_all])
        row1_dx = row1['DX'] * np.ones([1, N_df_all])
        row1_dy = row1['DY'] * np.ones([1, N_df_all])
        all_ox=(df_all_flows['OX'].values).reshape(1,N_df_all)
        all_oy = (df_all_flows['OY'].values).reshape(1, N_df_all)
        all_dx = (df_all_flows['DX'].values).reshape(1, N_df_all)
        all_dy = (df_all_flows['DY'].values).reshape(1, N_df_all)
        o_dis=pow((row1_ox-all_ox)**2+(row1_oy-all_oy)**2,1/2)
        d_dis=pow((row1_dx-all_dx)**2+(row1_dy-all_dy)**2,1/2)
        dis=np.vstack((o_dis,d_dis))
        disM1=np.max(dis, axis=0)

        disMtemp = np.where(disM1 <= r_scale, 1, 0)
        count = np.sum(disMtemp == 1)

        Local_L = count / (lamda * math.pi * math.pi) - pow(r_scale, 4)
        result.append([int(row1['ID']),r_scale, count, Local_L])
    df_result = pd.DataFrame(result, columns=['ID', 'r',  'count', 'LocalL'])

    return df_result


def searchmaxCluster(df_flows,df_LocalST_K,topNum,r_scale):
    df_LocalST_K = pd.merge(df_flows, df_LocalST_K, on=['ID'])
    df_LocalST_K.sort_values(by=['LocalL'], ascending=False, inplace=True)
    df_LocalST_K_top = df_LocalST_K.head(topNum)
    result_obs_type = []
    N_df_LocalST_K_top = len(df_LocalST_K_top)
    for index1, row1 in df_flows.iterrows():
        print(index1)
        # ----空间距离----
        row1_ox = row1['OX'] * np.ones([1, N_df_LocalST_K_top])
        row1_oy = row1['OY'] * np.ones([1, N_df_LocalST_K_top])
        row1_dx = row1['DX'] * np.ones([1, N_df_LocalST_K_top])
        row1_dy = row1['DY'] * np.ones([1, N_df_LocalST_K_top])
        all_ox = (df_LocalST_K_top['OX'].values).reshape(1, N_df_LocalST_K_top)
        all_oy = (df_LocalST_K_top['OY'].values).reshape(1, N_df_LocalST_K_top)
        all_dx = (df_LocalST_K_top['DX'].values).reshape(1, N_df_LocalST_K_top)
        all_dy = (df_LocalST_K_top['DY'].values).reshape(1, N_df_LocalST_K_top)
        o_dis = pow((row1_ox - all_ox) ** 2 + (row1_oy - all_oy) ** 2, 1 / 2)
        d_dis = pow((row1_dx - all_dx) ** 2 + (row1_dy - all_dy) ** 2, 1 / 2)
        dis = np.vstack((o_dis, d_dis))
        disM1 = np.max(dis, axis=0)
        disMtemp = np.where(disM1 <= r_scale, 1, 0)
        sum = np.sum(disMtemp == 1)
        obs_type = 0
        if sum != 0:
            obs_type = 1
        result_obs_type.append(
            [row1['ID'], obs_type, row1['OX'], row1['OY'], row1['DX'], row1['DY']])
    df_result_obs_type = pd.DataFrame(result_obs_type,
                                      columns=['ID',  'obs_type', 'OX', 'OY','DX', 'DY'])

    return df_result_obs_type


df_in_flows = pd.read_csv(r'/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_in/flows_in_7.csv')  # Input the flows within study domain
df_all_flows = pd.read_csv(r'/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/flows_all/flows_all_7.csv')  # Input the flows within the buffer (for avoiding edge effects)

lamda = len(df_in_flows)/(((1*1)**2)) # Set flow density manually

#r_scale = 0.29  # obtianed from "L.py" file
r_scale = 0.3 # obtianed from "L.py" file

df_LocalST_K = LocalST_K(df_in_flows, df_all_flows, lamda, r_scale)

#---2寻找簇
topNum = int(0.01*(len(df_in_flows)))
obs_cluster_path = r'/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/2023-Spatiotemporal Flow L-function/SpatialflowL/SpatialGlobalL/SpatialflowL_cluster/SpatialflowL_cluster_7_0222-2.csv'

df_result_obs_type = searchmaxCluster(df_in_flows, df_LocalST_K, topNum, r_scale)
df_result_obs_type.to_csv(obs_cluster_path, index=False, header=True)
