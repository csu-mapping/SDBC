import pandas as pd
import math
import numpy as np
from tqdm import trange, tqdm


# 流类
class OD_Flow(object):
    # def __init__(self, f_id, ox, oy, otime, dx, dy, dtime):
    def __init__(self, f_id, ox, oy, dx, dy):
        self.f_id = f_id
        self.OX = ox
        self.OY = oy
        self.DX = dx
        self.DY = dy
        # self.Otime = otime
        # self.Dtime = dtime
        self.neighs = list()
        self.neighs_id = list()
        self.density = 0
        #self.wij = 0  # 指示函数
        self.Gi = 0
        self.nor_density = 0
        self.index = 0


# 读取OD流文件
def read_flow(data_file):
    d = np.loadtxt(data_file, delimiter=',', skiprows=1)
    flows = list()
    for i in range(d.shape[0]):
        flows.append(OD_Flow(d[i, 0], d[i, 1], d[i, 2], d[i, 3], d[i, 4]))
    return flows


# Euclidean distance formula Distance metric
def flow_distance(flow_1, flow_2):
    d_ox = flow_1.OX - flow_2.OX
    d_oy = flow_1.OY - flow_2.OY
    d_dx = flow_1.DX - flow_2.DX
    d_dy = flow_1.DY - flow_2.DY
    dis = math.sqrt(d_ox ** 2 + d_oy ** 2) + math.sqrt(d_dx ** 2 + d_dy ** 2)
    return dis/2


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


def calculate_neighborhood_density(flows, R , eps):
    """
        :param flows: 流数据的集合
        :param r: 距离阈值
        :param sin: 角度阈值
    """
    den_list = list()
    num = len(flows)
    S = np.zeros(shape=(num, num))  # 初始化 关系矩阵 w 为 n*n的矩阵
    for i in range(num):
        for j in range(i + 1, num, 1):
            distance = flow_distance(flows[i], flows[j])
            #disO, disD = flow_distance1(flows[i], flows[j])
            angle = calc_angel(flows[i], flows[j])
            # if disO < k_distance and angle < eps and disD < k_distance:
            if distance < R and angle < eps:
                flows[i].neighs.append(j)
                flows[j].neighs.append(i)
                wij = 1
                S[i, j] = wij
                S[j, i] = S[i, j]
    for l in flows:
        density = len(l.neighs)
        l.density = density
        den_list.append(density)
    return den_list


def calculate_density_variance(flows, R, theta):
    # 这里假设有一个函数 calculate_neighborhood_density，它接受流列表和 R 值，
    # 然后返回一个密度列表
    densities = calculate_neighborhood_density(flows, R, theta)
    return np.var(densities)


def calculate_Rtheta_constant(flows, R_values, theta_values, delta_R, delta_theta):
    variance_ratios = []
    for R in R_values:
        for theta in theta_values:
            var_R = calculate_density_variance(flows, R, theta)
            var_R_plus_delta = calculate_density_variance(flows, R + delta_R, theta + delta_theta)
            variance_ratios.append(var_R / var_R_plus_delta)
    # 计算方差比率的平均值或中位数
    Rtheta_constant = np.mean(variance_ratios)  # 或 np.median(variance_ratios)
    return Rtheta_constant


def calculate_RRD(flows, R, delta_R, theta, delta_theta, Rk_constant):
    var_R = calculate_density_variance(flows, R, theta)
    var_R_plus_delta = calculate_density_variance(flows, R + delta_R, theta + delta_theta )
    return var_R / var_R_plus_delta * Rk_constant


def main():
    # 读取流数据
    print('reading flow data file...')
    final_dir = 'Data/'
    flows = read_flow(final_dir + 'SD7.csv')
    print('done')
    # 初始化 R 值范围和 delta_R
    R_values = np.arange(150, 500, 10)  # 例如，从 100 到 1000，步长为 100
    delta_R = 10  # R 值的增量
    theta_values = np.arange(0, 45, 5)
    delta_theta = 5  # theta 值的增量
    print('calculating Rtheta constant...')
    Rk_constant = calculate_Rtheta_constant(flows, R_values, theta_values, delta_R, delta_theta)
    print('done')
    # 为每个 R 和 theta 组合计算 RRD
    RRD_values = []
    print('calculating RDV wrt R and Theta values...')
    for R in R_values:
        for theta in theta_values:
            RRD = calculate_RRD(flows, R, theta, delta_R, delta_theta, Rk_constant)
            RRD_values.append([R, theta, RRD])

    # 将数据转换为 NumPy 数组以便绘图
    RRD_array = np.array(RRD_values)
    print('done')
    # 将 NumPy 数组转换为 pandas DataFrame
    # 假设列名分别为 'R', 'Theta', 'RRD'
    df = pd.DataFrame(RRD_array, columns=['R', 'Theta', 'RRD'])
    # 将 DataFrame 保存为 CSV 文件
    csv_file = 'RDV_data.csv'
    df.to_csv(csv_file, index=False)
    print(f'saved the computed RDVs to {csv_file}')


if __name__ == '__main__':
    main()



