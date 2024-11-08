import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def Drawing_3D(df):
    R, Theta = np.meshgrid(df['R'].unique(), df['Theta'].unique())
    RRD_grid = df.pivot_table(index='Theta', columns='R', values='RRD').values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(R, Theta, RRD_grid, cmap='viridis',alpha=0.8) # alpha设置透明度)
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # 显示图形
    plt.show()


def line_3d(RRD_array):
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_weight('bold')
    ax.plot(RRD_array[:, 0], RRD_array[:, 1], RRD_array[:, 2], c='r', marker='o')
    ax.set_xlabel('R Value', fontproperties=font)
    ax.set_ylabel('Theta Value', fontproperties=font)
    ax.set_zlabel('RDV', fontproperties=font)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    if hasattr(ax, 'get_zticklabels'):
        for label in ax.get_zticklabels():
            label.set_fontproperties(font)
    plt.show()


def plot_gradient_fields1(grad_R, grad_theta, R_grid, theta_grid):
    # 创建梯度场图
    plt.figure(figsize=(14, 6))
    # 设置 Times New Roman 字体
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_weight('bold')  # 设置字体为粗体
    # 绘制 R 梯度场
    plt.subplot(1, 2, 1)
    norm = Normalize(vmin=grad_R.min(), vmax=grad_R.max())  # 归一化梯度值
    plt.quiver(R_grid, theta_grid, grad_R, np.zeros_like(grad_R),
               norm(norm(grad_R)),  scale_units='xy', scale=1)
    plt.colorbar()
    plt.title('Gradient Field of RDV wrt R',fontproperties=font)
    plt.xlabel('R Value',fontproperties=font)
    plt.ylabel('Theta Value',fontproperties=font)
    # 绘制 θ 梯度场
    plt.subplot(1, 2, 2)
    norm = Normalize(vmin=grad_theta.min(), vmax=grad_theta.max())  # 归一化梯度值
    plt.quiver(R_grid, theta_grid, np.zeros_like(grad_theta), grad_theta,
               norm(norm(grad_theta)), scale_units='xy', scale=1)
    plt.colorbar()
    plt.title('Gradient Field of RDV wrt Theta',fontproperties=font)
    plt.xlabel('R Value',fontproperties=font)
    plt.ylabel('Theta Value',fontproperties=font)
    plt.tight_layout()
    plt.show()


def plot_gradient_fields(grad_R, grad_theta, R_grid, theta_grid):
    # 创建图形并设置大小
    plt.figure(figsize=(14, 6))
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_weight('bold')
    # 绘制 R 方向的梯度
    plt.subplot(1, 2, 1)
    levels_R = MaxNLocator(nbins=50).tick_values(grad_R.min(), grad_R.max())
    contour_R = plt.contourf(R_grid, theta_grid, grad_R, levels=levels_R, cmap='viridis', extend='both')
    plt.colorbar(contour_R, label='Gradient magnitude')
    plt.title('Gradient of RDV wrt R', fontproperties=font)
    plt.xlabel('R Value', fontproperties=font)
    plt.ylabel('Theta Value', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    # 绘制 θ 方向的梯度
    plt.subplot(1, 2, 2)
    levels_theta = MaxNLocator(nbins=50).tick_values(grad_theta.min(), grad_theta.max())
    contour_theta = plt.contourf( R_grid,theta_grid, grad_theta, levels=levels_theta, cmap='viridis', extend='both')
    plt.colorbar(contour_theta, label='Gradient magnitude')
    plt.title('Gradient of RDV wrt Theta', fontproperties=font)
    plt.xlabel('R Value', fontproperties=font)
    plt.ylabel('Theta Value', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('RDV_data.csv', encoding='utf-8')
    # Drawing_3D(df)
    RRD_array = df.values
    # line_3d(RRD_array)
    R_values = np.arange(150, 500, 10)  # 例如，从 100 到 1000，步长为 100
    delta_R = 10  # R 值的增量
    theta_values = np.arange(0, 45, 5)
    delta_theta = 5  # theta 值的增量

    # 计算 RRD 相对于 R 和 θ 的梯度
    # 创建 R 和 theta 的网格
    R_grid, theta_grid = np.meshgrid(R_values, theta_values)
    RRD_grid = np.full((len(theta_values), len(R_values)), np.nan)
    for r, theta, rrd in RRD_array:
        r_index = np.where(R_values == r)[0][0]
        theta_index = np.where(theta_values == theta)[0][0]
        RRD_grid[theta_index, r_index] = rrd

    print("RDV_grid contains NaN values:", np.isnan(RRD_grid).any())

    # 计算梯度
    # 使用 R 和 θ 的步长作为梯度计算的间隔
    grad_R, grad_theta = np.gradient(RRD_grid, 10, 5)  # 假设 R 的步长是 10，θ 的步长是 5
    plot_gradient_fields(grad_R, grad_theta, R_grid, theta_grid)
    # 创建梯度场图
    plt.figure(figsize=(8, 6))
    Q = plt.quiver(R_values, theta_values, grad_R, grad_theta, scale=1e-4)
    plt.colorbar(Q, label='Gradient magnitude')
    plt.title('Gradient Field of RDV')
    plt.xlabel('R Value')
    plt.ylabel('θ Value')
    plt.show()

    selected_R = 230  # 示例中选取的 R 值
    selected_RRD = RRD_array[RRD_array[:, 0] == selected_R]
    # 设置 Times New Roman 字体
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_weight('bold')  # 设置字体为粗体

    # 绘制 RRD 与 θ 的关系
    plt.figure()
    plt.plot(selected_RRD[:, 1], selected_RRD[:, 2])
    plt.xlabel('θ Value', fontproperties=font)
    plt.ylabel('RDV Value', fontproperties=font)
    # 应用 Times New Roman 字体到刻度标签
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.show()

    selected_theta = 40
    # selected_theta = 30
    selected_RRD = RRD_array[RRD_array[:, 1] == selected_theta]

    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_weight('bold')

    # 绘制 RRD 与 R 的关系
    plt.figure()
    plt.plot(selected_RRD[:, 0], selected_RRD[:, 2])
    plt.xlabel('R Value', fontproperties=font)
    plt.ylabel('RDV Value', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.title(f'RDV vs R for Theta = {selected_theta}', fontproperties=font)
    plt.show()


if __name__ == '__main__':
    main()
