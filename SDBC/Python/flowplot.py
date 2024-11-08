# coding=utf-8
# copyright(c) 2021 Central South University. all rights reserved.
# jianbo.tang@csu.edu.cn
# 2024/10/15

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Arial'
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)


def flowplot(flows, IDX=None, linewidth=2):
    if IDX is None:
        IDX = np.ones(flows.shape[0], dtype=int)

    n = flows.shape[0]
    if len(IDX) < n:
        IX = np.zeros(n, dtype=int)
        IX[IDX] = 1
        IDX = IX

    IX = np.unique(IDX[IDX > 0])
    n = len(IX)
    cluster_colors = 'b'
    if n > 1:
        cluster_colors = np.random.rand(n, 3)
    noise_colors = [0.7, 0.7, 0.7]

    X, Y = flow_to_sequence(flows, IDX < 1)
    plt.plot(X, Y, '-', color=noise_colors, linewidth=linewidth)

    for i in range(len(IX)):
        X, Y = flow_to_sequence(flows, IDX == IX[i])
        plt.plot(X, Y, '-', color=cluster_colors[i] if n > 1 else cluster_colors, linewidth=linewidth)

    plt.xlabel('X', fontsize=14, fontweight='bold')
    plt.ylabel('Y', fontsize=14, fontweight='bold')


def hold(mode='on'):
    if mode == 'on':
        plt.ion()
    else:
        plt.ioff()


def draw_now(fig=None, sleep_sec=0.001):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(sleep_sec)


def flow_to_sequence(flows, flow_ids):
    X = list()
    Y = list()
    if np.any(flow_ids):
        n = len(flow_ids)
        if n == flows.shape[0]:
            n = np.sum(flow_ids)
        X = np.full(3 * n, np.nan)
        Y = np.full(3 * n, np.nan)
        X[::3] = flows[flow_ids, 0]
        X[1::3] = flows[flow_ids, 2]
        Y[::3] = flows[flow_ids, 1]
        Y[1::3] = flows[flow_ids, 3]
    return X, Y


# main
if __name__ == '__main__':
    pass
