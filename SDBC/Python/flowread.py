# coding=utf-8
# copyright(c) 2021 Central South University. all rights reserved.
# jianbo.tang@csu.edu.cn
# 2024/10/15

import numpy as np
import pandas as pd
import os


def flowread(csvfile):
    flows = pd.read_csv(csvfile).values
    m = flows.shape[1]

    if m == 6:
        # fid, ox, oy, dx, dy, label
        flow_times = np.array([])
        IX = flows[:, 5]
        flows = flows[:, 1:5]
    elif m == 8:
        # fid, ox, oy, ot, dx, dy, dt, label
        flow_times = flows[:, [3, 6]]
        IX = flows[:, 7]
        flows = flows[:, [1, 2, 4, 5]]
    else:
        raise ValueError('flow data format is not valid.')

    vmin = np.min([np.min(flows[:, :2]), np.min(flows[:, 2:4])])
    flows[:, [0, 2]] -= vmin
    flows[:, [1, 3]] -= vmin

    return flows, flow_times, IX


# main
if __name__ == '__main__':
    pass
