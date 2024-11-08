import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from fontTools.unicodedata import block
from sklearn.metrics import adjusted_rand_score
from flowread import*
from flowSDBC import*
from flowCSRtests import*


# Loading flow data
flows, flow_times, groundTruthIDX = flowread('../Data/SD6.csv')

# Parameter setting
spatial_R = 300  # spatial distance threshold
angle_T = 30  # direction consistency threshold (in degree)
mc_RepeatTimes = 999  # Monte Carlo simulation tests

# Clustering flows using flowSDBC
IDX, seeds, flow_density, spatialneighbors =  flowSDBC(flows, flow_times, spatial_R, angle_T)

# Significant test under CSR hypothesis
print('statistical significance inference of candidate clusters based on MC...')
p_value = flowCSRtests(flows, IDX, mc_RepeatTimes, spatial_R)
IDX[p_value > 0.05] = 0
print('done')

# ARI
ARI = adjusted_rand_score(IDX, groundTruthIDX)
print(f'adjusted_rand_score = {ARI}')
filename = 'flow_label.txt'
save_result(filename, IDX)
print('clustering result is saved to: ' + filename)

# Show clustering results
# Assuming flowplot is a custom function, you need to implement it in Python
# For now, let's comment it out
flowplot(flows, IDX)
plt.title('Final clustering result', fontsize=13, fontweight='bold')
plt.show(block=True)

