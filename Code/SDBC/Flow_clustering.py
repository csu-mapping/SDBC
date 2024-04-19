from Method import *
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from matplotlib.pyplot import MultipleLocator
global polyOD


"""



This python file is used to clustering.



"""


def run():
    
    path = '/Users/schnappi/PycharmProjects/pythonProject/zyx-clustering/SDBC/Data'
    # parameter setting
    eps = 30
    k_distance = 240

    sf = path + '/sim_7o.txt'
    nf = path + '/sim_7d.txt'

    cf = path + 'Clusters_{}.txt'.format(k_distance)
    df = path + 'represent_id.txt'.format(k_distance)


    flows = read_flow(sf, nf)

    flows, S = flow_neigh(flows, k_distance, eps)
    flows = compute_G(flows, S)
    clusters = identify_cluster(flows)

    λ = -1.25
    represent_flows = find_hotpot(flows, λ)

    if len(represent_flows) < 1:
       print('Not any clusters detected.')

    represent_id = list()

    # Draw a representative flow diagram
    for i in range(len(represent_flows)):
        represent_flow = represent_flows[i][1]
        id = represent_flows[i][0]
        represent_id.append(id)
        x1 = represent_flow.OX
        x2 = represent_flow.DX
        y1 = represent_flow.OY
        y2 = represent_flow.DY
        plt.plot([x1, x2], [y1, y2], color='#1f77b4')
        plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color='#1f77b4'))
    plt.show()

    f = open(df, 'w')
    f.writelines(str(represent_id))
    f.close()

    final_cluster = list()
    n_cluster = len(clusters)
    n = len(flows)
    while n_cluster > 1:
        I, J, max_Gi = search_max_GR(clusters, represent_flows, n)
        print(I, J, max_Gi)
        if (I < 1) or (J < 1):
            break
        clusters = update_cluster(I, J, max_Gi, clusters, flows)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    fig.suptitle('轨迹聚类')

    final_cluster = cluster_compute(clusters, flows)
    final_cluster = merge_final_cluster(final_cluster)

    result_cluster = list()
    result_clusters = list()

    i = 0
    for cluster in final_cluster:
        color = random_color()
        if len(cluster.members) > 0 and len(cluster.neighs) > 0:
            cluster.members.extend(cluster.neighs[0])
            cluster.members = list(set(cluster.members))
            if len(cluster.members) > 5:
              print(len(cluster.members), cluster.members)
              result_cluster.append(cluster)
              result_clusters.append(cluster.members)
              i += 1
              index = -1
              for member in cluster.members:
                index += 1
                x1 = flows[member].OX
                x2 = flows[member].DX
                y1 = flows[member].OY
                y2 = flows[member].DY
                if index == len(cluster.members) - 1:
                    plt.plot([x1, x2], [y1, y2], color=color, label='簇' + str(len((cluster.members))), alpha=0.5)
                    plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=color))
                else:
                    plt.plot([x1, x2], [y1, y2], color=color, alpha=0.5)
                    plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=color))
    plt.legend(loc=3)
    plt.show()
    print(result_clusters)

# Save results
    file = open(cf, 'w')
    for cluster in result_cluster:
        for j in cluster.members:
            file.writelines('{0},'.format(j))
        file.writelines('\n')
    file.close()

# Color
def random_color():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def plot_value(accuracy, recall, f_value):
    X = [i + 1 for i in range(len(accuracy))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, accuracy, c='r', marker='s', label='Accuracy')
    ax.plot(X, recall, c='g', marker='v', label='Recall')
    ax.plot(X, f_value, c='b', marker='d', label='F1')
    plt.xlabel('k_value', size=15)
    plt.tick_params(labelsize=10)
    plt.legend(bbox_to_anchor=(0.95, 0.6), loc='upper right', ncol=1, borderaxespad=0, fontsize='large')
    plt.show()

if __name__ == '__main__':
    run()

