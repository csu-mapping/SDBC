import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

def flowCSRtests(flows, IDX, mc_times=999, R=0):
    # Significant test
    IX = np.unique(IDX[IDX > 0])

    cluster_num = len(IX)
    if cluster_num > 0:
        clust = [{'Label': None, 'Members': None, 'Density': None, 'OBox': None, 'DBox': None, 'Pvalue': None} for _ in range(cluster_num)]

        for i in range(cluster_num):
            clust[i]['Label'] = IX[i]
            clust[i]['Members'] = np.where(IDX == clust[i]['Label'])[0]
            clust[i]['Density'] = len(clust[i]['Members'])

            x = flows[clust[i]['Members'], 0:2]
            hull = ConvexHull(x)
            x = x[hull.vertices]
            polygon = Polygon(x).buffer(R)
            clust[i]['OBox'] = np.array(polygon.exterior.coords)

            x = flows[clust[i]['Members'], 2:4]
            hull = ConvexHull(x)
            x = x[hull.vertices]
            polygon = Polygon(x).buffer(R)
            clust[i]['DBox'] = np.array(polygon.exterior.coords)

            clust[i]['Pvalue'] = 1.0

        # Run monte carlo permutation tests under CSR
        npts = flows.shape[0]
        mc_density = np.zeros((cluster_num, mc_times))
        for r in range(mc_times):
            rand_flows = np.hstack((flows[np.random.permutation(npts), 0:2], flows[np.random.permutation(npts), 2:4]))
            for j in range(cluster_num):
                isInOBox = np.array([Point(p).within(Polygon(clust[j]['OBox'])) for p in rand_flows[:, 0:2]])
                isInDBox = np.array([Point(p).within(Polygon(clust[j]['DBox'])) for p in rand_flows[:, 2:4]])
                mc_density[j, r] = np.sum(isInOBox & isInDBox)

        # P-value
        p_value = np.ones(npts)
        for i in range(cluster_num):
            clust[i]['Pvalue'] = np.sum(mc_density[i, :] >= clust[i]['Density']) / mc_times
            p_value[IDX == clust[i]['Label']] = clust[i]['Pvalue']

    else:
        p_value = np.array([])

    return p_value

