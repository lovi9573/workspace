'''
Created on Mar 17, 2016

@author: jlovitt
'''

import numpy as np
import sys
from sklearn.cluster import DBSCAN,SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

if __name__ == '__main__':
    X = np.load(sys.argv[1])
    labels_true = np.load(sys.argv[2])
#     X = StandardScaler().fit_transform(X)
    X = X[0:20000,:]
    labels_true = labels_true[0:20000]
#     print(X)
    print("scaled to range {}->{}".format(np.min(X),np.max(X)))
#     db = DBSCAN(eps=0.1, min_samples = 10).fit(X)
    cl = SpectralClustering(n_clusters=10).fit(X)
    labels = cl.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))