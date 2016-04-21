'''
Created on Mar 17, 2016

@author: jlovitt
'''

import numpy as np
import sys
from sklearn.cluster import DBSCAN,SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from dataio import MnistDataProvider

class Object:
    pass
  
DATA_PARAM = Object()
DATA_PARAM.batch_size = 6000


TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 28
TRANSFORM_PARAM.mirror = False  



if __name__ == '__main__':
  DATA_PARAM.source = sys.argv[1:]
  dp = MnistDataProvider(DATA_PARAM,TRANSFORM_PARAM )
  X,true_labels,keys = dp.get_mb().next()
  X = X.reshape([DATA_PARAM.batch_size,-1])
#     X = StandardScaler().fit_transform(X)
#     print(X)
#     print("scaled to range {}->{}".format(np.min(X),np.max(X)))
#     db = DBSCAN(eps=0.1, min_samples = 10).fit(X)
  cl = SpectralClustering(n_clusters=10).fit(X)
  labels = cl.labels_

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  
  print("Accuracy {}".format(metrics.accuracy_score(true_labels,labels)))
#   print('Estimated number of clusters: %d' % n_clusters_)
#   print('Estimated number of clusters: %d' % n_clusters_)
#   print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, labels))
#   print("Completeness: %0.3f" % metrics.completeness_score(true_labels, labels))
#   print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, labels))
#   print("Adjusted Rand Index: %0.3f"
#         % metrics.adjusted_rand_score(true_labels, labels))
#   print("Adjusted Mutual Information: %0.3f"
#         % metrics.adjusted_mutual_info_score(true_labels, labels))