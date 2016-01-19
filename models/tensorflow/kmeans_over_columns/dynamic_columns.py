'''
Created on Jan 13, 2016

@author: jlovitt
'''
import sys
from dataio import LMDBDataProvider
from matplotlib import pyplot as plt
from sigmoid_autoencoder import *

class Object:
    pass

N_COLUMNS = 3
N_STEPS = 2
# LAYERS = [Layer()]
DATA_PARAM = Object()
DATA_PARAM.batch_size = 16
TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [0,0,0]
TRANSFORM_PARAM.crop_size = 225
TRANSFORM_PARAM.mirror = False

def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b



if __name__ == '__main__':
    DATA_PARAM.source = sys.argv[1]
    dp = LMDBDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    imgkeys = dp.get_keys()
    columns = {}
    for i in range(N_COLUMNS):
      columns[i] = AutoEncoder(dp)
    print columns
#   dat = dp.get_mb()
#   for d in dat:
#       i = d[0][0,:]
#       print i.shape
#       plt.imshow(i.reshape((225,225*3)),shape=(225,225*3))
#       plt.show()
# for i in range(N_COLUMNS):
#     columns[i] = make_column()
#     immap = map_img_2_col(imgkeys, columns)
#   for l in range(2):
#     immap_old = None
#     for column in columns.values():
#       column.add_layer(LAYERS[l])
#     while(not converged(immap, immap_old)):
#       encode(immap, N_STEPS)
#       immap_old = immap
#       immap = map_img_2_col(imgkeys, columns)