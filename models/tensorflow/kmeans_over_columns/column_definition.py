'''
Created on Apr 14, 2016

@author: jlovitt
'''

from autoencoder import *
from dataio import LMDBDataProvider, CifarDataProvider, MnistDataProvider

class Object:
    pass

DATA_PARAM = Object()
TRANSFORM_PARAM = Object()

"""
Global Settings -----------------------------------------------------
"""
#Data provider to use for training
DATA_PROVIDER=CifarDataProvider
# DATA_PROVIDER=MnistDataProvider

# Number of epochs to wait for improved loss during pretraining
DEFAULT_PATIENCE=5

# Improvement threshold to use as "improvement" during pretraining
DEFAULT_PATIENCE_DELTA=0.0003

DATA_PARAM.batch_size = 64

TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 31
TRANSFORM_PARAM.mirror = False  

NUM_LABELS = 10
RECON_SHAPE = [DATA_PARAM.batch_size,TRANSFORM_PARAM.crop_size,TRANSFORM_PARAM.crop_size,3 ]

"""
All Examples Pretraining -----------------------------------------------------
"""


"""
Labeled example pretraining -----------------------------------------------------
"""
N_LABELED_EXAMPLES = 8

"""
Clustering -----------------------------------------------------
"""
# number of columns to route data to dynamically
N_COLUMNS = 10

# Number of batches to train between data reroute's
TRAIN_BATCHES = 20

# Growth rate for TRAIN_BATCHES every training cycle
D_TRAIN_BATCHES = 10




  
  



# """
# Mnist setup
# """
# LAYERS = [
# #           {"Layerdef":CorruptionLayerDef(0.0),
# #            "Train":False},
# #           {'Layerdef':FCLayerDef(1024),
# #            "Pretrain_all":-1,
# #            "Convergence_threshold":0.0},
# #1
#           {"Layerdef":ConvLayerDef(5,2,4,padding='SAME',tied_weights=False ),
#               "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 14
# #2
#           {"Layerdef":ConvLayerDef(5,2,8,padding='SAME',tied_weights=False),
#                "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 7
# #3
#           {"Layerdef":ConvLayerDef(3,1,8,tied_weights=False),
#                "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out:5
#  #4
#            {"Layerdef":ConvLayerDef(3,1,6,tied_weights=False),
#                 "Pretrain_all":-1,
#                 "Patience": DEFAULT_PATIENCE,
#                 "Patience_delta": DEFAULT_PATIENCE_DELTA,
#                 'Use_To_Map_Samples':True,
#                 "Convergence_threshold":0.9  
#            }, #out: 3
# #5
#           {"Layerdef":ConvLayerDef(3,1,4,tied_weights=False),
#                "Pretrain_all":0,
#                "Patience": 1,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#                'Use_To_Map_Samples':True,
#                "Convergence_threshold":0.9  
#           }, #out: 1
# #6
#           {'Layerdef':FCLayerDef(2,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
#            "Pretrain_all":0,
#            "Patience": 5,
#            "Patience_delta": DEFAULT_PATIENCE_DELTA,
#            'Use_To_Map_Samples':True,
#            "Convergence_threshold":0.97
#            },
# #           {"Layerdef":ConvLayerDef(3,1,32,sparsity_target=0.03, sparsity_lr=0.1),
# #                "Pretrain_all":-1,
# #                "Patience": 5,
# #                "Patience_delta": 0.001,
# #                "Convergence_threshold":0.0},
# #           {"Layerdef":ConvLayerDef(3,1,32),
# #             "Pretrain_all":-1,
# #             "Convergence_threshold":0.0},
# #           {"Layerdef":ConvLayerDef(3,1,1),
# #            "Pretrain_all":10,
# #            "Convergence_threshold":0.99}
#           ]


"""
Cifar setup
"""
LAYERS = [
#1
          {"Layerdef":ConvLayerDef(7,2,32,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False ),  #Could go with 4
           "Decodedef":[ConvLayerDef(7,2,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":True},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 13
# 2
#            {"Layerdef":ConvLayerDef(5,2,16,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
#             "Decodedef":[ConvLayerDef(15,4,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
#             "All":{"N_epochs":-1,
#                    "Patience":DEFAULT_PATIENCE,
#                    "Patience_delta":DEFAULT_PATIENCE_DELTA,
#                    "Freeze":True},
#             "Labeled":{"N_epochs":0,
#                    "Patience":DEFAULT_PATIENCE,
#                    "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
#                    "Freeze":True},
#             "Mapped":{"N_epochs":0,
#                    "Patience":DEFAULT_PATIENCE,
#                    "Patience_delta":DEFAULT_PATIENCE_DELTA,
#                    "Freeze":False},
#             'Use_To_Map_Samples':True,
#             "Convergence_threshold":0.9
#            }, #out: 5
#3
#           {"Layerdef":ConvLayerDef(3,1,384,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
#            "Decodedef":[ConvLayerDef(23,4,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )], 
#            "All":{"N_epochs":-1,
#                   "Patience":DEFAULT_PATIENCE,
#                   "Patience_delta":DEFAULT_PATIENCE_DELTA,
#                   "Freeze":False},
#            "Labeled":{"N_epochs":-1,
#                   "Patience":DEFAULT_PATIENCE,
#                   "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
#                   "Freeze":False},
#            "Mapped":{"N_epochs":0,
#                   "Patience":DEFAULT_PATIENCE,
#                   "Patience_delta":DEFAULT_PATIENCE_DELTA,
#                   "Freeze":False},
#           }, #out:3
#4 
          {'Layerdef':FCLayerDef(1024,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
           "Decodedef":[FCLayerDef([31,31,3],sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           'Use_To_Map_Samples':True,
           "Convergence_threshold":0.9
            }
          ]


def get_dp(data_param, transform_param):
  return DATA_PROVIDER(data_param, transform_param)

