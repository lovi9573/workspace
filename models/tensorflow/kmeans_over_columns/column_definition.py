'''
Created on Apr 14, 2016

@author: jlovitt
'''

"""
======================== Do not modify ================
"""
from autoencoder import *
from dataio import LMDBDataProvider, CifarDataProvider, MnistDataProvider

class Object:
    pass

DATA_PARAM = Object()
TRANSFORM_PARAM = Object()
"""
======================== Do not modify ================
"""

"""
=========================== Global Settings ==============================================
"""
#Data provider to use for training
DATA_PROVIDER=CifarDataProvider
# DATA_PROVIDER=MnistDataProvider
NUM_LABELS = 10

# Number of epochs to wait for improved loss during pretraining
DEFAULT_PATIENCE=15

# Improvement threshold to use as "improvement" during pretraining
DEFAULT_PATIENCE_DELTA=0.0001



DATA_PARAM.batch_size = 64

TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 31
TRANSFORM_PARAM.mirror = False 


"""
=======================  Layer Definitions ===========================
"""
  

# """
# Mnist setup
# """
LAYERS_mnist = [
#1
          {"Layerdef":CorruptionLayerDef(0.1),
           "Train":False},
#2
          {"Layerdef":ConvLayerDef(5,2,2,padding='VALID' ,sparsity_target=0.01, sparsity_lr=0,tied_weights=False),  
           "Decodedef":[ConvLayerDef(5,2,1,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False)],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 12
#3
          {"Layerdef":ConvLayerDef(4,2,8,padding='VALID' ,sparsity_target=0.01, sparsity_lr=0,tied_weights=False),  
           "Decodedef":[ConvLayerDef(11,4,1,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False)],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 5
#4
          {"Layerdef":ConvLayerDef(3,1,16,padding='VALID' ,sparsity_target=0.01, sparsity_lr=0,tied_weights=False),  
           "Decodedef":[ConvLayerDef(13,7,1,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False)],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 3
#5
          {"Layerdef":ConvLayerDef(3,1,2,padding='VALID' ,sparsity_target=0.01, sparsity_lr=0,tied_weights=False),  
           "Decodedef":[ConvLayerDef(27,27,1,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False)],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          } #out: 1
          ]


"""
Cifar setup
"""
LAYERS_cifar = [
#1
          {"Layerdef":CorruptionLayerDef(0.01),
           "Train":False},
#2
          {"Layerdef":ConvLayerDef(5,2,8,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False ),  
           "Decodedef":[ConvLayerDef(7,2,3,padding='VALID', sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
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
#3
            {"Layerdef":ConvLayerDef(5,2,32,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
             "Decodedef":[ConvLayerDef(15,4,3,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
             "All":{"N_epochs":-1,
                    "Patience":DEFAULT_PATIENCE,
                    "Patience_delta":DEFAULT_PATIENCE_DELTA,
                    "Freeze":True},
             "Labeled":{"N_epochs":0,
                    "Patience":DEFAULT_PATIENCE,
                    "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                    "Freeze":True},
             "Mapped":{"N_epochs":0,
                    "Patience":DEFAULT_PATIENCE,
                    "Patience_delta":DEFAULT_PATIENCE_DELTA,
                    "Freeze":False},
             'Use_To_Map_Samples':True,
             "Convergence_threshold":0.9
            }, #out: 5
#4
          {"Layerdef":ConvLayerDef(3,1,128,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
           "Decodedef":[ConvLayerDef(23,4,3,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False )], 
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA/10,
                  "Freeze":False},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out:3
#5 
          {'Layerdef':FCLayerDef(64,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
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

 

LAYERS = LAYERS_cifar

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



"""
======================== Do not modify ================
"""
def get_dp(data_param, transform_param):
  return DATA_PROVIDER(data_param, transform_param)

