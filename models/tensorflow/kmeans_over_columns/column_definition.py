'''
Created on Apr 14, 2016

@author: jlovitt
'''

from autoencoder import *



# Number of epochs to wait for improved loss during pretraining
DEFAULT_PATIENCE=15

# Improvement threshold to use as "improvement" during pretraining
DEFAULT_PATIENCE_DELTA=0.0001

"""
Cifar Setup
LAYERS = [
          {"Layerdef":CorruptionLayerDef(0.9),
           "Train":False},
          {"Layerdef":ConvLayerDef(2,1,8),
           "Pretrain_epochs":-1,
           "Convergence_threshold":0.0},
          {"Layerdef":ConvLayerDef(3,2,16),
           "Pretrain_epochs":-1,
           "Convergence_threshold":0.0},
          {"Layerdef":ConvLayerDef(3,2,16),
           "Pretrain_epochs":-1,
           "Convergence_threshold":0.0},
          {"Layerdef":ConvLayerDef(3,2,32),
           "Pretrain_epochs":-1,
           "Convergence_threshold":0.0},
          {"Layerdef":ConvLayerDef(3,1,32),
            "Pretrain_epochs":-1,
            "Convergence_threshold":0.0},
#           {"Layerdef":ConvLayerDef(3,1,1),
#            "Pretrain_epochs":10,
#            "Convergence_threshold":0.99}
          ]
"""
"""
LAYERS = [
          {"Layerdef":CorruptionLayerDef(0.15),
           "Train":False},
          {'Layerdef':FCLayerDef(128,lr=0.7),
           "Pretrain_epochs":-1,
           "Patience": 10,
           "Patience_delta": 0.0001,
           "Convergence_threshold":0.0},
          {'Layerdef':FCLayerDef(64,lr=0.9),
           "Pretrain_epochs":-1,
           "Patience": 10,
           "Patience_delta": 0.0001,
           "Convergence_threshold":0.0},
          {'Layerdef':FCLayerDef(32),
           "Pretrain_epochs":-1,
           "Patience": 10,
           "Patience_delta": 0.0001,
           "Convergence_threshold":0.0},
          {'Layerdef':FCLayerDef(16),
           "Pretrain_epochs":-1,
           "Patience": 10,
           "Patience_delta": 0.0001,
           "Convergence_threshold":0.0},
          {'Layerdef':FCLayerDef(10),
           "Pretrain_epochs":-1,
           "Patience": 10,
           "Patience_delta": 0.0001,
           "Convergence_threshold":0.0},
         ]
"""
# """
# Mnist setup
# """
# LAYERS = [
# #           {"Layerdef":CorruptionLayerDef(0.0),
# #            "Train":False},
# #           {'Layerdef':FCLayerDef(1024),
# #            "Pretrain_epochs":-1,
# #            "Convergence_threshold":0.0},
# #1
#           {"Layerdef":ConvLayerDef(5,2,4,padding='SAME',tied_weights=False ),
#               "Pretrain_epochs":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 14
# #2
#           {"Layerdef":ConvLayerDef(5,2,8,padding='SAME',tied_weights=False),
#                "Pretrain_epochs":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 7
# #3
#           {"Layerdef":ConvLayerDef(3,1,8,tied_weights=False),
#                "Pretrain_epochs":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out:5
#  #4
#            {"Layerdef":ConvLayerDef(3,1,6,tied_weights=False),
#                 "Pretrain_epochs":-1,
#                 "Patience": DEFAULT_PATIENCE,
#                 "Patience_delta": DEFAULT_PATIENCE_DELTA,
#                 'Use_To_Map_Samples':True,
#                 "Convergence_threshold":0.9  
#            }, #out: 3
# #5
#           {"Layerdef":ConvLayerDef(3,1,4,tied_weights=False),
#                "Pretrain_epochs":0,
#                "Patience": 1,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#                'Use_To_Map_Samples':True,
#                "Convergence_threshold":0.9  
#           }, #out: 1
# #6
#           {'Layerdef':FCLayerDef(2,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
#            "Pretrain_epochs":0,
#            "Patience": 5,
#            "Patience_delta": DEFAULT_PATIENCE_DELTA,
#            'Use_To_Map_Samples':True,
#            "Convergence_threshold":0.97
#            },
# #           {"Layerdef":ConvLayerDef(3,1,32,sparsity_target=0.03, sparsity_lr=0.1),
# #                "Pretrain_epochs":-1,
# #                "Patience": 5,
# #                "Patience_delta": 0.001,
# #                "Convergence_threshold":0.0},
# #           {"Layerdef":ConvLayerDef(3,1,32),
# #             "Pretrain_epochs":-1,
# #             "Convergence_threshold":0.0},
# #           {"Layerdef":ConvLayerDef(3,1,1),
# #            "Pretrain_epochs":10,
# #            "Convergence_threshold":0.99}
#           ]


"""
Mnist setup
"""
LAYERS = [
#1
          {"Layerdef":CorruptionLayerDef(0.2),
           "Train":False},
#2
          {"Layerdef":ConvLayerDef(5,2,2,padding='SAME',tied_weights=False ),
              "Pretrain_epochs":-1,
              "Patience": DEFAULT_PATIENCE,
              "Patience_delta": DEFAULT_PATIENCE_DELTA,
          }, #out: 14
#3
          {"Layerdef":ConvLayerDef(5,2,4,padding='SAME',tied_weights=False),
               "Pretrain_epochs":-1,
               "Patience": DEFAULT_PATIENCE,
               "Patience_delta": DEFAULT_PATIENCE_DELTA,
          }, #out: 7
#4
          {"Layerdef":ConvLayerDef(3,1,6,tied_weights=False),
               "Pretrain_epochs":-1,
               "Patience": DEFAULT_PATIENCE,
               "Patience_delta": DEFAULT_PATIENCE_DELTA,
          }, #out:5
  #5
            {"Layerdef":ConvLayerDef(3,1,18,tied_weights=False),
                 "Pretrain_epochs":-1,
                 "Patience": DEFAULT_PATIENCE,
                 "Patience_delta": DEFAULT_PATIENCE_DELTA,
            }, #out: 3
#6
          {"Layerdef":FCLayerDef(64,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
               "Pretrain_epochs":-1,
               "Patience": DEFAULT_PATIENCE,
               "Patience_delta": DEFAULT_PATIENCE_DELTA*0.1,
          }, 
#7
          {"Layerdef":FCLayerDef(16,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
               "Pretrain_epochs":-1,
               "Patience": DEFAULT_PATIENCE,
               "Patience_delta": DEFAULT_PATIENCE_DELTA*0.1,
          }, 
#8
          {'Layerdef':FCLayerDef(4,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
           "Pretrain_epochs":-1,
           "Patience": DEFAULT_PATIENCE,
           "Patience_delta": DEFAULT_PATIENCE_DELTA*0.1,
           'Use_To_Map_Samples':True,
           "Convergence_threshold":0.95
           },
#           {"Layerdef":ConvLayerDef(3,1,32,sparsity_target=0.03, sparsity_lr=0.1),
#                "Pretrain_epochs":-1,
#                "Patience": 5,
#                "Patience_delta": 0.001,
#                "Convergence_threshold":0.0},
#           {"Layerdef":ConvLayerDef(3,1,32),
#             "Pretrain_epochs":-1,
#             "Convergence_threshold":0.0},
#           {"Layerdef":ConvLayerDef(3,1,1),
#            "Pretrain_epochs":10,
#            "Convergence_threshold":0.99}
          ]


class Object:
    pass
  
  
DATA_PARAM = Object()
DATA_PARAM.batch_size = 64


TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 28
TRANSFORM_PARAM.mirror = False  

NUM_LABELS = 10

