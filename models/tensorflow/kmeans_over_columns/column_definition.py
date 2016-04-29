'''
Created on Apr 14, 2016

@author: jlovitt
'''

from autoencoder import *



# Number of epochs to wait for improved loss during pretraining
DEFAULT_PATIENCE=5

# Improvement threshold to use as "improvement" during pretraining
DEFAULT_PATIENCE_DELTA=0.001



class Object:
    pass
  
  
DATA_PARAM = Object()
DATA_PARAM.batch_size = 64


TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 31
TRANSFORM_PARAM.mirror = False  

NUM_LABELS = 10
RECON_SHAPE = [DATA_PARAM.batch_size,TRANSFORM_PARAM.crop_size,TRANSFORM_PARAM.crop_size,3 ]

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
"""
TODO:
- Add a setting for training epochs per phase: i.e. pretrain on all data, pretrain on labeled data, pretrain on mapped data, vs the current pretrain_epochs.
- Add a setting for freezing/thawing per phase
- Refactor to read encode/decode per layer.  The encode would need to be added to the stack, The decode would be a full decoder definition for that level.
"""
LAYERS = [
# #1
#           {"Layerdef":CorruptionLayerDef(0.1),
#            "Train":False},
#1
          {"Layerdef":ConvLayerDef(7,2,96,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False ),  #Could go with 4
           "Decodedef":[ConvLayerDef(7,2,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":True},
           "Labeled":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":True},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 13
# #2
#           {"Layerdef":ConvLayerDef(1,1,128,padding='VALID',sparsity_target=0.0, sparsity_lr=0.0,tied_weights=False,activation_function=tf.nn.relu), #could go 10
#                "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 25
# #3
#           {"Layerdef":ConvLayerDef(1,1,128,sparsity_target=0.0, sparsity_lr=0.0,tied_weights=False,activation_function=tf.nn.relu), 
#                "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out:25
#4
          {"Layerdef":ConvLayerDef(5,2,256,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
           "Decodedef":[ConvLayerDef(15,4,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )],
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":True},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Mapped":{"N_epochs":0,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out: 5
#5
#           {"Layerdef":ConvLayerDef(1,1,256,sparsity_target=0.0, sparsity_lr=0.0,padding='VALID',tied_weights=False,activation_function=tf.nn.relu ),
#               "Pretrain_all":-1,
#               "Patience": DEFAULT_PATIENCE,
#               "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 19
# #6
#           {"Layerdef":ConvLayerDef(1,1,256,padding='VALID',sparsity_target=0.0, sparsity_lr=0.0,tied_weights=False,activation_function=tf.nn.relu),
#                "Pretrain_all":-1,
#                "Patience": DEFAULT_PATIENCE,
#                "Patience_delta": DEFAULT_PATIENCE_DELTA,
#           }, #out: 19
#7
          {"Layerdef":ConvLayerDef(3,1,384,padding='VALID',sparsity_target=0.01, sparsity_lr=0,tied_weights=False),
           "Decodedef":[ConvLayerDef(23,4,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )], 
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Mapped":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
          }, #out:3
#TODO: implement this for FC layer
          {'Layerdef':FCLayerDef(1024,sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False),
           "Decodedef":[FCLayerDef(512,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False ),
                        ConvLayerDef(7,2,3,padding='VALID', recon_shape=RECON_SHAPE,sparsity_target=0.01, sparsity_lr=0,tied_weights=False )], 
           "All":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Labeled":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           "Mapped":{"N_epochs":-1,
                  "Patience":DEFAULT_PATIENCE,
                  "Patience_delta":DEFAULT_PATIENCE_DELTA,
                  "Freeze":False},
           'Use_To_Map_Samples':True,
           "Convergence_threshold":0.9
           }
#           {"Layerdef":ConvLayerDef(3,1,32,sparsity_target=0.03, sparsity_lr=0.1),
#                "Pretrain_all":-1,
#                "Patience": 5,
#                "Patience_delta": 0.001,
#                "Convergence_threshold":0.0},
#           {"Layerdef":ConvLayerDef(3,1,32),
#             "Pretrain_all":-1,
#             "Convergence_threshold":0.0},
#           {"Layerdef":ConvLayerDef(3,1,1),
#            "Pretrain_all":10,
#            "Convergence_threshold":0.99}
          ]



