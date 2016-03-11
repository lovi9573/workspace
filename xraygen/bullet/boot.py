import sys
import os
import bpy

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)
print(blend_dir)
print(os.getcwd())

import main 
import imp
imp.reload(main)
#import config_pb2 as pbconf
#from google.protobuf import text_format

class Settings():
  N_IMAGES = 20
  N_STEPS = 200
  #N_FRIES = 1534
  VOLUME = 8000
  PHYSICS_FREQ = 180
  PHYSICS_SOLVER_ITER = 5
  SAVE_PATH = "/home/jlovitt/storage/data/xrays/stl/"
  FILE_PREFIX = 'generated'
  WIDTH = 0.635
  LENGTH_MIN = 2.5
  LENGTH_MAX = 25
  LENGTH_MEAN = 9.0
  PCURVE = 0.00
  PROTO = '/home/jlovitt/git/workspace/xraygen/config.prototxt'

if __name__ == "__main__":
  s = Settings()
  main.init(s)
  main.run(s)