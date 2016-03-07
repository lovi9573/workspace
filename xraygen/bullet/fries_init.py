import bpy
import bmesh
import random
import math
import mathutils as mu
import generate_fries as gf
import save_and_purge as sp
#import config_pb2 as pbconf
#from google.protobuf import text_format
import imp
#imp.reload(gf)
#imp.reload(sp)

N_IMAGES = 1
N_STEPS = 250
#N_FRIES = 1534
VOLUME = 5000
PHYSICS_FREQ = 100
PHYSICS_SOLVER_ITER = 10
SAVE_PATH = "/home/jlovitt/git/workspace/xraygen/xray/stl/"
FILE_PREFIX = 'generated'
WIDTH = 0.635
LENGTH_MIN = 1.0
LENGTH_MAX = 25
LENGTH_MEAN = 9.0
PCURVE = 0.00
PROTO = '/home/jlovitt/git/workspace/xraygen/config.prototxt'

if __name__ == "__main__":
#  with open(PROTO,'r') as fin:
#    config = pbconf.Config()
#    text_format.Merge(str(fin.read()), config)
#  generatorconfig = config.generatorconfig()
#  physicsconfig = config.physicsconfig()  
  
  bpy.ops.object.select_all(action='DESELECT')
  bpy.ops.group.create(name="Fries-Auto")
  
  def setint(self,v):
      self["gennum"] = v
  def getint(self):
      return self["gennum"] 
  bpy.types.Scene.gen_number = bpy.props.IntProperty(get=getint, set= setint)
  bpy.context.scene.gen_number=0

  sp.remove_fries()
  bpy.ops.object.select_all(action='DESELECT')
  bpy.data.scenes['Scene'].rigidbody_world.steps_per_second = PHYSICS_FREQ
  bpy.data.scenes['Scene'].rigidbody_world.solver_iterations = PHYSICS_SOLVER_ITER
  print("Fry Init complete")
  for mean in range(2,14,2):
      for gen in range(N_IMAGES):
          bpy.context.scene.frame_set(0)
          gf.generate(VOLUME, WIDTH, LENGTH_MIN, LENGTH_MAX, mean, PCURVE)
          for i in range(N_STEPS):
              bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
              if i %10 ==0:
                  print("\tTime Step {}".format(i))
          sp.save(SAVE_PATH,FILE_PREFIX+str(mean))
          sp.remove_fries()
  print("Complete!")

