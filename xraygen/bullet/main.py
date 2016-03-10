import bpy
import generate_fries as gf
import save_and_purge as sp
#import config_pb2 as pbconf
#from google.protobuf import text_format
import imp
imp.reload(gf)
imp.reload(sp)



#  with open(PROTO,'r') as fin:
#    config = pbconf.Config()
#    text_format.Merge(str(fin.read()), config)
#  generatorconfig = config.generatorconfig()
#  physicsconfig = config.physicsconfig()  


def init(s):
  bpy.ops.object.select_all(action='DESELECT')
  if "Fries-Auto" not in bpy.data.groups:
    bpy.ops.group.create(name="Fries-Auto")
  if "Auto-Curves" not in bpy.data.groups:
    bpy.ops.group.create(name="Auto-Curves")
  
  def setint(self,v):
      self["gennum"] = v
  def getint(self):
      return self["gennum"] 
  bpy.types.Scene.gen_number = bpy.props.IntProperty(get=getint, set= setint)
  bpy.context.scene.gen_number=0

  sp.remove_fries()
  bpy.ops.object.select_all(action='DESELECT')
  bpy.data.scenes['Scene'].frame_start = 0
  bpy.data.scenes['Scene'].frame_end = 500
  bpy.data.scenes['Scene'].rigidbody_world.steps_per_second = s.PHYSICS_FREQ
  bpy.data.scenes['Scene'].rigidbody_world.solver_iterations = s.PHYSICS_SOLVER_ITER
  print("Fry Init complete")

def run(s):
  uid = 1
  for mean in range(12,17,2):
      for gen in range(s.N_IMAGES):
          bpy.context.scene.frame_set(0)
          gf.generate(s.VOLUME, s.WIDTH, s.LENGTH_MIN, s.LENGTH_MAX, mean, s.PCURVE)
          for i in range(s.N_STEPS):
              bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
              if i %10 ==0:
                  print("\tTime Step {}".format(i))
          sp.save(s.SAVE_PATH,s.FILE_PREFIX+"{:0>2}".format(mean)+"-",gen)
          sp.remove_fries()
          uid += 1
  print("Complete!")


  
