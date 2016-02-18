import bpy
import bmesh
import random
import math
import mathutils as mu
import generate_fries as gf
import imp
imp.reload(gf)

def save():
    bpy.ops.screen.frame_jump(end=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_mesh.stl(filepath="/home/jlovitt/git/workspace/xraygen/xray/stl/generated{:0>5}.stl".format(gf.n()), check_existing=False, axis_forward='Z', axis_up='Y', filter_glob="*.stl", global_scale=1.0, use_scene_unit=False, ascii=False, use_mesh_modifiers=True)

def remove_fries():
    bpy.ops.object.select_all(action='DESELECT')
    group = bpy.data.groups["Fries-Auto"]
    for obj in group.objects:
        obj.select= True
    bpy.ops.object.delete(use_global=False)
    
save()
remove_fries()
print("Fries saved as stl and removed.")
gf.generate()
if gf.n() < 5:
    bpy.ops.view3d.game_start()
