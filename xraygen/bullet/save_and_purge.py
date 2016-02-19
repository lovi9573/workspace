import bpy
import bmesh
import random
import math
import mathutils as mu
import generate_fries as gf
#import bge
import time
import imp
imp.reload(gf)


def select_fries():
    bpy.ops.object.select_all(action='DESELECT')
    group = bpy.data.groups["Fries-Auto"]
    for obj in group.objects:
        obj.select= True
    return group.objects   

def get_max_keyframe(obj_list):
    max_keyframe = 0
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for curve in anim.action.fcurves:
                for keyframe in curve.keyframe_points:
                    x, _ = keyframe.co
                    if x > max_keyframe:
                        max_keyframe=math.ceil(x)
    return max_keyframe


def save():
    #bpy.ops.object.select_all(action='SELECT')
    select_fries()
    bpy.ops.export_mesh.stl(\
        filepath="/home/jlovitt/git/workspace/xraygen/xray/stl/generated{:0>5}.stl".format(gf.gen_num()),\
        check_existing=False, \
        axis_forward='-Z', \
        axis_up='-Y', \
        filter_glob="*.stl", \
        global_scale=1.0, \
        use_scene_unit=False, \
        ascii=False, \
        use_mesh_modifiers=True\
    )

def remove_fries():
    select_fries()
    bpy.ops.object.delete(use_global=False)
 
