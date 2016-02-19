import bpy
import bmesh
import random
import math
import mathutils as mu
import generate_fries as gf
import save_and_purge as sp
import imp
imp.reload(gf)
imp.reload(sp)

N_IMAGES = 5
N_STEPS = 250
N_FRIES = 1024

if __name__ == "__main__":
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
    print("Fry Init complete")

    for gen in range(N_IMAGES):
        bpy.context.scene.frame_set(0)
        gf.generate(N_FRIES)
        for i in range(N_STEPS):
            bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
            print(i)
        sp.save()
        sp.remove_fries()

