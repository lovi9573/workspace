import bpy
import bmesh
import random
import math
import mathutils as mu
import generate_fries as gf
import imp
imp.reload(gf)


if __name__ == "__main__":
    bpy.ops.object.select_all(action='DESELECT')
    if bpy.ops.rigidbody.world_add.poll():
        bpy.ops.rigidbody.world_add()
    bpy.ops.group.create(name="Fries-Auto")
    
    def setint(self,v):
        self["gennum"] = v
    def getint(self):
        return self["gennum"]
    
    bpy.types.Scene.gen_number = bpy.props.IntProperty(get=getint, set= setint)
    bpy.context.scene.gen_number=1

    # test call
    bpy.context.scene.render.engine = 'BLENDER_GAME'
    bpy.context.scene.game_settings.use_animation_record = True
    bpy.ops.object.select_all(action='DESELECT')
    gf.generate()
    bpy.ops.view3d.game_start()
    print("Fry Init complete")