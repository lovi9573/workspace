import bpy
import bmesh
import random
import math
import time
import mathutils as mu
from bpy_extras import object_utils
import curve_add as ca
import imp
imp.reload(ca)

def gen_num():
    return bpy.context.scene.gen_number

def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """
    
    segments = 16
    verts = []
    for d in range(segments+1):
        offset = 2.0*d/segments
        verts += [(+1.0, -1.0+offset, -1.0),
                 (-1.0, -1.0+offset, -1.0),
                 (-1.0, -1.0+offset, +1.0),
                 (+1.0, -1.0+offset, +1.0)
                 ]

    n_verts = len(verts)    
    faces = [(3,2,1,0)]
    for f in range(4*segments):
        if (f+1)%4 != 0:
            faces += [(f,f+1,f+5,f+4)]
        else:
            faces +=[(f, f-3 , f+1, f+4)] 

    faces += [(n_verts-4,n_verts-3,n_verts-2,n_verts-1)]
    # apply size
    for i, v in enumerate(verts):
        verts[i] = v[0] * width, v[1] * depth, v[2] * height
    return verts, faces


from bpy.props import FloatProperty, BoolProperty, FloatVectorProperty

class Timer():
        
    def __init__(self):
        self.base = time.time()
        self.times = {}
       
    def time(self,id):
        t = time.time()
        if id not in self.times:
            self.times[id] = 0
        self.times[id] += t-self.base
        base = t
         
    def report(self):
        for k in self.times.keys():
            v = self.times[k]
            print("{}: {}".format(k,int(v*1000)))

class AddBox():
    """Add a simple box mesh"""
    LOCATION_RANGE=45
    bl_idname = "mesh.fries_add"
    bl_label = "Add Fries"
    bl_options = {'REGISTER', 'UNDO'}

    width = FloatProperty(
            name="Width",
            description="Box Width",
            min=0.01, max=100.0,
            default=0.25,
            )
    height = FloatProperty(
            name="Height",
            description="Box Height",
            min=0.01, max=100.0,
            default=0.25,
            )
    depth = FloatProperty(
            name="Depth",
            description="Box Depth",
            min=0.01, max=100.0,
            default=1.0,
            )

    # generic transform props
    view_align = BoolProperty(
            name="Align to View",
            default=False,
            )
    location = FloatVectorProperty(
            name="Location",
            subtype='TRANSLATION',
            default=mu.Vector([0,0,0])
            )
    rotation = FloatVectorProperty(
            name="Rotation",
            subtype='EULER',
            )

    def execute(self, context,n, w, lmin, lmax, pcurve=False):
        bpy.context.scene.cursor_location = mu.Vector([0,0,0])
        t = Timer()
        objs = []
        for i in range(n):
            self.depth = random.uniform(lmin,lmax)
            verts_loc, faces = add_box(w,
                                       w,
                                       self.depth/2,
                                       )
            name = "Fry{}-{:0>4}".format(gen_num(),i)
            mesh = bpy.data.meshes.new(name)
            internal_name = mesh.name
            
            
            bm = bmesh.new()
            
            for v_co in verts_loc:
                bm.verts.new(v_co)

            bm.verts.ensure_lookup_table()
    
            for f_idx in faces:
                bm.faces.new([bm.verts[i] for i in f_idx])
    
            bm.to_mesh(mesh)
            mesh.update()
            # add the mesh as an object into the scene with this utility module
            ob = bpy.data.objects.new(internal_name,mesh)
            objs.append(ob)
            bpy.context.scene.objects.link(ob)
            #curvature
            if random.uniform(0.0,1.0) < pcurve:
                curve = ca.add_curve(context,self.depth, 3)
                ca.apply_curve(ob,curve)
        t.time('object_create')
        for obj in objs:
            obj.location[:] = \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(self.LOCATION_RANGE,6*self.LOCATION_RANGE)
            obj.rotation_euler[:] = random.uniform(-math.pi/8,math.pi/8),\
                     random.uniform(0,2*math.pi),\
                     random.uniform(0,2*math.pi)
        t.time('transform')
        fry_group = bpy.data.groups['Fries-Auto']
        for obj in objs:               
            fry_group.objects.link(obj)
            obj.modifiers.new('col','COLLISION')
            bpy.context.scene.rigidbody_world.group.objects.link(obj)
            ##bpy.ops.object.group_link(group="Fries-Auto")
            #bpy.context.scene.objects.active = obj
            ##bpy.ops.object.modifier_add(type='COLLISION')
            #bpy.ops.rigidbody.object_add()
            #obj.rigid_body.use_deactivation = True
            ##bpy.context.object.game.physics_type = 'RIGID_BODY'
            ##bpy.context.object.game.use_collision_bounds = True
            ##bpy.context.object.game.collision_bounds_type = 'BOX'
        t.time('group/collision')
        t.report()
        return {'FINISHED'}


def generate(n,w, lmin, lmax, pcurve):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.screen.frame_jump(end=False)
    op = AddBox()
    op.execute(bpy.context,n,w, lmin, lmax, pcurve)
    print("Fries Generated {}".format(bpy.context.scene.gen_number))
    bpy.context.scene.gen_number += 1
    


if __name__ == "__main__":
    generate(1,1,12,16)