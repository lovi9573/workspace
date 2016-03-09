import bpy
import bmesh
import random
import math
import time
import mathutils as mu
from bpy_extras import object_utils
#import curve_add as ca
import imp
#imp.reload(ca)

def add_curve( context, length, num_verts):
    group = bpy.data.groups["Auto-Curves"]
        
    segment_length = length/num_verts
    y_0 = -length/2 - segment_length
    num_verts +=2
    verts = []
    n = lambda : random.normalvariate(0,2)
    for v in range(num_verts+1 ):
        offset = v*segment_length
        verts += [(n(),y_0+offset,n())]

    data = bpy.data.curves.new('curve', type='CURVE')
    data.dimensions = '3D'
    data.resolution_u = 2

    polyline = data.splines.new('NURBS')
    polyline.points.add(len(verts)-1)
    #polyline.bezier_points.add(len(verts))
    for i, v in enumerate(verts):
        x,y,z = v
        polyline.points[i].co = (x, y, z, 1)
        #polyline.bezier_points[i].co = (x, y, z)

    curveOB = bpy.data.objects.new('curve', data)
    #curveOB.location = mu.Vector([0,0,0])
    bpy.context.scene.objects.link(curveOB)
    context.scene.cursor_location = verts[1]
    bpy.ops.object.select_all(action='DESELECT')
    curveOB.select = True
    bpy.context.scene.objects.active = curveOB
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    return curveOB

def apply_curve(obj, curve, axis='POS_Y'):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    bpy.ops.object.modifier_add(type='CURVE')
    obj.modifiers['Curve'].object = curve
    obj.modifiers['Curve'].deform_axis = axis
    bpy.context.scene.objects.active = obj
    bpy.ops.object.modifier_apply(apply_as='DATA',modifier='Curve')
    bpy.ops.object.select_all(action='DESELECT')
    curve.select = True
    bpy.ops.object.delete(use_global=False)


def gen_num():
    return bpy.context.scene.gen_number

def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """
    
    segments = 1
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
    w = width/2.0
    h = height/2.0
    d = depth/2.0
    for i, v in enumerate(verts):
        verts[i] = v[0] * w, v[1] * d, v[2] * h
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
    LOCATION_RANGE=15
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

    def execute(self, context,volume, w, lmin, lmax, mean, pcurve=False):
        bpy.context.scene.cursor_location = mu.Vector([0,0,0])
        t = Timer()
        objs = []
        generated_volume = 0.0
        i = 0
        while generated_volume < volume:
            self.depth = random.normalvariate(mean,2.5)
            self.depth = max(self.depth,lmin)
            self.depth = min(self.depth,lmax)
            verts_loc, faces = add_box(w,
                                       w,
                                       self.depth,
                                       )
            generated_volume += w*w*self.depth
            name = "Fry-{:0>5}".format(i)
            if name not in bpy.data.meshes:
              mesh = bpy.data.meshes.new(name)
            else:
              mesh = bpy.data.meshes[name]
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
                curve = add_curve(context,self.depth, 3)
                apply_curve(ob,curve)
            i +=1
        t.time('object_create')
        for obj in objs:
            obj.location[:] = \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(self.LOCATION_RANGE,8*self.LOCATION_RANGE)
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
        print("{} Fries Generated".format(i))
        return {'FINISHED'}


def generate(volume,w, lmin, lmax, mean, pcurve):
    print("Generation {}".format(bpy.context.scene.gen_number))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.screen.frame_jump(end=False)
    op = AddBox()
    op.execute(bpy.context,volume,w, lmin, lmax, mean, pcurve)
    bpy.context.scene.gen_number += 1
    


if __name__ == "__main__":
    generate(1000,0.635,2,12,9,0)