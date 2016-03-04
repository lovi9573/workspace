import bpy
import bmesh
import random
import math
import mathutils as mu



from bpy.props import FloatProperty, BoolProperty, FloatVectorProperty


def add_curve( context, length, num_verts):
    bpy.ops.group.create()
    group = bpy.data.groups[-1]
    group.name = "Auto-Curves"+str(len(bpy.data.groups))
    
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

if __name__ == "__main__":
    curve = add_curve(bpy.context,16, 3)
    apply_curve(bpy.data.objects['Cube'],curve)
    
