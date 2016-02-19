import bpy
import bmesh
import random
import math
import mathutils as mu


def gen_num():
    return bpy.context.scene.gen_number

def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """

    verts = [(+1.0, +1.0, -1.0),
             (+1.0, -1.0, -1.0),
             (-1.0, -1.0, -1.0),
             (-1.0, +1.0, -1.0),
             (+1.0, +1.0, +1.0),
             (+1.0, -1.0, +1.0),
             (-1.0, -1.0, +1.0),
             (-1.0, +1.0, +1.0),
             ]

    faces = [(0, 1, 2, 3),
             (4, 7, 6, 5),
             (0, 4, 5, 1),
             (1, 5, 6, 2),
             (2, 6, 7, 3),
             (4, 0, 3, 7),
            ]

    # apply size
    for i, v in enumerate(verts):
        verts[i] = v[0] * width, v[1] * depth, v[2] * height

    return verts, faces


from bpy.props import FloatProperty, BoolProperty, FloatVectorProperty


class AddBox():
    """Add a simple box mesh"""
    LOCATION_RANGE=25
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

    def execute(self, context,n):
        bpy.context.scene.cursor_location = mu.Vector([0,0,0])
        for i in range(n):
            self.depth = random.uniform(1.0,5.0)
            verts_loc, faces = add_box(0.25,
                                       0.25,
                                       self.depth,
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
            from bpy_extras import object_utils
            new_object = object_utils.object_data_add(context, mesh)
            rot = bpy.context.scene.objects.active.rotation_euler
            rot[:] = random.uniform(0,2*math.pi),random.uniform(0,2*math.pi),random.uniform(0,2*math.pi)
            loc = bpy.context.scene.objects.active.location
            loc[:] = random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                  random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE)
            
            bpy.ops.object.group_link(group="Fries-Auto")
            bpy.ops.object.modifier_add(type='COLLISION')
            bpy.ops.rigidbody.object_add()
            bpy.data.objects[internal_name].rigid_body.use_deactivation = True
            #bpy.context.object.game.physics_type = 'RIGID_BODY'
            #bpy.context.object.game.use_collision_bounds = True
            #bpy.context.object.game.collision_bounds_type = 'BOX'
            
        return {'FINISHED'}


def generate(n):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.screen.frame_jump(end=False)
    op = AddBox()
    op.execute(bpy.context,n)
    print("Fries Generated {}".format(bpy.context.scene.gen_number))
    bpy.context.scene.gen_number += 1
    


