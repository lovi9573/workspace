import bpy
import bmesh
import random
import math
import mathutils as mu


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


class AddBox(bpy.types.Operator):
    """Add a simple box mesh"""
    LOCATION_RANGE=15
    N=144
    bl_idname = "mesh.primitive_box_add"
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
            )
    rotation = FloatVectorProperty(
            name="Rotation",
            subtype='EULER',
            )

    def execute(self, context):
        if bpy.ops.rigidbody.world_add.poll():
            bpy.ops.rigidbody.world_add()
        for i in range(self.N):
            self.depth = random.uniform(1.0,5.0)
            verts_loc, faces = add_box(self.width,
                                       self.height,
                                       self.depth,
                                       )
    
            mesh = bpy.data.meshes.new("Fry"+str(i))
    
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
            object_utils.object_data_add(context, mesh, operator=self)
            rot = bpy.context.scene.objects.active.rotation_euler
            rot[:] = random.uniform(0,2*math.pi),random.uniform(0,2*math.pi),random.uniform(0,2*math.pi)
            loc = bpy.context.scene.objects.active.location
            loc += mu.Vector([random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                              random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE), \
                              random.uniform(-self.LOCATION_RANGE,self.LOCATION_RANGE)])
            #bpy.context.scene.objects.active.rotation_euler = (12,14,15,'XYZ')
            bpy.context.object.game.physics_type = 'RIGID_BODY'
            bpy.context.object.game.use_collision_bounds = True
            bpy.context.object.game.collision_bounds_type = 'BOX'
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(AddBox.bl_idname, icon='MESH_CUBE')


def register():
    bpy.utils.register_class(AddBox)
    bpy.types.INFO_MT_mesh_add.append(menu_func)


def unregister():
    bpy.utils.unregister_class(AddBox)
    bpy.types.INFO_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
    register()

    # test call
    bpy.ops.mesh.primitive_box_add()
