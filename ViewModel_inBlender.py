import bpy, pickle, os

"""
Stackexchange thread: (Good code examples for Blender 2.8x and Blender 2.7x)
https://blender.stackexchange.com/questions/23086/add-a-simple-vertex-via-python

Adding text: https://www.youtube.com/watch?v=D-pTF3KrTOQ
"""
root = r"C:\Users\gsd\Desktop\Math_Docs\FEA_Test\FEA_Tests\\"

def Print(*args):
    with open(os.path.join(root, "BlenderOutput.txt"), 'a') as file:
        file.write(' '.join(args)+"\n")

def addText_2Blender(text, location=(0, 0, 0), rotation=(0, 0, 0)):
    bpy.ops.object.text_add(location=location, rotation=rotation) #This returns a set: {"FINISHED"}
    bpy.ops.object.editmode_toggle()
    bpy.ops.font.move_select(type="LINE_BEGIN")
    bpy.ops.font.delete(type="SELECTION")
    bpy.ops.font.text_insert(text=text, accent=False)
    bpy.ops.object.editmode_toggle()
    
def addModel_2collection(obj_name, coords, edges=[], faces=[], showName=False):
    """
    coords (tuples of node coordinates): [[0, 0, 0], [1.2, 1.4, 5], [2, 3, 3], [10, 3, 3.22]]
    edges examples (tuples of node indices): [[0, 1], [1, 2], [0, 3]]
    faces examples (tuples of node indices): [[0, 1, 2], [0, 2, 3], [1, 2, 3]]
    """
    # Create the object
    mesh = bpy.data.meshes.new(obj_name + "Mesh")
    mesh.from_pydata(coords, edges, faces)
    #me.update() #Probably only needed for changing existing model mesh
    obj = bpy.data.objects.new(obj_name, mesh)
    obj.show_name = showName
    # Link object to the active collection
    bpy.context.collection.objects.link(obj)
    # Alternatively Link object to scene collection
    #bpy.context.scene.collection.objects.link(obj)
    obj.select_set(state=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    obj.select_set(state=False)
    

with open(os.path.join(root, "Saved_Domain_3D.pkl"), "rb") as f:
    cs_tetras, idx_tetras, ndCoors_idx = pickle.load(f)
    
    
idx_4choose3 = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
for cs_tetra, idx in zip(cs_tetras, idx_tetras):
    name = "Tetra-"+'-'.join(str(i) for i in sorted(idx))
    addModel_2collection(name, cs_tetra, faces=idx_4choose3, showName=False)
    
for c, i in ndCoors_idx:
    addModel_2collection(str(i), [c], showName=True)
    
for obj in bpy.data.objects:
    if obj.type == 'FONT': obj.name = obj.data.body[:10]