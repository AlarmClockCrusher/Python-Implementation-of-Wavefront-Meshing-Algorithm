# Python-Implementation-of-Wavefront-Meshing-Algorithm
An educational example of how wavefront meshing algorithm is implemented.

**Required packages: numpy, matplotlib. Required software: Blender >2.9x.**

## STL models are loaded and meshed in this project, with Test_2D_Meshing.py and Test_3D_Meshing.py. 
### Explanation of scripts
CollisionDetection.py has functions for checking if a geometric object (triangles/tetrahedrons) cuts other existing objects in the meshing process.

GenMesh_WaveFront.py has the core algorithm of meshing.

GeometryFuncs.py has some simpler functions for geometry calculations (can have many unused methods).

PlotMethods.py has basic functions that plots geometry objects in the mesh.

SortPolygonVerts_fromSTL.py has functions for converting STL model triangles into domain boundaries with normal of each edge.

Test_2D_Meshing.py meshes a plane. (Vertices on the same plane, either in xy/yz/xz planes or tilted plane)

Test_3D_Meshing.py meshes a 3D model, and generates Saved_Domain_3D.pkl pickle file

ViewModel_inBlender.py **needs to run in Blender**. It loads the generated Saved_Domain_3D.pkl file and creates models of tetrahedrons.

ViewSTLModel_NumpySTL.py provides basic viewing of 2DModel.stl and 3DModel.stl.

## Demo of meshing results
In 2D case, first load the 2DModel.stl, which is comprised of triangles. Arrows represent inward normals from outer boundaries and outward normals from inner boundaries (holes).

![2D_STL_Triangles_Eges_Normals](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/d2f49378-d5d2-42b0-bf17-4759a1578e9b)

After meshing.

![2D_meshing](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/4f24bce2-a807-428b-a1cf-d5bd10fcf94f)

In 2D case with tilted plane.

![3D_STL_Triangles_Eges_Normals](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/81c29d49-8649-4925-8b9a-d8f882ee5c9f)

After meshing

![3D_meshing](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/3f60e2c2-842c-47be-ba90-f98f0331235d)


In 3D case.

![3D_STL](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/a223a616-dc12-4906-bbc8-54d919d59dd2)

After meshing (only generate 1 layer)

![3D_ModelMeshing](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/4123a5bb-5b31-4d74-aa02-e10fc0db0211)
