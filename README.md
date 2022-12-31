# Python-Implementation-of-Wavefront-Meshing-Algorithm
An educational example of how wavefront meshing algorithm is implemented.

## Run Test_2D_Meshing.py and Test_3D_Meshing.py to view meshing results.
### Test_Tri_Cuts_Tetra.py is to check the collision between tetrahedron and triangle
### Required packages: numpy, matplotlib

In 2D case, start with boundaries. Each boundary edge must have a normal that points inward the domain. For holes, the edges need to flipped outward.

![2D_meshing_boundary](https://user-images.githubusercontent.com/61217720/210152892-841cf16a-d6f2-49b8-b8a6-3fba37a9caec.png)

Result of triangle meshing:

![2D_meshing](https://user-images.githubusercontent.com/61217720/210152777-5d06d8b9-9d23-4f3a-b8c0-e21fb2f84c53.png)

#### Most important thing in the meshing algorithm is to detection of collision between element to generate and existing elements. No touching of these elements is allowed, except nodes:
<img width="250" alt="Untitled5" src="https://user-images.githubusercontent.com/61217720/210154601-c8577c59-d19b-4d40-a0a5-cffff8e301b5.png">


An edge can be in 2 triangles (facesMax=2) (boundaries can only have 1 instead), and each time a triangle element is formed, the edges will check if numFaces < facesMax. New edges generated are seen as ACTIVE, old edges are no longer active. New edges will have their own normals (pointing outward of the triangles). New triangles formed must have its 3rd node have a positive projection along the normal (**right direction**).
### When generating a new triangle off an edge AB:

  1. Check if AB has its 2 nodes connected by a 3rd node C. If yes, then forming a triangle using these A&B&C would be closing up a void. There can be multiple "3rd" nodes that satisfy this criteria, then pick the node in right direction and closest to the AB. If any such node C is found, then this will be the new triangle.
  <img width="250" alt="Untitled" src="https://user-images.githubusercontent.com/61217720/210154261-51a12f52-fb73-4bcb-a7e3-8438e7c78675.png">

  
  2. Check if one node A of the edge AB has another edge AC that can form a small enough angle $\angle BAC$ with this edge. Try to close up the two hinged edges AB&AC by connecting the other two nodes BC. In this process, we check if this new triangle ABC will collide with existing edges. If no, this will be a qualified new triangle; if yes, then this triangle must be abondoned. There can be multiple nodes that form small angles, and we start with the nodes in the right direction and closest to edge AB. If any such node C is found, then this will be the new triangle.
  <img width="215" alt="Untitled2" src="https://user-images.githubusercontent.com/61217720/210154264-eb7cb1b9-3047-4bfb-8b49-5dece284978a.png">

  3. Look for nodes that are close enough to the center of AB and in the right direction. There can be multiple such nodes, and start with the node C closest to the edge AB, and make sure ABC doesn't collide with existing edges. If any such node C is found, then this will be the new triangle.
  <img width="185" alt="Untitled3" src="https://user-images.githubusercontent.com/61217720/210154266-b836b458-c9a5-4885-bcb2-8a7b585f58d1.png">

  3. Use the edge AB to form a isosceles triangle ABC with a given height h. Check if ABC collide with any existing edge. If no, then ABC will be the new triangle; if yes, then get all nodes {C} that belong to the collided edges (excluding A&B). Start with node closest to the edge AB, and get the first one that forms a triangle that doesn't collide with existing edges.
  <img width="169" alt="Untitled4" src="https://user-images.githubusercontent.com/61217720/210154269-cc9285d3-5562-4dc9-8fff-da02674365ab.png">
