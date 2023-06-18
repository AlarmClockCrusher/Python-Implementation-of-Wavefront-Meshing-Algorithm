### Most important thing in the meshing algorithm is to detection of collision between element to generate and existing elements. No touching of these elements is allowed, except coincidig nodes and edges:
<img width="250" alt="Untitled5" src="https://user-images.githubusercontent.com/61217720/210154601-c8577c59-d19b-4d40-a0a5-cffff8e301b5.png">


An edge can be in at most 2 triangles (facesMax=2) (boundaries can only have 1 instead). If an edge is in less than facesMax triangles, it is treated as active.

### The generation of new elements proceeds as waves. We can generate triangle elements from active edges. These elements have new nodes, whose envelope is like a wave front.

![2D_1Front](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/5323d772-8713-4a00-b25c-03843c1c86dd)

  1. Check if there are existing nodes close enough to an active edge. If there are nearby nodes, pick the one closest to the edge's bisector(normal) and doesn't form triangles that collide with existing edges.
  
  <img width="200" alt="2D_Nearby" src="https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/c04c43b8-c61e-4644-8418-077d0a26f3db">
  
  2. Try to generate an element from the height. The height is multiplied by a factor > 1 during detection, just to be safe. If there is a collision, pick a node that collides and forms a new triangle that doesn't collide with existing elements.

<img width="400" alt="Bud" src="https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/947fbfd8-70d4-437d-8520-0094adef17c9">

  3. After finishing active edges, close up the space around nodes on the wave front. If the angle between the two edges is small, try form triangle using those two edges; if the angle is large, perform step 2 from one the two edge then come back to step 3 if the node still has opening space. 
  <img width="400" alt="Closeup" src="https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/478d9e20-3619-432e-a7d4-fce209d91a0b">

### After 1 wave front generation
![2D_1Front_Finished](https://github.com/AlarmClockCrusher/Python-Implementation-of-Wavefront-Meshing-Algorithm/assets/61217720/15a0e310-f3eb-4138-87aa-937868cc3e8c)
