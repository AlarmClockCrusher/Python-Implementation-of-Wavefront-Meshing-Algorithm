from GenMesh_WaveFront import *

fig, ax = plt.subplots(dpi=200)
fig.set_size_inches(3.2, 3.2)

"""
Create the coors, vecs, normals necessary for meshing boundaries
"""

#Circle. Permutation of many vertices could be slow. Here use predetermined order
thetas = np.linspace(0, 2*np.pi, 22)[:-1]
arr_verts = npA([0.6 * np.cos(thetas), 0.6 * np.sin(thetas)]).T
cs, vecs, normals = splitConvexPoly(*getVecs_AlreadyConvexandSorted(arr_verts), 0.12)
ax_plotDots_Vecs_Normals(ax, cs, vecs, normals)

#A hole in the circle. The normals of a closed hole must point outward. Need to flip the inward normals
thetas = np.linspace(0, 2*np.pi, 6)[:-1]
arr_verts_new = npA([0.2 * np.cos(thetas), 0.3 * np.sin(thetas)]).T
cs_new, vecs_new, normals_new = splitConvexPoly(*getVecs_AlreadyConvexandSorted(arr_verts_new), 0.12)
normals_new = - normals_new
ax_plotDots_Vecs_Normals(ax, cs_new, vecs_new, normals_new)

#Square.
#arr_verts = genXYarray(0, 1.2, 2, 0, 1.2, 2)
#cs, vecs, normals = splitConvexPoly(*getVecs_ifCanFormConvexPolyPlane(arr_verts), 0.15)
#ax_plotDots_Vecs_Normals(ax, cs, vecs, normals)

"""
Use the 
"""
domain = Domain_2D()
nodes = [Node(domain, i, c) for i, c in enumerate(cs)]
edges = [Edge(domain, nd1, nd2) for nd1, nd2 in zip(nodes, nodes[1:]+nodes[:1])]
for edge, normal in zip(edges, normals): edge.normal = normal


nodes_new = [Node(domain, i, c) for i, c in enumerate(cs_new)]
edges_new = [Edge(domain, nd1, nd2) for nd1, nd2 in zip(nodes_new, nodes_new[1:]+nodes_new[:1])]
for edge, normal in zip(edges_new, normals_new): edge.normal = normal


domain.prepare4Mesh(edges+edges_new, nodes+nodes_new)
f_height = lambda c: 0.04 + 0.2*np.linalg.norm(c) #A variable meshing height

#try_catch = True
#if try_catch:
#	try: domain.genTriangleEle_Wavefront(f_height, maxIter=9)
#	except Exception as e: print("Error!", e)
#else: domain.genTriangleEle_Wavefront(f_height, maxIter=9)


domain.dissolveSmallTriangles(0.3)

domain.plotAllEles(fig, ax, showLabel=True, size=4)
domain.plotAllNodes(ax, s=0.1)
domain.textAllNodes(ax, size=4, color="red")
domain.plotAllEdges(ax, lw=0.2)
domain.plotAllEdges_Active(ax, lw=0.8)

plt.show()