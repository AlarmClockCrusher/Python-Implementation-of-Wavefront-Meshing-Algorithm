from GenMesh_WaveFront import *
from SortPolygonVerts_fromSTL import *
import stl

"""
Load STL model, either as 2D or 3D
"""
stl_mesh = stl.mesh.Mesh.from_file('2DModel.stl')
dim, idx = checkif2D_getXYZIdx(stl_mesh) #[0, 1]: x-y plane, [0, 2]: x-z plane, [1, 2]: y-z plane, [0, 1, 2]: 3D

#height = lambda c: 0.02 + 0.11*npNorm(c) #A variable meshing height
height = 0.2

cs_tris = stl_mesh.vectors[..., idx]
ls_pts_onBoundaries, tris = extract_sort_verts_boundaries(cs_tris)
ls_cs, ls_normals = getVecs_for1Plane_sortedOuterInner(ls_pts_onBoundaries, height=height)


"""
View ls_cs, ls_normals
"""
if dim == 2:
	fig, ax = plt.subplots(dpi=100)
	ax.set(xlabel="x", ylabel='y')
else:
	ax = plt.figure().add_subplot(projection="3d")
	ax.set(xlabel="x", ylabel='y', zlabel='z')

for pts in ls_pts_onBoundaries:
	for pt in pts:
		ax.text(*pt.coor, str(pt.i))#, ha="center", va="center")

if dim == 2:
	for cs_tri in cs_tris: ax.fill(*cs_tri.T, alpha=0.3)
else: ax_plotPolygons(ax, cs_tris)

for cs, normals in zip(ls_cs, ls_normals):
	print(cs.shape, normals.shape)
	ax.scatter(*cs.T, s=5)
	if dim == 2:
		for n, c1, c2 in zip(normals, cs, np.roll(cs, -1, axis=0)):
			ax.arrow(*(c1+c2)/2, *n/15, head_width=0.02, color="red")
	else:
		print("Really")
		ax.quiver(*((cs+np.roll(cs, -1, axis=0))/2).T, *(normals/8).T)

print("Finished")
plt.show()

"""
Mesh 2D
"""
domain = Domain_2D()
ls_nodes, ls_edges, j = [], [], 0
for cs in ls_cs:
	ls_nodes.append([Node(domain, j + i, c) for i, c in enumerate(cs)])
	j += len(ls_nodes)

for nodes, normals in zip(ls_nodes, ls_normals):
	edges = [Edge(domain, nd1, nd2) for nd1, nd2 in zip(nodes, nodes[1:] + nodes[:1])]
	ls_edges.append(edges)
	for edge, normal in zip(edges, normals): edge.normal = normal

domain.prepare4Mesh([eg for edges in ls_edges for eg in edges], [nd for nodes in ls_nodes for nd in nodes])

ax.scatter(*npA([nd.coor for nd in domain.nodes]).T)
for nd in domain.nodes:
	ax.text(*nd.coor, str(nd.i))
if dim == 2: ax.set_aspect(1)
plt.show()


try_catch, maxIter = True, 10
if try_catch:
	try: domain.genTriangleEle_Wavefront(height, maxIter=maxIter)
	except Exception as e: print("Error!", e)
else: domain.genTriangleEle_Wavefront(height, maxIter=maxIter)


try: domain.dissolveSmallTriangles(0.3)
except Exception as e:
	print("Error during dissolving")
	
	
cs_tris, idx_tris = [], []
for ele in domain.elements_2D:
	nds = list(ele.nodes)
	cs_tris.append(npA([nd.coor for nd in nds]))
	idx_tris.append([nd.i for nd in nds])
ndCoors_idx = [(nd.coor, nd.i) for nd in domain.nodes]

with open("Saved_Domain_3D.pkl", 'wb') as file:
	pickle.dump((cs_tris, idx_tris, ndCoors_idx), file)
	
	
if dim == 2:
	fig, ax = plt.subplots(dpi=100)
	ax.tick_params("both")
	for ele in domain.elements_2D: ax.fill(*ele.coors.T, alpha=0.3)
else:
	ax = (fig := plt.figure(dpi=100)).add_subplot(111, projection="3d")
	ax_plotPolygons(ax, np.array([ele.coors for ele in domain.elements_2D]))
for node in domain.nodes: ax.text(*node.coor, str(node.i))

if dim == 2: ax.set_aspect(1)
plt.show()