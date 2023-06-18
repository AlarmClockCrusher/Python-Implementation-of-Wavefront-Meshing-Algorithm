from SortPolygonVerts_fromSTL import *
from GenMesh_WaveFront import *
import stl, time

your_mesh = stl.mesh.Mesh.from_file('3DModel.stl')
cs_tris = your_mesh.vectors

func_height = 0.5
ls_polies = getVecsPolies_splitConns_fromSTL(cs_tris, height=func_height)
domain_3D, ls_domains_2D = create3DDomain_init2DBoundaries(ls_polies, func_height)


"""
Plot meshed boundaries
"""
ax = plt.figure(dpi=100).add_subplot(projection="3d")
ax.set(xlabel="x", ylabel="y", zlabel="z")

i = 0
for i, (domain, poly) in enumerate(zip(ls_domains_2D, ls_polies)):
	cs, n_edges = npA([pt.coor for pt in poly.pts_split]), poly.n_edges
	ax.quiver(*((cs + np.roll(cs, -1, axis=0)) / 2).T, *(0.1 * n_edges).T, color="red")
	ax_plotPolygons(ax, [ele.coors for ele in domain.elements_2D])
	for node in domain.nodes:
		ax.scatter(*node.coor)
		ax.text(*node.coor, node.i, color="orange")
	normals = npA([ele.normal for ele in domain.elements_2D])
	cs_cen = npA([ele.c_cen for ele in domain.elements_2D])
	ax.quiver(*cs_cen.T, *(normals/4).T, color="green")

cs_triVerts = cs_tris.reshape(-1, 3)
cs_max, cs_min = cs_triVerts.max(axis=0), cs_triVerts.min(axis=0)
range_max, (x, y, z) = (cs_max - cs_min).max(), (cs_max + cs_min) / 2
ax.set(xlim=(x-range_max/2, x+range_max/2),
	   ylim=(y-range_max/2, y+range_max/2), zlim=(z-range_max/2, z+range_max/2))

plt.show()


"""
Generate tetra elements
"""
domain_3D.prepare4Mesh()
#for f in domain_3D.elements_2D: f.normal = -f.normal

t1 = time.time()
try_catch, maxIter = True, 1
if try_catch:
	try: domain_3D.genTetrahedronEle_Wavefront(func_height, maxIter=maxIter)
	except Exception as e: print("Error!", e)
else: domain_3D.genTetrahedronEle_Wavefront(func_height, maxIter=maxIter)
t2 = time.time()
print("Done meshing. Time {}s".format(t2 - t1))


"""
Save the meshing information as pickle, for Blender viewing
"""
cs_tetras, idx_tetras = [], []
for ele in domain_3D.elements_3D:
	nds = list(ele.nodes)
	cs_tetras.append(npA([nd.coor for nd in nds]))
	idx_tetras.append([nd.i for nd in nds])
ndCoors_idx = [(nd.coor, nd.i) for nd in domain_3D.nodes]

with open("Saved_Domain_3D.pkl", 'wb') as file:
	pickle.dump((cs_tetras, idx_tetras, ndCoors_idx), file)