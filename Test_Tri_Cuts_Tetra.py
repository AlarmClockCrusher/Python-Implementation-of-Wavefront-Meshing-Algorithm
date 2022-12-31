from timeit import timeit
from GeometryFuncs import *

gs = gridspec.GridSpec(2, 3)
gs.update(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.35, hspace=0.34)
ax = plt.subplot(gs[0:2, 0:2], projection="3d")
ax1, ax2 = plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2])
ax.set_box_aspect((1, 1, 1))

cs_tetra = np.random.uniform(0, 0.6, size=(4, 3))
cs_tri = np.random.uniform(0, 0.6, size=(3, 3))
cs_tetra = npA([[4.08603747, 6.3281761 , 9.86933249],
       [4.41421356, 9.        , 4.41421356],
       [9.        , 4.41421356, 8.41421356],
       [4.53007524, 4.18593322, 6.31454868]])

cs_tri = npA([[4.41421356, 9. ,        4.41421356],
			  [9.  ,       4.41421356, 8.41421356],
			  [ 6. , 0., 12.]])
#cs_tetra[0] = cs_tri[0] = [1, 0, 0]
#cs_tetra[1] = cs_tri[1] = [0, 1, 0]
#cs_tetra[2] = cs_tri[2] = [0, 0, 1]
compareTime = False

v_ithTriV_from_jthTetraV = (np.tile(cs_tri.reshape(-1, 1, 3), (1, 4, 1)) - cs_tetra).round(DIGIT_ACCU - 1)
isSame_ithTri_jthTetra = (v_ithTriV_from_jthTetraV == 0).all(axis=2)
inds_tri, inds_tetra = np.nonzero(isSame_ithTri_jthTetra)
inds_tetra_diff = [i for i in (0, 1, 2, 3) if i not in inds_tetra]

ns_edge = np.cross(cs_tri[[2, 0, 1]] - cs_tri[[1, 2, 0]], n_face := np.cross(*(cs_tri[1:] - cs_tri[0])))
ns_tetraFace = getNormals_OutofTetra(cs_tetra)

if compareTime:
	number = int(4e3)
	print("allTriHalfPlanes_touch_Tetra\n",
		  timeit("allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra)",
				 setup="from __main__ import allTriHalfPlanes_touch_Tetra, cs_tri, cs_tetra, ns_edge, n_face", number=number))
	print("verify_Triangle_cuts_Tetra\n",
		  timeit("verify_Triangle_cuts_Tetra(cs_tri, cs_tetra)",
				 setup="from __main__ import verify_Triangle_cuts_Tetra, cs_tri, cs_tetra, ns_edge, n_face", number=number))
	print("check_Triangle_cuts_Tetrahedron\n",
		  timeit("check_Triangle_cuts_Tetrahedron(cs_tri, cs_tetra)",
				 setup="from __main__ import check_Triangle_cuts_Tetrahedron, cs_tri, cs_tetra", number=number))
	print("\nverify_triV_to_tetraN\n",
		  timeit("verify_triV_to_tetraN(v_ithTriV_from_jthTetraV, cs_tetra, ns_tetraFace)",
				 setup="from __main__ import verify_triV_to_tetraN, v_ithTriV_from_jthTetraV, cs_tetra, ns_tetraFace", number=number))


ax_plotPolygons(ax, [cs_tri])
ax_textPoints(ax, cs_tri)
ax_plotTetra(ax, cs_tetra, color='red')
ax_textPoints(ax, cs_tetra, color='red')
ax.set(xlabel='x', ylabel="y", zlabel="z")
ax.quiver(0, 0, 0, *np.diag([0.5, 0.5, 0.5]))
ax.quiver(*np.average(cs_tri, axis=0).T, *n_face)
ax.quiver(*((cs_tri[[2, 0, 1]]+cs_tri[[1, 2, 0]])/2).T, *ns_edge.T)


"""View the projection of unshared tetra verts along triangle edges"""
for n_edge, c0, color in zip(ns_edge, cs_tri[[1, 2, 0]],
							 ("red", "green", "blue")):
	coors_wrt_edge = np.array([n_edge.dot((cs_tetra_wrt:=cs_tetra-c0).T), n_face.dot(cs_tetra_wrt.T)]).T
	coors_wrt_edge = coors_wrt_edge[inds_tetra_diff]
	for (x1, y1), (x2, y2) in combinations(coors_wrt_edge, 2):
		ax1.plot([x1, x2], [y1, y2], color=color)
	for i, c in zip(inds_tetra_diff, coors_wrt_edge):
		ax1.text(*c, i, color=color)
	if len(inds_tetra_diff) > 1:
		print(color, "\tcheck_negX_cuts_quadrilateral_2D", check_negX_cuts_SegmentsbetweenVerts_2D(coors_wrt_edge))
ax1.plot([-100, 0], [0, 0], color="black")
ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.grid()

print("allTriHalfPlanes_touch_Tetra: (doesn't consider vert coinciding)")
print(allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra[[0, 2, 3]], ns_edge, n_face))
print("\nverify_Triangle_cuts_Tetra")
print(verify_Triangle_cuts_Tetra(cs_tri, cs_tetra, ns_edge, n_face))

cm = plt.cm.get_cmap('rainbow')
sols = np.random.uniform(-1, 1, size=(10, 2, 3))
for sol1, sol2 in sols:
	v = np.cross(np.cross(sol1, sol2), sol1 - sol2)
	color = cm(np.random.uniform())
	
	ax.quiver(0, 0, 0, *v, color=color)
	ax.plot(*npA([[0, 0, 0], sol1, sol2, [0, 0, 0]]).T, color=color)
	#ax.quiver(0, 0, 0, *sol1, color=color)
	#ax.quiver(0, 0, 0, *sol2, color=color)

"""Test the check_negX_cuts_SegmentsbetweenVerts_2D function"""
cs = np.random.uniform(-1, 1, size=(4, 2))

for (x1, y1), (x2, y2) in combinations(cs, 2):
	ax2.plot([x1, x2], [y1, y2], color="red")
ax2.text(-1.3, 1.1, "$-\hat x$ cuts quadri {}".format(check_negX_cuts_SegmentsbetweenVerts_2D(cs)))
ax2.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))
ax2.plot([-2, 0], [0, 0], color="blue")
ax_textPoints(ax2, cs)
ax2.grid()

plt.show()

