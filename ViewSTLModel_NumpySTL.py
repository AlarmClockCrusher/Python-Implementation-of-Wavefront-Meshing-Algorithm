from GeometryFuncs import *
from PlotMethods import *
import stl

gs = gridspec.GridSpec(1, 2)
figure = plt.figure(dpi=200)
figure.set_size_inches(6, 3)

stlMesh = stl.mesh.Mesh.from_file('3DModel.stl')

axes = plt.subplot(gs[0, 0], projection='3d')
axes1 = plt.subplot(gs[0, 1], projection='3d')
axes.set_title("Triangles of stl file")
axes1.set_title("Edges of stl file")

#Plot the triangles that form the model in axes, and edges those triangles have (non-repetitive) in axes1
axes.add_collection3d(a3.art3d.Poly3DCollection(stlMesh.vectors, alpha=0.5, edgecolor="k"))
cs_nodes, edges_nonrepetitive, nodePairs_nonrepetitive, normals = STLMesh_Verts_EdgeVecs_EdgeNodePairs_TriangleNormals(stlMesh, fullyClosed=False)
axes1.add_collection3d(a3.art3d.Line3DCollection(nodePairs_nonrepetitive.reshape(-1, 2, 3), lw=0.5))

#Tweak the scale of axes for display
xs, ys, zs = stlMesh.x.flatten(), stlMesh.y.flatten(), stlMesh.z.flatten()
range_x, range_y, range_z = xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
range_max = max(range_x, range_y, range_z)
x_cen, y_cen, z_cen = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2, (zs.max() + zs.min()) / 2
axes.set_box_aspect((1, 1, 1))
axes1.set_box_aspect((1, 1, 1))
axes.set(xlabel="x", xlim=(x_cen-range_max/2, x_cen+range_max/2),
		 ylabel="y", ylim=(y_cen-range_max/2, y_cen+range_max/2),
		 zlabel="z", zlim=(z_cen-range_max/2, z_cen+range_max/2))
axes1.set(xlabel="x", xlim=(x_cen-range_max/2, x_cen+range_max/2),
		  ylabel="y", ylim=(y_cen-range_max/2, y_cen+range_max/2),
		  zlabel="z", zlim=(z_cen-range_max/2, z_cen+range_max/2))

plt.show()