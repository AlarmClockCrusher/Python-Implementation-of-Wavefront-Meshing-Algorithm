import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go


def ax_plotPolygons(ax, coors, alpha=0.2, color="blue"):
	#coors.shape: (m, 3, 3)
	polyCol = a3.art3d.Poly3DCollection(coors, alpha=alpha, color=color, edgecolor='k')
	ax.add_collection3d(polyCol)
	return polyCol


def go_plotTetra(fig, cs_tetra, color="blue", opacity=0.35):
	#cs_tetra.shape: (m, 4, 3) -- m tetrahedrons, each with 4 vertices
	x, y, z = cs_tetra.reshape(-1, 3).T
	fig.add_mesh3d(x=x, y=y, z=z, i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3],
				   # colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']], intensity=[0, 0.33, 0.66, 1],
				   color=color, opacity=opacity)
	Xe, Ye, Ze = cs_tetra[[0, 1, 2, 0, 3, 2, 3, 1]].T
	lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color='rgb(50,50,50)', width=1.5))
	fig.add_trace(lines)


def go_plotTri(fig, cs_tri, color="blue", opacity=0.35):
	# cs_tri.shape: (m, 3, 3) -- m tetrahedrons, each with 3 vertices
	x, y, z = cs_tri.reshape(-1, 3).T
	fig.add_mesh3d(x=x, y=y, z=z, color=color, opacity=opacity)
	Xe, Ye, Ze = cs_tri[[0, 1, 2, 0]].T
	lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color='rgb(50,50,50)', width=1.5))
	fig.add_trace(lines)  # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker_size=3))

def go_scatter_text(fig, cs_nodes, withMarkers=True, withText=True,
		textposition="top right", text=None, textcolor="black"):
	# cs_nodes.shape: (m, 3) -- m nodes. ith node will be labeled with a text in the plot
	xs, ys, zs = cs_nodes.T
	if withMarkers:
		if withText: mode = "markers+text"
		else: mode = "markers"
	elif withText: mode, textposition = 'text', "middle center"
	else: mode = "markers"
	if text is None: text = [str(i) for i in range(len(cs_nodes))]
	fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode=mode, text=text, textposition=textposition,
							   textfont=dict(family="sans serif", size=15, color=textcolor)
							   ))