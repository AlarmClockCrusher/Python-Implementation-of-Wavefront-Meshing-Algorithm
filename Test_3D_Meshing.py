from GenMesh_WaveFront import *


arr_verts = genXYZarray(0, 12, 2, 0, 12, 2, 0, 12, 2)

nodes_total = []
domain_3D = Domain_3D()

for f in (lambda c: c[0] == 0, lambda c: c[0] == 12,
			lambda c: c[1] == 0, lambda c: c[1] == 12,
			lambda c: c[2] == 0, lambda c: c[2] == 12,
		  ):
	c_verts = arr_verts[[i for i, c in enumerate(arr_verts) if f(c)]]
	c_verts, vecs_verts, normals = getVecs_ifCanFormConvexPolyPlane(c_verts)
	cs, vecs, normals = splitConvexPoly(c_verts, vecs_verts, normals, 3)
	faceNormal = getNormal_ofPlane(c_verts, normalize=True)
	if faceNormal.dot(np.array([6, 6, 6]) - np.average(c_verts)) < 0:
		faceNormal = -faceNormal
	
	domain = Domain_2D()
	nodes, edges = [], []
	for i, c in enumerate(cs):
		if nd := next((nd for nd in nodes_total if not (nd.coor - c).any()), None):
			nodes.append(nd)
		else:
			nodes.append(nd:=Node(domain, i, c))
			nodes_total.append(nd)
	for nd1, nd2 in zip(nodes, nodes[1:] + nodes[:1]):
		if edge := nd1.sharesEdgewith(nd2): edges.append(edge)
		else: edges.append(Edge(domain, nd1, nd2))
	for edge, normal in zip(edges, normals): edge.normal = normal
	
	#ax_plotDots_Vecs_Normals(ax, cs, vecs, normals)
	domain.prepare4Mesh(edges, nodes)
	domain.genTriangleEle_Wavefront(4)
	
	#domain.plotAllEles(fig, ax, showLabel=True, size=8)
	
	for face in domain.elements_2D: face.normal = faceNormal
	domain_3D.elements_2D += domain.elements_2D
	domain_3D.nodes += domain.nodes
	domain_3D.edges += domain.edges

domain_3D.prepare4Mesh()

try_catch = True
if try_catch:
	try:
		domain_3D.genTetrahedronEle_Wavefront(1, maxIter=3)
	except Exception as e: print("Error!", e)
else: domain_3D.genTetrahedronEle_Wavefront(1, maxIter=3)

fig = plt.figure()
fig.set_size_inches(9, 4.5)

def oneIncreGen(ax1, ax2, domain, height=3, explosionCenter=None):
	try: domain.genTetrahedronEle_Wavefront(height, maxIter=1, maxNEle=1, reportSummary=True)
	except Exception as e: print("Error!", e)
	domain.update_activeFaces(ax1)
	domain.update_elements(ax2, explosionCenter=explosionCenter)
	
gs = gridspec.GridSpec(1, 2)
gs.update(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.15, hspace=0.14)
ax = plt.subplot(gs[0, 0], projection="3d")
ax1 = plt.subplot(gs[0, 1], projection="3d")
ax.set_box_aspect((1, 1, 1))
ax1.set_box_aspect((1, 1, 1))

domain_3D.plotAllActiveFaces(fig, ax)
domain_3D.plotAllEdges(ax, lw=0.4)


domain_3D.plotAllElements(ax1, explosionCenter=npA([6, 6, 6]), explosionFactor=0.5)
domain_3D.plotAllEdges(ax1, lw=0.4)

btn_OneIncre = Button(ax=plt.axes([0.05, 0.02, 0.12, 0.04]), label="Incre 1", hovercolor="0.95")
btn_OneIncre.on_clicked(lambda event: oneIncreGen(ax, ax1, domain_3D, explosionCenter=npA([6, 6, 6])))

txt_face = TextBox(plt.axes([0.25, 0.02, 0.10, 0.05]), 'Face', initial='')
btn_OneSearchFace = Button(ax=plt.axes([0.36, 0.02, 0.10, 0.04]), label="Search", hovercolor="0.95")
def getFaceCoors(domain, s):
	if f := next((f for f in domain.elements_2D if "{}".format(f) == s), None):
		print("Found face:", f, "\n", (f.coors,))
	else: print("Didn't find face. Check input")
	
btn_OneSearchFace.on_clicked(lambda event: getFaceCoors(domain_3D, txt_face.text))

ax_Init(ax, widgets := [])
plt.show()