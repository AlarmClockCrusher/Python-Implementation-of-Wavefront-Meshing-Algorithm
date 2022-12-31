from GeometryFuncs import *


class Node:
	def __init__(self, domain, i, coor):
		self.i, self.coor = i, coor
		self.edges, self.faces, self.volumes = set(), set(), set()
		domain.nodes.append(self)
	
	def __repr__(self): return "N%d" % self.i
	
	def sharesEdgewith(self, node):
		if commonEdges := self.edges.intersection(node.edges): return next(iter(commonEdges))
		else: return None

	def nds_connectedbyEdge(self):
		return {nd for eg in self.edges for nd in eg.nodes if nd is not self}
	
class Edge:
	def __init__(self, domain, *nodes):
		self.nodes = nd1, nd2 = set(nodes)
		self.coors = npA([nd1.coor, nd2.coor])
		self.c_cen, self.radius = np.average(self.coors, axis=0), np.linalg.norm(nd1.coor-nd2.coor) / 2
		for nd in nodes: nd.edges.add(self)
		self.faces, self.numFaces, self.facesMax = set(), 0, 2
		self.normal = None
		domain.edges.append(self)
	
	def __repr__(self):
		n1, n2 = self.nodes
		if n1.i < n2.i: return "E%d-%d" % (n1.i, n2.i)
		else: return "E%d-%d" % (n2.i, n1.i)
		

class Ele_Triangle:
	def __init__(self, domain, nodes, is2D=True):
		self.nodes, self.edges, self.volumes = set(nodes), set(), set()
		if len(nodes) != 3: print("Num nodes wrong when gen triangle", nodes, self.nodes); raise Exception
		domain.tups_eleNodes.append(self.nodes)
		self.numVolumes, self.volumesMax = 0, 2
		self.coors = cs = npA([nd.coor for nd in nodes])
		self.c_cen, self.radius = (c_cen:=np.average(cs, axis=0)), max(np.linalg.norm(cs-c_cen, axis=1))
		self.area = abs(np.linalg.norm(np.cross(cs[1]-cs[0], cs[2]-cs[0]))) / 2
		self.d_nd_to_angles = {nodes[i]: angle_between_3coors(nodes[i].coor, nodes[j].coor, nodes[k].coor)
					   for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1))}
		self.ns_edge, self.normal = getNormals_OutofTriangle(cs, normalize=True)
		# Create edges or use existing edges. Those edges will record this triangle face.
		for i in (0, 1, 2):
			(n1, n2), n3 = (nodes[j] for j in (0, 1, 2) if j != i), nodes[i]
			if edge := n1.sharesEdgewith(n2): self.edges.add(edge)
			else:
				self.edges.add(edge := Edge(domain, n1, n2))
				if is2D: domain.edges_active.append(edge)
			edge.normal = self.ns_edge[i]
			edge.faces.add(self)
		if is2D:
			for edge in self.edges:
				edge.numFaces += 1
				if edge.numFaces >= edge.facesMax and edge in domain.edges_active:
					domain.edges_active.remove(edge)  # print("Remove active edge from meshing", edge, "after removal, num of active edges:", len(domain.edges_active))
			#print("\tAfter removal, edges_active:", domain.edges_active)
		for nd in nodes: nd.faces.add(self)
		
	def __repr__(self):
		n1, n2, n3 = self.nodes
		i, j, k = sorted([n1.i, n2.i, n3.i])
		return "F%d-%d-%d" % (i, j, k)

	def fs_connectedbyEdge(self):
		return {f for eg in self.edges for f in eg.faces if f is not self}
	
	def sharesEdgewith(self, face):
		if eg_common := self.edges.intersection(face.edges): return next(iter(eg_common))
		else: return None

class Ele_Tetrahedron:
	def __init__(self, domain, nodes):
		self.nodes, self.faces = set(nodes), set()
		if len(nodes) != 4: print("Num nodes wrong when gen triangle", nodes, self.nodes); raise Exception
		domain.tups_eleNodes.append(self.nodes)
		self.coors = cs = npA([nd.coor for nd in nodes])
		#print("Creating tetra from nodes", nodes, "\n", cs)
		self.c_cen, self.radius = (c_cen := np.average(self.coors, axis=0)), max(np.linalg.norm(cs - c_cen, axis=1))
		self.volume = volume_between_3vecs(*(cs[1:] - cs[0]))
		normals = getNormals_OutofTetra(cs, normalize=True)
		# Create edges or use existing edges. Those edges will record this triangle face.
		for i in (0, 1, 2, 3):
			(n1, n2, n3), n4 = (nodes[j] for j in (0, 1, 2, 3) if j != i), nodes[i]
			if faces := n1.faces.intersection(n2.faces).intersection(n3.faces):
				self.faces.add(face := next(iter(faces)))
				#print("Using a face already existing", faces)
			else:
				self.faces.add(face := Ele_Triangle(domain, [n1, n2, n3], is2D=False))
				print("\tAdd active face", face)
				domain.elements_2D_active.append(face)
				domain.elements_2D.append(face)
			face.normal = normals[i]
			#face.normal = getNormal_arr4Coor_PerptoC0(npA([n4.coor, n1.coor, n2.coor, n3.coor]))
			face.volumes.add(self)
		for face in self.faces:
			face.numVolumes += 1
			# print("Check face {} and its numVolumes {}/{}".format(face, face.numVolumes, face.volumesMax))
			if face.numVolumes >= face.volumesMax and face in domain.elements_2D_active:
				print("\t\tRemove active face", face)
				domain.elements_2D_active.remove(face)
		print("After check, faces_active:", len(domain.elements_2D_active), len(domain.elements_2D))
		for nd in nodes: nd.volumes.add(self)
	
	def __repr__(self):
		n1, n2, n3, n4 = self.nodes
		i, j, k, l = sorted([n1.i, n2.i, n3.i, n4.i])
		return "V%d-%d-%d-%d" % (i, j, k, l)


# 如果希望用提前生成的节点的方法来确定元素，则应该尽量先从边界下手，生成所有边界的edge。这些edge只能包含1个2个面元素
# 然后生成2维面划分过程中，每个已有的edge都寻找一个离自己中点最近，且不会产生与其他已有元素相切割的节点。同时需要一个方法来确定节点不可以与生成方向有冲突。
class Domain_2D:
	def __init__(self):
		self.nodes, self.elements_2D, self.edges, self.edges_active = [], [], [], []
		self.arr_coors = None
		self.tups_eleNodes = []
		self.patches = []
		
	def prepare4Mesh(self, edges, nodes):
		for edge in edges: edge.facesMax, edge.faces = 1, set()
		self.edges, self.edges_active, self.nodes = edges, edges[:], nodes
		for i, nd in enumerate(self.nodes): nd.i = i
		self.arr_coors = npA([nd.coor for nd in self.nodes])
		print("\n\n\n\nBefore starting 2D meshing of domain. Edges must all be boundaries. Total faces: {}, active faces {}".format(len(self.edges), len(self.edges_active)))
		print("All starting edges must have 1 facesMax:", all(e.facesMax == 1 for e in self.edges_active), all(e.facesMax == 1 for e in self.edges))
		print("All nodes must correctly know the edges they have:", [(nd.i, len(nd.edges)) for nd in nodes])
		print("Nodes: ", len(nodes))
		
	def getEdges_cutby_face(self, coors3):
		c_new_cen = np.average(coors3, axis=0)#.round(DIGIT_ACCU)
		r = max(np.linalg.norm(coors3 - c_new_cen, axis=1))
		return [eg for eg in self.edges if twoCoorsClose(c_new_cen, eg.c_cen, r+eg.radius) and check_segmentMeets_3CoorFace(*eg.coors, coors3)[0]]
		
	def face_cuts_anyEdge(self, coors3):
		c_new_cen = np.average(coors3, axis=0)#.round(DIGIT_ACCU)
		r = max(np.linalg.norm(coors3 - c_new_cen, axis=1))
		return any(twoCoorsClose(c_new_cen, eg.c_cen, r + eg.radius) and check_segmentMeets_3CoorFace(*eg.coors, coors3)[0] for eg in self.edges)
	
	def face_cuts_anyElement(self, coors3):
		c_new_cen = np.average(coors3, axis=0)#.round(DIGIT_ACCU)
		r = max(np.linalg.norm(coors3 - c_new_cen, axis=1))
		return any(twoCoorsClose(c_new_cen, ele.c_cen, r+ele.radius) and check_2Triangles_CutEachOther(coors3, ele.coors) for ele in self.elements_2D)
	
	def genTriangleEle_Wavefront(self, f_height, maxIter=10):
		num = 0
		check_2SegmentsMeet = check_2SegmentsIntersect_within_3D if len(self.nodes[0].coor) == 3 else check_2SegmentsIntersect_within_2D
		while edges := self.edges_active:
			if num >= maxIter: break
			print("*******\nIteration %d\n********" % num)
			print("Num of active edges", len(edges), edges)
			for edge in edges[:]:
				if edge not in self.edges_active: continue
				print("\nTry to grow from active edge", edge)
				n1, n2 = ownNodes = edge.nodes
				c1, c2 = n1.coor, n2.coor
				node_new, nodes, c_cen, normal = None, self.nodes, edge.c_cen, edge.normal
				height = f_height(c_cen) if callable(f_height) else f_height
				#Check if there is a triangle patch that closes void
				node_new = self.search_4edgesClosing(edge)
				#if not node_new: node_new = self.search_smallAnglebetween2Planes(edge)
				# If there is existing node that is very close and forms a large angle from the edge and won't cause intersection, then use it
				if not node_new: node_new = self.search_nearbyNodes(edge, height)
				#Generate a new coor. If it's lying outside domain or in existing element, then skip this edge and leave it to next iteration
				if not node_new:
					coors_triang_new = npA([c1, c2, c_new:= (c_cen + normal * height)])
					print("Potential new location from edge {} is:".format(edge), c_new, "\n", (edge.coors,))
					#欲生成新三角形，需要先检测其是否与已有的edge相重叠（自己的底边会被认为不发生重叠）。如没有，则可以直接生成；如果有，则找到所有有重叠的edge，然后找这些edge中最靠近底边的节点。
						#尝试用靠近底边的节点生成三角形，然后看其是否会产生与其他edge的重叠。如果有，则尝试距离稍远的节点。
					if edgesCut := self.getEdges_cutby_face(coors_triang_new):
						nds_fromEdgesCut = list({nd for ele in edgesCut for nd in ele.nodes if nd} - ownNodes)
						#print("New tri hits existing edges:", nds_fromEdgesCut)
						arr_d_toEdge = normal.dot((npA([nd.coor for nd in nds_fromEdgesCut]) - c_cen).T).round(DIGIT_ACCU)
						for i in arr_d_toEdge.argsort():
							if arr_d_toEdge[i] <= 0: continue
							nd0 = nds_fromEdgesCut[i]
							if not self.face_cuts_anyEdge(npA([c1, c2, nd0.coor])):
								print("Pick a node within the potential new triangle to form", nd0)
								node_new = nd0; break
					else:
						node_new = Node(self, len(nodes), c_new)
						print("Use the new node")
				if node_new:
					print("Generate a triangle", edge, node_new)
					self.elements_2D.append(Ele_Triangle(self, [node_new, n1, n2]))
				else: print("Failed to grow edge:", edge, edge.coors)
			num += 1
		
		print("Number of elements_2D created:", len(self.elements_2D), "Total area:", round(sum(ele.area for ele in self.elements_2D), DIGIT_ACCU))
		print("Nodes in domain: {}. Edges in domain: {}".format(len(self.nodes), len(self.edges)))
		print("Total number of faces", len(self.elements_2D))
		print("Boundary faces", sum(len(face.volumes) == 1 for face in self.elements_2D))
	
	def search_4edgesClosing(self, edge):
		nodes, normal, ownNodes, c_cen = self.nodes, edge.normal, edge.nodes, edge.c_cen
		nds_nextto_1, nds_nextto_2 = [nd.nds_connectedbyEdge() for nd in ownNodes]
		if nds_cmn := [nd for nd in nds_nextto_1.intersection(nds_nextto_2) 
					   if nd in nodes and {nd}|ownNodes not in self.tups_eleNodes]:
			arr_d_toEdge = normal.dot(npA([nd.coor - c_cen for nd in nds_cmn]).T)
			for i in arr_d_toEdge.argsort(): # There can be multiple nodes that can connect to both ends of edge. Pick the closest one in right direction
				if arr_d_toEdge[i] <= 0: continue
				else:
					print("Use existing 3 nodes for triangle. Closing:", edge, nds_cmn[i])
					return nds_cmn[i]
	
	def search_smallAnglebetween2Planes(self, edge):
		c_cen, normal, ownNodes = edge.c_cen, edge.normal, edge.nodes
		edges = self.edges
		if egs := [eg for nd in ownNodes for eg in nd.edges
				   if eg is not edge and eg in edges and normal.dot(eg.c_cen - c_cen).round(DIGIT_ACCU) > 0 and normal.dot(eg.normal) < -0.1]:
			nds = [next(iter(eg.nodes - ownNodes)) for eg in egs]
			arr_d_toedge = npA([normal.dot(nd.coor - c_cen) for nd in nds])
			for i in arr_d_toedge.argsort():
				#print("Check if can gen using a two edges", edge, nds[i])
				if not self.face_cuts_anyEdge(npA([*edge.coors, (nd := nds[i]).coor])):
					print("Connect to another edge's dot", edge, nd)
					return nd
	
	def search_nearbyNodes(self, edge, height):
		nodes, normal, c_cen, ownNodes = self.nodes, edge.normal, edge.c_cen, edge.nodes
		arr_d = np.linalg.norm(arr_vecs := npA([nd.coor for nd in nodes]) - c_cen, axis=1)
		for i in arr_d.argsort():
			if arr_d[i] > 1.3 * height: break
			elif (nd := nodes[i]) in ownNodes: continue
			elif normal.dot(arr_vecs[i]) > np.linalg.norm(arr_vecs[i]) / 4:
				if not self.face_cuts_anyEdge(npA([*edge.coors, nd.coor])):
					print("Connect to a nearby node", nd)
					return nd
					
	def removeaNode(self, node, nds_outer):
		nds012 = nds230 = []
		if len(nds_outer) == 4:
			cs, vecs, normals = getVecs_ifCanFormConvexPolyPlane(npA([nd.coor for nd in list(nds_outer)]))
			if np.linalg.norm(cs[2]-cs[0]) < np.linalg.norm(cs[3]-cs[1]): cs1, cs2 = cs[:3], cs[[2, 3, 0]]
			else: cs1, cs2 = cs[[0, 1, 3]], cs[1:]
			nds012, nds230 = [nd for nd in nds_outer if not (cs1-nd.coor).any(axis=1).all()], [nd for nd in nds_outer if not (cs2-nd.coor).any(axis=1).all()]
		for f in node.faces:
			self.elements_2D.remove(f)
			self.tups_eleNodes.remove(f.nodes)
		for eg in node.edges:
			self.edges.remove(eg)
			for nd in nds_outer:
				if eg in nd.edges: nd.edges.remove(eg)
		for nd in nds_outer:
			for f in node.faces:
				if f in nd.faces: nd.faces.remove(f)
		self.nodes.remove(node)
		if len(nds_outer) == 3: self.elements_2D.append(Ele_Triangle(self, list(nds_outer)))
		if nds012: #Split the quadrilateral into two, using the shorter diagonal
			self.elements_2D.append(Ele_Triangle(self, nds012))
			self.elements_2D.append(Ele_Triangle(self, nds230))
			
	def dissolveSmallTriangles(self, height):
		area_min = height ** 2 / 2
		for ele in self.elements_2D:
			if ele.area < area_min:
				for nd in (nds:=ele.nodes):
					if len(nd.edges) in (3, 4) and not any(eg.facesMax == 1 for eg in nd.edges) and ele.d_nd_to_angles[nd] > 110 \
							and all(f.nodes in self.tups_eleNodes for f in nd.faces):
						print("Found a node to remove", len(nd.edges), nd, nd.nds_connectedbyEdge())
						self.removeaNode(nd, nd.nds_connectedbyEdge())
						break
						
	def plotAllEles(self, fig, ax, showLabel=False, size=1):
		dim = len(self.nodes[0].coor)
		fig.canvas.mpl_connect("button_press_event", lambda event: self.mouseClicked(ax, event))
		for i, ele in enumerate(self.elements_2D):
			if dim == 2: self.patches.append(ax.fill(*ele.coors.T, alpha=0.2)[0])
			else: self.patches.append(ax_plotPolygons(ax, [ele.coors], alpha=0.2))
			ax.text(*np.average(ele.coors, axis=0), i, size=size,
					horizontalalignment="center", verticalalignment="center")
	
	def mouseClicked(self, ax, event):
		if event.inaxes == ax and (patch := next((patch for patch in self.patches if patch.contains(event)[0]), None)):
			ele = self.elements_2D[(i:=self.patches.index(patch))]
			print("Triangle:", i, ele)
			
	def plotAllEdges(self, ax, lw):
		for edge in self.edges: ax.plot(*edge.coors.T, lw=lw)
	
	def plotAllEdges_Active(self, ax, lw):
		for edge in self.edges_active: ax.plot(*edge.coors.T, lw=lw)
		
	def plotAllNodes(self, ax, *args, **kwargs):
		ax.scatter(*npA([nd.coor for nd in self.nodes]).T, *args, **kwargs)
		
	def textAllNodes(self, ax, *args, **kwargs):
		for nd in self.nodes: ax.text(*nd.coor, nd.i, *args, **kwargs)



class Domain_3D:
	def __init__(self):
		self.nodes, self.edges, self.tups_eleNodes = [], [], []
		self.arr_coors = None
		self.elements_2D, self.elements_2D_active, self.elements_3D = [], [], []
		self.domains_2D = []
		
		self.ithElement, self.buttons = 0, [] #For cycling through individual tetrahedrons
		self.patches_activeFaces, self.texts_activeFaces = [], []
		self.scatter_Nodes1, self.texts_Nodes1 = None, []
		self.patches_elements, self.texts_elements = [], []
		self.scatter_Nodes2, self.texts_Nodes2 = None, []
		#self.scatter_nodes
		
	def prepare4Mesh(self):
		self.nodes, self.elements_2D = list(set(self.nodes)), list(set(self.elements_2D))
		self.edges = list(set(self.edges))
		for i, nd in enumerate(self.nodes): nd.i = i
		#if not isinstance(faces, list): faces = list(faces)
		for f in self.elements_2D: f.volumesMax = 1
		self.elements_2D_active = self.elements_2D[:]
		self.nodes = sorted(set(self.nodes), key=lambda nd: nd.i)
		self.arr_coors = arr_coors = npA([nd.coor for nd in self.nodes])
		print("\n\n\nBefore starting 3D meshing of domain. Faces must all be boundaries. Total faces: {}, active faces {}".format(len(self.elements_2D), len(self.elements_2D_active)))
		print("All starting faces must have 1 volumesMax:", all(f.volumesMax == 1 for f in self.elements_2D_active), all(
			f.volumesMax == 1 for f in self.elements_2D))
		print("All nodes must correctly know the faces they have:", [(nd.i, len(nd.faces)) for nd in self.nodes])
		print("Nodes: ", len(self.nodes))
		print("Active faces:", self.elements_2D_active)
	
	def tetra_cuts_anyFace(self, coors4, excludeF=None):
		c_new_cen = np.average(coors4, axis=0)
		r = max(np.linalg.norm(coors4 - c_new_cen, axis=1))
		ns_tetraFace = getNormals_OutofTetra(coors4)
		return any(f is not excludeF and twoCoorsClose(c_new_cen, f.c_cen, r + f.radius) and \
					verify_Triangle_cuts_Tetra(f.coors, coors4, f.ns_edge, f.normal, ns_tetraFace) for f in self.elements_2D)
					
	def getFaces_cutby_tetra(self, coors4, excludeF=None):
		c_new_cen = np.average(coors4, axis=0)
		r = max(np.linalg.norm(coors4 - c_new_cen, axis=1))
		ns_tetraFace = getNormals_OutofTetra(coors4)
		return [f for f in self.elements_2D if f is not excludeF and twoCoorsClose(c_new_cen, f.c_cen, r + f.radius) and
				verify_Triangle_cuts_Tetra(f.coors, coors4, f.ns_edge, f.normal, ns_tetraFace)]
				
	def genTetrahedronEle_Wavefront(self, height, maxIter=10, maxNEle=0,
									reportSummary=True, stopat1stFailure=True):
		numIter = numEle = 0
		stop = False
		while faces := self.elements_2D_active:
			if numIter >= maxIter: break
			print("*******\nIteration %d\n********" % numIter)
			print("Num of active faces", len(faces), len(self.elements_2D_active))
			for face in faces[:]:
				if face not in self.elements_2D_active: continue
				print("\nTry to grow from active face", face)
				if face.numVolumes >= face.volumesMax: raise Exception
				n1, n2, n3 = ownNodes = face.nodes
				c1, c2, c3 = n1.coor, n2.coor, n3.coor
				node_new, nodes, c_cen, normal = None, self.nodes, face.c_cen, face.normal
				# Check if there is a tetrahedron patch that closes void
				node_new = self.search_4facesClosing(face)
				#Check if this face has formed a 2-face closure
				if not node_new: node_new = self.search_smallAnglebetween2Planes(face)
				# If there is existing node that is very close and forms a large angle from the edge and won't cause intersection, then use it
				if not node_new: node_new = self.search_nearbyNodes(face, height)
				
				if not node_new:
					coors_tetra_new = npA([c1, c2, c3, c_new:= (c_cen + normal * height)])
					if facesCut := self.getFaces_cutby_tetra(coors_tetra_new, excludeF=face):
						nds_fromFacesCut = list({nd for ele in facesCut for nd in ele.nodes}-ownNodes)
						print("New tetra would collide with", [f for f in facesCut])
						print("Nodes of those cut faces", nds_fromFacesCut)
						print("Potential new tetra:\n", (coors_tetra_new,))
						arr_d_toFace = normal.dot((npA([nd.coor for nd in nds_fromFacesCut]) - c_cen).T).round(DIGIT_ACCU)
						for i in arr_d_toFace.argsort():
							if arr_d_toFace[i] <= 0: continue
							nd0 = nds_fromFacesCut[i]
							if not self.tetra_cuts_anyFace(npA([c1, c2, c3, nd0.coor]), excludeF=face):
								print("Pick a node within the potential new tetra to form", nd0)
								node_new = nd0; break
					else:
						node_new = Node(self, len(nodes), c_new)
						print("Use the new node", node_new)
				if node_new:
					numEle += 1
					self.elements_3D.append(ele:=Ele_Tetrahedron(self, [node_new, n1, n2, n3]))
					print("Generate a tetrahedron", face, ele, "New ele volume:", ele.volume)
					for f in ele.faces:
						if f.numVolumes < f.volumesMax:
							print("Attempt to span generated", f)
							if node_new := self.search_4facesClosing(f):
								print("-----\n   2ndary gen. Closing", f, node_new)
								self.elements_3D.append(ele := Ele_Tetrahedron(self, [node_new, *f.nodes]))
							elif node_new := self.search_smallAnglebetween2Planes(f):
								print("-----\n2ndary gen. Small angle", f, node_new)
								self.elements_3D.append(ele := Ele_Tetrahedron(self, [node_new, *f.nodes]))
							elif node_new := self.search_nearbyNodes(f, height):
								print("-----\n2ndary gen. Close node", f, node_new)
								self.elements_3D.append(ele := Ele_Tetrahedron(self, [node_new, *f.nodes]))
					if numEle >= maxNEle > 0: stop = True; break
				else:
					print("Failed to gen a tetra from face", face)
					if stopat1stFailure: stop = True; break
			if stop: break
			numIter += 1
			
		if reportSummary:
			print("Number of elements_3D created:", len(self.elements_3D), "Total volume:", round(sum(ele.volume for ele in self.elements_3D), DIGIT_ACCU))
			print("Nodes in domain: {}. Edges in domain: {}".format(len(self.nodes), len(self.edges)))
			print("Total number of faces", len(self.elements_2D))
			print("Boundary faces", sum(len(face.volumes) == 1 for face in self.elements_2D))
			#print("Active faces left:", self.elements_2D_active)
		
	def search_4facesClosing(self, face):
		nodes, normal, ownNodes, c_cen = self.nodes, face.normal, face.nodes, face.c_cen
		nds_nextto_1, nds_nextto_2, nds_nextto_3 = [nd.nds_connectedbyEdge() for nd in ownNodes]
		if nds_cmn := [nd for nd in nds_nextto_1.intersection(nds_nextto_2).intersection(nds_nextto_3) 
					   if nd in nodes and {nd}|ownNodes not in self.tups_eleNodes]:
			arr_d_toFace = normal.dot(npA([nd.coor - c_cen for nd in nds_cmn]).T)
			if np.count_nonzero(arr_d_toFace > 0) > 1:
				print("more than 1 possible closing")
				print([(nd, d) for d, nd in zip(arr_d_toFace, nds_cmn)])
			for i in arr_d_toFace.argsort():# There can be multiple nodes that can connect to both ends of edge. Pick the closest one in right direction
				if arr_d_toFace[i] <= 0: continue
				elif not self.tetra_cuts_anyFace(npA([*face.coors, nds_cmn[i].coor]), excludeF=face):
					print("Use existing 4 nodes for tetra. Closing:", face, nds_cmn[i])
					return nds_cmn[i]
					
	def search_smallAnglebetween2Planes(self, face):
		c_cen, normal, ownNodes = face.c_cen, face.normal, face.nodes
		if fs := [f for f in list(face.fs_connectedbyEdge())
				  if normal.dot(f.c_cen - c_cen).round(DIGIT_ACCU) > 0 and normal.dot(f.normal) < 0.8]:
			nds = [next(iter(f.nodes - ownNodes)) for f in fs]
			arr_d_toFace = npA([normal.dot(nd.coor - c_cen) for nd in nds])
			for i in arr_d_toFace.argsort():
				print("Check if can gen using a two plane", face, nds[i])
				if not self.tetra_cuts_anyFace(npA([*face.coors, (nd := nds[i]).coor]), excludeF=face):
					print("Connect to another face's dot", face, nd)
					return nd
		
	def search_nearbyNodes(self, face, height):
		nodes, normal, c_cen, ownNodes = self.nodes, face.normal, face.c_cen, face.nodes
		arr_d = np.linalg.norm(arr_vecs := npA([nd.coor for nd in nodes]) - c_cen, axis=1)
		for i in arr_d.argsort():
			if arr_d[i] > 1.6 * height: break
			elif (nd := nodes[i]) in ownNodes: continue
			elif normal.dot(arr_vecs[i]) > np.linalg.norm(arr_vecs[i]) / 4:
				if not self.tetra_cuts_anyFace(npA([*face.coors, nd.coor]), excludeF=face):
					print("Connect to a nearby node", face, nd)
					return nd
					
	"""Visualize meshing"""
	def plotAllEdges(self, ax, lw):
		for edge in self.edges: ax.plot(*edge.coors.T, lw=lw)
	
	def plotEdgesofActiveFaces(self, ax, lw):
		edges = {eg for f in self.elements_2D_active for eg in f.edges}
		for eg in edges:
			print("\t", eg, len(fs:=[f for f in eg.faces if f in self.elements_2D_active]), fs)
		for eg in edges: ax.plot(*eg.coors.T, lw=lw)
		
	def plotAllActiveFaces(self, fig, ax, alpha=0.15, size=8):
		self.scatter_Nodes1 = ax.scatter(*npA([nd.coor for nd in self.nodes]).T)
		for nd in self.nodes:
			self.texts_Nodes1.append(txt:=ax.text(*nd.coor, nd.i))
			txt.node = nd
		fig.canvas.mpl_connect("button_press_event", lambda event: self.mouseClicked_onFaceActive(ax, event))
		for i, ele in enumerate(self.elements_2D_active):
			self.plotActiveFace(ax, i, ele)
			#ax.quiver(*ele.c_cen, *ele.normal)
			
	def plotActiveFace(self, ax, i, ele, alpha=0.15, size=8):
		cm = plt.cm.get_cmap('rainbow')
		patch = ax_plotPolygons(ax, [ele.coors], alpha=alpha, color=cm(np.random.uniform()))
		txt = ax.text(*np.average(ele.coors, axis=0), i, size=size, horizontalalignment="center", verticalalignment="center")
		patch.element = txt.element = ele
		self.patches_activeFaces.append(patch)
		self.texts_activeFaces.append(txt)
	
	def update_activeFaces(self, ax):
		self.scatter_Nodes1._offsets3d = npA([nd.coor for nd in self.nodes]).T
		for patch, text in zip(self.patches_activeFaces[:], self.texts_activeFaces[:]):
			if patch.element not in self.elements_2D_active:
				self.patches_activeFaces.remove(patch)
				self.texts_activeFaces.remove(text)
				patch.remove()
				text.remove()
			text.set_text(text.element.numVolumes)
		for nd in self.nodes:
			if not any(txt.node is nd for txt in self.texts_Nodes1):
				self.texts_Nodes1.append(txt := ax.text(*nd.coor, nd.i))
				txt.node = nd
		for i, ele in enumerate(self.elements_2D_active):
			if not any(patch.element is ele for patch in self.patches_activeFaces):
				self.plotActiveFace(ax, ele.numVolumes, ele)
				
	def mouseClicked_onFaceActive(self, ax, event):
		if event.inaxes == ax and (patch := next((patch for patch in self.patches_activeFaces if patch.contains(event)[0]), None)):
			ele = patch.element
			print("Triangle:", ele, (ele.coors,))
			
	def plotAllElements(self, ax, size=8, explosionCenter=None, explosionFactor=0.2):
		self.scatter_Nodes2 = ax.scatter(*npA([nd.coor for nd in self.nodes]).T)
		for nd in self.nodes:
			self.texts_Nodes2.append(txt:=ax.text(*nd.coor, nd.i))
			txt.node = nd
		for i, ele in enumerate(self.elements_3D):
			self.plotElement(ax, i, ele, explosionCenter=explosionCenter, explosionFactor=explosionFactor)
			
	def plotElement(self, ax, i, ele, explosionCenter=None, explosionFactor=0.2, alpha=0.15, size=8):
		cm = plt.cm.get_cmap('rainbow')
		offset = np.zeros((3, )) if explosionCenter is None else explosionFactor * (ele.c_cen-explosionCenter)
		patch = ax_plotPolygons(ax, [offset+f.coors for f in ele.faces], alpha=0.15, color="green")
		txt = ax.text(*ele.c_cen, i, size=size, color="red",
					  verticalalignment="center", horizontalalignment="center")
		self.patches_elements.append(patch)
		self.texts_elements.append(txt)
		patch.element = txt.element = ele
		
	def update_elements(self, ax, explosionCenter=None, explosionFactor=0.2):
		self.scatter_Nodes2._offsets3d = npA([nd.coor for nd in self.nodes]).T
		for patch, text in zip(self.patches_elements[:], self.texts_elements[:]):
			if patch.element not in self.elements_3D:
				self.patches_elements.remove(patch)
				self.texts_elements.remove(text)
				patch.remove()
				text.remove()
		for nd in self.nodes:
			if not any(txt.node is nd for txt in self.texts_Nodes2):
				self.texts_Nodes2.append(txt := ax.text(*nd.coor, nd.i))
				txt.node = nd
		for i, ele in enumerate(self.elements_3D):
			if not any(patch.element is ele for patch in self.patches_elements):
				self.plotElement(ax, i, ele, explosionCenter=explosionCenter, explosionFactor=explosionFactor)
				
	def prepare_to_viewIndividual(self, ax):
		polyCol = ax_plotPolygons(ax, [], alpha=0.4, color="green")
		(b_last := Button(ax=plt.axes([0.02, 0.2, 0.12, 0.04]), label="Last", hovercolor="0.95")).on_clicked(lambda event: self.newEletoView(ax, polyCol, -1))
		(b_next := Button(ax=plt.axes([0.02, 0.25, 0.12, 0.04]), label="Next", hovercolor="0.95")).on_clicked(lambda event: self.newEletoView(ax, polyCol, 1))
		self.buttons += [b_last, b_next]
		
	def newEletoView(self, ax, polyCol, incre):
		self.ithElement += incre
		ele = self.elements_3D[self.ithElement % len(self.elements_3D)]
		# print("View element", ele, ele.faces)
		polyCol.set_verts([f.coors for f in ele.faces])
	
	def saveMesh(self, filename="Mesh.txt"):
		with open(filename, 'w') as fout:
			fout.write("Nodes:\n")
			for nd in self.nodes: fout.write("{},{}\n".format(nd.i, nd.coor))
			fout.write("Volumes:\n")
			for volume in self.elements_3D:
				n1, n2, n3, n4 = volume.nodes
				fout.write("{},{},{},{}\n".format(n1.i, n2.i, n3.i, n4.i))