import numpy as np

from GeometryFuncs import *
import pickle
from PlotMethods import *
from CollisionDetection import *

class Node:
	def __init__(self, domain, i, coor):
		self.i, self.coor = i, coor
		self.edges, self.faces, self.volumes = set(), set(), set()
		domain.nodes.append(self)
		
	def __repr__(self): return "N%d" % self.i
	
	def edgeSharedwith(self, node):
		return next(iter(self.edges & node.edges), None)
		
	@property
	def nds_connectedbyEdge(self):
		return {nd for eg in self.edges for nd in eg.nodes if nd is not self}
	
class Edge:
	def __init__(self, domain, *nodes):
		self.nodes = nd1, nd2 = set(nodes)
		self.coors = npA([nd1.coor, nd2.coor])
		self.c_cen, self.radius = np.average(self.coors, axis=0), npNorm(nd1.coor-nd2.coor) / 2
		for nd in nodes: nd.edges.add(self)
		self.faces, self.facesMax = set(), 2
		self.normal = None
		domain.edges.append(self)
		self.domains = {domain}
	
	def __repr__(self):
		n1, n2 = self.nodes
		if n1.i < n2.i: return "E%d-%d" % (n1.i, n2.i)
		else: return "E%d-%d" % (n2.i, n1.i)
	
	@property
	def getVolumes(self): return {v for f in self.faces for v in f.volumes}
	
	@property
	def isActive(self): return len(self.faces) < self.facesMax
	
	def angle_withEdge_deg(self, edge):
		dotProd = ((normalSelf := self.normal) @ edge.normal).round(3)
		angle = 180 - np.degrees(np.arccos(dotProd))
		if (normalSelf @ (edge.c_cen - self.c_cen)).round(3) < 0:
			return 360 - angle
		return angle
		
class Ele_Triangle:
	def __init__(self, domain, nodes, calcFEAMatrices=False):
		self.nodes, self.edges, self.volumes = set(nodes), set(), set()
		self.volumesMax = 2
		self.coors = cs = npA([nd.coor for nd in nodes])
		self.c_cen, self.radius = (c_cen:=np.average(cs, axis=0)), max(npNorm(cs-c_cen, axis=1))
		self.area = abs(npNorm(np.cross(*(cs[1:]-cs[0])))) / 2
		self.d_nd_to_angles = {nodes[i]: angle_btw_2vecs(cs[j]-cs[i], cs[k]-cs[i])
					   				for i, j, k in ([0, 1, 2], [1, 2, 0], [2, 0, 1])}
		self.ns_edge, self.normal = getNormals_OutofTriangle(cs)
		# Create edges or use existing edges. Those edges will record this triangle face.
		for i in (0, 1, 2):
			(n1, n2), n3 = (nodes[j] for j in (0, 1, 2) if j != i), nodes[i]
			if edge := n1.edgeSharedwith(n2): self.edges.add(edge)
			else: self.edges.add(edge := Edge(domain, n1, n2))
			edge.normal = self.ns_edge[i]
			edge.faces.add(self)
		for nd in nodes: nd.faces.add(self) #Used to count how many active faces a node has.
		
		self.dict_node_to_index = {nd: i for i, nd in enumerate(nodes)}
		if calcFEAMatrices: self.initFEAMatrices()
		else:
			self.M_xy_2_xieta = self.M_inv = self.absdet_M_inv = None
			self.arr_gradVi_gradVj_IntoverArea = self.arr_Vi_Vj_IntoverArea = None
			self.dict_edgeIdx_to_arr_Vi_gradVj_IntoverEdge = self.dict_edgeIdx_to_arr_Vi_Vj_IntoverEdge = None
			
	def initFEAMatrices(self):
		# Prepare for integration using linear functions
		self.M_inv = M_inv = (self.coors[1:] - self.coors[0]).T
		M, absdet_M_inv = self.M_xy_2_xieta, self.absdet_M_inv = np.linalg.inv(M_inv), abs(np.linalg.det(M_inv))
		# For triangle element with linear functions, a tri can be linearly transformed into a right isosceles triangle.
		# For right isosceles triangle in xi,eta, the 3 nodes define linear functions:
		#	u0: 1-xi-eta (1 at origin), u1: xi (1 at (1, 0)), u2: eta (1 at (0, 1))
		# The tri vert in real xy coor -> right isosceles tri is by c_xi_eta = M c_x_y
		# grad_Vi_xyz = M.T @ grad_Ui_xieta
			# grad{v0}_xy = (-a-c, c-d); grad{v1}_xy = (a, b); grad{v2}_xy = (c, d)
		grads_v012_xy = npA([-M.sum(axis=0), *M])
		self.arr_gradVi_gradVj_IntoverArea = 0.5 * absdet_M_inv * grads_v012_xy.dot(grads_v012_xy.T)
		self.arr_Vi_Vj_IntoverArea = absdet_M_inv * (np.ones((3, 3)) + np.diag([1, 1, 1])) / 24
		# The integrals of u0&u1&u2 products over triangle or along 01&02&12 are easy to calc
		arr_Vi_Vj_01edge = npA([[2, 1, 0], [1, 2, 0], [0]*3]) / 6 * (coeff_01 := npNorm(M_inv[:, 0]))
		arr_Vi_Vj_02edge = npA([[2, 0, 1], [0]*3, [1, 0, 2]]) / 6 * (coeff_02 := npNorm(M_inv[:, 1]))
		arr_Vi_Vj_12edge = npA([[0]*3, [0, 2, 1], [0, 1, 2]]) / 6 * (coeff_12 := npNorm(M_inv[:, 1] - M_inv[:, 0]))
		self.dict_edgeIdx_to_arr_Vi_Vj_IntoverEdge = {(0, 1): arr_Vi_Vj_01edge, (1, 0): arr_Vi_Vj_01edge,
													   (1, 2): arr_Vi_Vj_12edge, (2, 1): arr_Vi_Vj_12edge,
													   (0, 2): arr_Vi_Vj_02edge, (2, 0): arr_Vi_Vj_02edge}
		# The integrals along 01&02&12 are most tricky
		self.dict_edgeIdx_to_arr_Vi_gradVj_IntoverEdge = d = {}
		M_T, grads_u_xieta, normals_xieta = M.T, npA([[-1, -1], [1, 0], [0, 1]]), npA([[0, -1], [-1, 0], [1/np.sqrt(2)]*2])
		grads_v_xy_T, normals_xy_T = M_T @ grads_u_xieta.T, M_inv @ normals_xieta.T
		normals_xy = (normals_xy_T / np.tile(npNorm(normals_xy_T, axis=0), (2, 1))).T
		#M_inv @ normal_xieta --> normal_xy, length_xieta * coeff --> length_xy
		for ind, (tup1, tup2, normal_xy, coeff) in enumerate(zip(((0, 1), (0, 2), (1, 2)), ((1, 0), (2, 0), (2, 1)),
																	normals_xy, (coeff_01, coeff_02, coeff_12))):
			(a := npA([0.5, 0.5, 0.5]))[2-ind] = 0 #along 01 edge, ui integrals are 1/2, 1/2, 0,  etc.
			d[tup1] = d[tup2] = np.outer(a, coeff * normal_xy.dot(grads_v_xy_T))
		
	def __repr__(self):
		n1, n2, n3 = self.nodes
		i, j, k = sorted([n1.i, n2.i, n3.i])
		return "F%d-%d-%d" % (i, j, k)

	@property
	def fs_connectedbyEdge(self):
		return {f for eg in self.edges for f in eg.faces if f is not self}
	
	def edgeSharedwith(self, face): #Doesn't include those that touch this by just 1 node
		return next(iter(self.edges & face.edges), None)
		
	@property
	def isActive(self): return len(self.volumes) < self.volumesMax
	
	def angle_withFace_deg(self, face):
		dotProd = ((normalSelf := self.normal) @ face.normal).round(3)
		angle = 180 - np.degrees(np.arccos(dotProd))
		if (normalSelf @ (face.c_cen - self.c_cen)).round(3) < 0:
			angle = 360 - angle
		return angle

class Ele_Tetrahedron:
	def __init__(self, domain, nodes, calcFEAMatrices=False):
		self.nodes, self.faces = set(nodes), set()
		self.coors = cs = npA([nd.coor for nd in nodes])
		self.c_cen, self.radius = (c_cen := np.average(self.coors, axis=0)), max(npNorm(cs - c_cen, axis=1))
		v01, v02, v03 = cs[1:] - cs[0]
		self.volume = abs(v01.dot(np.cross(v02, v03))) / 6
		normals = getNormals_OutofTetra(cs)
		# Create edges or use existing edges. Those edges will record this triangle face.
		for i in (0, 1, 2, 3):
			(n1, n2, n3), n4 = (nodes[j] for j in (0, 1, 2, 3) if j != i), nodes[i]
			if faces := n1.faces & n2.faces & n3.faces:
				self.faces.add(face := next(iter(faces)))
				#print("\t\tUsing an existing face", face)
			else:
				self.faces.add(face := Ele_Triangle(domain, [n1, n2, n3]))
				#print("\t\tAdd active face", face, face.isActive)
				domain.elements_2D.append(face)
			face.normal = normals[i]
			face.volumes.add(self)
		for nd in nodes: nd.volumes.add(self)
		
		self.dict_node_to_index = {nd: i for i, nd in enumerate(nodes)}
		if calcFEAMatrices: self.initFEAMatrices()
		else:
			self.M_xy_2_xieta = self.M_inv = self.absdet_M_inv = None
			self.arr_gradVi_gradVj_IntoverVolume = self.arr_Vi_Vj_IntoverVolume = None
			self.dict_faceIdx_to_arr_Vi_gradVj_IntoverFace = self.dict_faceIdx_to_arr_Vi_Vj_IntoverFace = None
			
	def __repr__(self):
		n1, n2, n3, n4 = self.nodes
		i, j, k, l = sorted([n1.i, n2.i, n3.i, n4.i])
		return "V%d-%d-%d-%d" % (i, j, k, l)

	@property
	def getEdges(self): return {eg for f in self.faces for eg in f.edges}
	
	def initFEAMatrices(self):
		# Prepare for integration using linear functions
		#Tetra can be transformed into one defined by xi, eta, rou axis
			#In xi,eta,rou the 4 nodes define linear functions:  u0: 1-xi-eta-rou (1 at origin), u1: xi (1 at (1, 0, 0)), u2: eta (1 at (0, 1, 0)), u3: rou (1 at (0, 0, 1))
			#grad_u0 = (-1, -1, -1), grad_U1 = (1, 0, 0), grad_U2 = (0, 1, 0), grad_U3 = (0, 0, 1)
			#The tetra vert in real xy coor -> xi, eta, rou is by c_xi_eta_rou = M c_x_y_z
		self.M_inv = M_inv = (self.coors[1:] - self.coors[0]).T
		M, absdet_M_inv = self.M_xy_2_xieta, self.absdet_M_inv = np.linalg.inv(M_inv), abs(np.linalg.det(M_inv))
		# grad_Vi_xyz = M.T @ grad_Ui_xietarou
		grads_v0123_xyz = npA([-M.sum(axis=0), *M])
		self.arr_gradVi_gradVj_IntoverVolume = absdet_M_inv / 6 * grads_v0123_xyz.dot(grads_v0123_xyz.T)
		self.arr_Vi_Vj_IntoverVolume = absdet_M_inv * (np.ones((4, 4)) + np.diag([1, 1, 1, 1])) / 120
		# The integrals of u0&u1&u2 products over triangle faces are easy to calc
		arr_Vi_Vj_012face = npA([[2, 1, 1, 0], [1, 2, 1, 0], [1, 1, 2, 0], [0]*4]) / 24 * (coeff_012 := npNorm(np.cross(M_inv[:,0], M_inv[:,1])))
		arr_Vi_Vj_013face = npA([[2, 1, 0, 1], [1, 2, 0, 1], [0]*4, [1, 1, 0, 2]]) / 24 * (coeff_013 := npNorm(np.cross(M_inv[:,0], M_inv[:,2])))
		arr_Vi_Vj_023face = npA([[2, 0, 1, 1], [0]*4, [1, 0, 2, 1], [1, 0, 1, 2]]) / 24 * (coeff_023 := npNorm(np.cross(M_inv[:,1], M_inv[:,2])))
		arr_Vi_Vj_123face = npA([[2, 0, 1, 1], [0]*4, [1, 0, 2, 1], [1, 0, 1, 2]]) / 24 * (coeff_123 := npNorm(np.cross(*(M_inv[:,1:].T-M_inv[:,0]))))
		self.dict_faceIdx_to_arr_Vi_Vj_IntoverFace = {t: arr for arr, tup in zip((arr_Vi_Vj_012face, arr_Vi_Vj_013face, arr_Vi_Vj_023face, arr_Vi_Vj_123face),
																					((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
													   				for t in permutations(tup)}
		# The integrals along 012&013&023&123 are most tricky
		self.dict_faceIdx_to_arr_Vi_gradVj_IntoverFace = d = {}
		M_T, grads_u_xieta = M.T, npA([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 012&013&023&123
		normals_xieta = npA([[0, 0, -1], [0, -1, 0], [-1, 0, 0], [1 / np.sqrt(3)] * 3])  # 012&013&023&123
		grads_v_xy_T, normals_xy_T = M_T @ grads_u_xieta.T, M_inv @ normals_xieta.T
		normals_xy = (normals_xy_T / np.tile(npNorm(normals_xy_T, axis=0), (3, 1))).T
		for ind, (tup, normal_xy, coeff) in enumerate(zip(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
														  normals_xy, (coeff_012, coeff_013, coeff_023, coeff_123))):
			(a := npA([1, 1, 1, 1])/6)[3 - ind] = 0  # along 01 edge, ui integrals are 1/2, 1/2, 0,  etc.
			arr = np.outer(a, coeff * normal_xy.dot(grads_v_xy_T))
			for t in permutations(tup): d[t] = arr
			
# 如果希望用提前生成的节点的方法来确定元素，则应该尽量先从边界下手，生成所有边界的edge。这些edge只能包含1个2个面元素
# 然后生成2维面划分过程中，每个已有的edge都寻找一个离自己中点最近，且不会产生与其他已有元素相切割的节点。同时需要一个方法来确定节点不可以与生成方向有冲突。
class Domain_2D:
	def __init__(self):
		self.nodes, self.elements_2D, self.edges = [], [], []
		self.tups_eleNodes = []
		
	def prepare4Mesh(self, edges, nodes):
		for edge in edges: edge.facesMax = 1 #Don't touch edge.faces, edges shared by two domains.
		self.edges, self.nodes = edges, nodes
		for i, nd in enumerate(self.nodes): nd.i = i
		for eg in self.edges:
			eg.domains.add(self)
			eg.facesMax = 1
		print("\n\nBefore starting 2D meshing of domain. Edges must all be boundaries")
		print("Total edges: {}, active edges {}".format(len(self.edges), sum(eg.isActive for eg in self.edges)))
		print("Nodes: ", len(nodes))
		
	def tri_cuts_anySeg(self, coors3, cs_edges=None):
		if cs_edges is None: cs_edges = npA([eg.coors for eg in self.edges])
		return check_Tri_cuts_Segs(coors3, cs_edges)
	
	def getEdges_cutby_tri(self, coors3, cs_edges=None):
		edges = self.edges
		if cs_edges is None: cs_edges = npA([eg.coors for eg in edges])
		return [edges[i] for i in check_Tri_cuts_Segs(coors3, cs_edges, any_not_idx=False)]
		
	def genTriangleEle_Wavefront(self, func_height, maxIter=10):
		numIter = 0
		while (edges_curFront := [eg for eg in self.edges if eg.isActive]) and numIter < maxIter:
			nds_curFront = {nd for eg in edges_curFront for nd in eg.nodes}
			print("*******\nIteration %d\n********" % numIter)
			print("Num of active edges", len(edges_curFront))
			for edge in edges_curFront:
				if not edge.isActive: continue
				print("\nTry to grow from active edge", edge, edge.normal)
				height = func_height(edge.c_cen) if callable(func_height) else func_height
				cs_edges = npA([eg.coors for eg in self.edges])
				if not (node_new := self.try_nearbyNodes(edge, height, cs_edges)):
					node_new = self.try_budEle_Height(edge, height, cs_edges)
				if not node_new:
					print("Failed to gen a tetra from face", edge)
					raise RuntimeError
				else:
					ele = self.genaTriElement([node_new, *edge.nodes])
					print("Genned a tetra", ele, "with", node_new, ". New ele area:", ele.area)
			
			for nd in nds_curFront:
				if sum(eg.isActive for eg in nd.edges) == 2:
					height = func_height(nd.coor) if callable(func_height) else func_height
					print("\nTry to close up nd space", nd, [eg for eg in nd.edges if eg.isActive], height)
					self.try_closeupSpacearound(nd, height)
			numIter += 1
			
		print("Finished meshing 2D domain.\nTotal faces:", len(self.elements_2D),
			  "Total area:", round(sum(ele.area for ele in self.elements_2D), DIGIT_HEIGHT))
		print("Total nodes: {}. Total edges: {}".format(len(self.nodes), len(self.edges)))
		
	def genaTriElement(self, nodes):
		if not isinstance(nodes, list): nodes = list(nodes)
		self.elements_2D.append(ele := Ele_Triangle(self, nodes))
		return ele
	
	def try_closeupSpacearound(self, node, height):
		if egs := [eg for eg in node.edges if eg.isActive and self in eg.domains]:
			while egs:
				nd_other_2 = next(iter((eg2 := egs[1]).nodes - (eg1 := egs[0]).nodes))
				largeAngle = eg1.angle_withEdge_deg(eg2) > 100
				#夹角<100度时，直接尝试连接nd_other1&nd_other2。如果其不与其他edge切割时，直接生成；若有切割，则取这些edge的node中最接近轴线的node
				#夹角>100度时，由eg1的height bud tri。这些tri绝不会与eg2切割。若无切割，则生成；若无切割，则取nodes
				if not (node_new := self.try_budEle_Height(eg1, height, cs_edges=npA([eg.coors for eg in self.edges]),
												preferNode=None if largeAngle else nd_other_2)):
					print("Wrong while closing node", node)
					raise RuntimeError
				ele = self.genaTriElement([node_new, *eg1.nodes])
				print("Genned a tri ele {} while closing {}".format(ele, node))
				egs = [eg2, next(iter(egs))] if (egs := {eg for eg in node.edges if eg.isActive and self in eg.domains} - {eg2}) else []
				
	def try_nearbyNodes(self, edge, height, cs_edges,
						lateral=0.52, vertical_min=0.5, vertical_max=1.2):
		nodes, normal, cs_eg, dim = self.nodes, edge.normal, edge.coors, len(edge.normal)
		vs_cen2Nodes = npA([nd.coor for nd in nodes]) - edge.c_cen  # shape(n, 3)
		heights_fromface = (normal @ vs_cen2Nodes.T)
		horDist_fromCen = npNorm(np.cross(normal, vs_cen2Nodes), axis=1) if dim == 3 else np.abs(np.cross(normal, vs_cen2Nodes))
		dist_fromCen, maxHorDist = npNorm(vs_cen2Nodes, axis=1), lateral * height
		idx_allowed = np.where((vertical_min * height < heights_fromface) & (heights_fromface < vertical_max * height) & (horDist_fromCen < maxHorDist))[0]
		nodes_ds = sorted(((nodes[i], dist_fromCen[i]) for i in idx_allowed), key=lambda tup: tup[1])
		return next((nd for nd, _ in nodes_ds if not self.tri_cuts_anySeg(npA([*cs_eg, nd.coor]), cs_edges=cs_edges)), None)
		
	def try_budEle_Height(self, edge, height, cs_edges, preferNode=None, largerHeight_2Try=1.2):
		nodes, normal, c_cen, ownNodes = self.nodes, edge.normal, edge.c_cen, edge.nodes
		c1, c2 = edge.coors  # (n1, n2) = ownNodes
		if preferNode: cs_tri_new = npA([c1, c2, preferNode.coor])
		else: cs_tri_new = npA([c1, c2, c_cen + largerHeight_2Try * normal * height])
		# 欲生成新三角形，需要先检测其是否与已有的edge相重叠（自己的底边会被认为不发生重叠）。如没有，则可以直接生成
		# 如果有，则找这些有重叠的edge的端点，取其中最靠近轴线的端点，尝试生成三角形，并检测它们是否不与已有的edge相重叠。
		if edgesCut := self.getEdges_cutby_tri(cs_tri_new, cs_edges):
			print("Collided with edges:", edgesCut)
			nds_fromEdgesCut = list({nd for ele in edgesCut for nd in ele.nodes if nd} - ownNodes)
			vs_cen_to_nds = npA([nd.coor for nd in nds_fromEdgesCut]) - c_cen
			cosines_from_normal = (normal.dot(vs_cen_to_nds.T) / npNorm(vs_cen_to_nds, axis=1)).round(DIGIT_HEIGHT)
			nds_cosines = sorted(zip(nds_fromEdgesCut, cosines_from_normal), key=lambda t: t[1], reverse=True)
			return next((nd for nd, cosine in nds_cosines if cosine > 0 and not self.tri_cuts_anySeg(npA([c1, c2, nd.coor]), cs_edges)), None)
		elif preferNode:
			print("Bud with existing node", preferNode)
			return preferNode
		else:
			print("Bud with new node")
			return Node(self, len(nodes), c_cen + normal * height)
	
	"""Dissolve triangles that are too small"""
	def dissolveaNode(self, node, nds_outer):
		print("Dissolving node:", node, node.coor, "nds_outer", nds_outer)
		nds021 = nds023 = []
		faces_2remove = node.faces
		if len(nds_outer) == 4:
			nds_orderd = [nd1 := next(iter(nds_outer))]
			while nd1 := next((nd for nd in nds_outer if not nd.edges.isdisjoint(nd1.edges) and nd not in nds_orderd), None):
				nds_orderd.append(nd1)
			if npNorm(nds_orderd[2].coor - nds_orderd[0].coor) < npNorm(nds_orderd[3].coor - nds_orderd[1].coor):
				idx_021, idx_023 = (0, 2, 1), (0, 2, 3)
			else: idx_021, idx_023 = (1, 3, 0), (1, 3, 2)
			nds021, nds023 = [nds_orderd[i] for i in idx_021], [nds_orderd[i] for i in idx_023]
			print("Will gen two triangles:", nds021, nds023)
		for eg in {eg for f in node.faces for eg in f.edges if eg not in node.edges}:
			eg.faces -= faces_2remove
		for f in node.faces:
			self.elements_2D.remove(f)
			self.tups_eleNodes.remove(f.nodes)
		for eg in node.edges:
			self.edges.remove(eg)
			for nd in nds_outer:
				if eg in nd.edges: nd.edges.remove(eg)
		for nd in nds_outer:
			nd.faces -= faces_2remove
		self.nodes.remove(node)
		if len(nds_outer) == 3: self.elements_2D.append(Ele_Triangle(self, list(nds_outer)))
		if nds021: #Split the quadrilateral into two, using the shorter diagonal
			self.elements_2D.append(Ele_Triangle(self, nds021))
			self.elements_2D.append(Ele_Triangle(self, nds023))
			
	def dissolveSmallTriangles(self, height):
		area_min = height ** 2 / 2
		self.tups_eleNodes = [f.nodes for f in self.elements_2D]
		for ele in self.elements_2D[:]:
			if ele in self.elements_2D and ele.area < area_min:
				for nd in ele.nodes:
					if len(nd.edges) in (3, 4) and not any(eg.facesMax == 1 for eg in nd.edges) and ele.d_nd_to_angles[nd] > 110 \
							and all(f.nodes in self.tups_eleNodes for f in nd.faces):
						print("Found {} with {} edges to remove".format(nd, len(nd.edges)), nd.nds_connectedbyEdge)
						self.dissolveaNode(nd, nd.nds_connectedbyEdge)
						break
		
	"""Given pre-determined nodes, try to form triangular elements among those nodes"""
	def saveMesh(self, filename="2D_Mesh.txt"):
		with open(filename, 'w') as fout:
			fout.write("Nodes:\n")
			for nd in self.nodes:
				nd.coor = nd.coor.round(DIGIT_HEIGHT)
				fout.write(("{}, [{},{}]\n" if len(nd.coor) == 2 else "{}, [{},{},{}]\n").format(nd.i, *nd.coor))
			fout.write("Faces:\n")
			for face in self.elements_2D:
				n1, n2, n3 = face.nodes
				fout.write("{},{},{}\n".format(n1.i, n2.i, n3.i))
				
	def loadMesh(self, filename="2D_Mesh.txt"):
		with open(filename, 'r') as fin:
			i = (lines := fin.readlines()).index("Faces:\n")
			lines_nodes, lines_faces = lines[1:i], lines[i + 1:]
			nodes, elements_2D = self.nodes, self.elements_2D = [], []
			for i, l in enumerate(lines_nodes):
				exec("Node(self, {}, np.array({}))".format(i, l[l.index(',') + 1:].strip("\n")))
			print("Nodes in mesh:\n", nodes)
			for l in lines_faces:
				i0, i1, i2 = l.strip("\n").split(',')
				i0, i1, i2 = int(i0), int(i1), int(i2)
				elements_2D.append(Ele_Triangle(self, [nodes[i0], nodes[i1], nodes[i2]], calcFEAMatrices=True))
			print("Triangles in mesh:\n", elements_2D)
	
	"""Calculate matrices neccesary for FEA."""
	def getArr_scalarProducts_IntoverArea_Linear(self, ViVj_not_gradVigradVj=False):
		arr = np.zeros([n:=len(self.nodes)] * 2)
		for ele in self.elements_2D:
			d, (nd0, nd1, nd2) = ele.dict_node_to_index, ele.nodes
			tups = ((nd0, d[nd0]), (nd1, d[nd1]), (nd2, d[nd2]))
			for (nd1, i1), (nd2, i2) in product(tups, tups):
				if (i:=nd1.i) <= (j:=nd2.i): #only consider diagonal and above
					if ViVj_not_gradVigradVj: arr[i,j] += ele.arr_Vi_Vj_IntoverArea[i1,i2]
					else: arr[i,j] += ele.arr_gradVi_gradVj_IntoverArea[i1,i2]
		for i in range(n):
			for j in range(n):
				if i > j: arr[i,j] = arr[j,i]
		return arr
		
	def getArr_Product_IntoverEdge_Linear(self, VigradVj_dot_normal_not_ViVj=False):
		arr = np.zeros([len(self.nodes)] * 2)
		for f, eg in ((next(iter(edge.faces)), edge) for edge in self.edges if len(edge.numFaces) == 1):
			(nd1, nd2), d = eg.nodes, f.dict_node_to_index
			if VigradVj_dot_normal_not_ViVj: arr_IntoverEdge = f.dict_edgeIdx_to_arr_Vi_gradVj_IntoverEdge[(d[nd1], d[nd2])]
			else: arr_IntoverEdge = f.dict_edgeIdx_to_arr_Vi_Vj_IntoverEdge[(d[nd1], d[nd2])]
			for nd1, nd2 in product(f.nodes, f.nodes): arr[nd1.i, nd2.i] += arr_IntoverEdge[d[nd1], d[nd2]]
		return arr
		
		
class Domain_3D:
	def __init__(self):
		self.nodes, self.edges = [], []
		self.elements_2D, self.elements_3D = [], []
		
	def prepare4Mesh(self):
		self.nodes, self.elements_2D = list(set(self.nodes)), list(set(self.elements_2D))
		self.edges = list(set(self.edges))
		for i, nd in enumerate(self.nodes): nd.i = i
		for f in self.elements_2D:
			if not f.volumes: f.volumesMax = 1
			else: print("There is one face:", f, len(f.volumes), f.volumesMax)
		print("\n\n\nBefore starting 3D meshing. Faces must all be boundaries. Total faces: {}".format(len(self.elements_2D)))
		
	def tetra_cuts_anyTri(self, coors4, cs_faces=None):
		if cs_faces is None: cs_faces = npA([f.coors for f in self.elements_2D])
		return check_Tetra_cuts_Tris(coors4, cs_faces)
		
	def getFaces_cutby_tetra(self, coors4, cs_faces):
		faces = self.elements_2D
		if cs_faces is None: cs_faces = npA([f.coors for f in faces])
		return [faces[i] for i in check_Tetra_cuts_Tris(coors4, cs_faces, any_not_idx=False)]
		
	def genTetrahedronEle_Wavefront(self, func_height, maxIter=10):
		numIter = 0
		while (faces_active := [f for f in self.elements_2D if f.isActive]) and numIter < maxIter:
			nds_curFront = {nd for ele in faces_active for nd in ele.nodes}
			egs_curFront = {eg for ele in faces_active for eg in ele.edges}
			print("*******\n3D Meshing Iteration %d\n********" % numIter)
			#首先只从当前faces生成萌芽bud状的tetra
			faces_active.sort(reverse=True, key=lambda tri: min(tri.d_nd_to_angles.values()))
			for face in faces_active:
				if not face.isActive: continue
				print("\nTry to grow from active face", face, face.normal)
				height = func_height(face.c_cen) if callable(func_height) else func_height
				node_new = self.try_nearbyNodes(face, height, cs_faces := npA([f.coors for f in self.elements_2D]))
				if not node_new: node_new = self.try_budEle_Height(face, height, cs_faces)[0]
				if not node_new:
					print("Failed to gen a tetra from face", face)
					raise RuntimeError
				else:
					ele = self.genaTetraElement([node_new, *face.nodes])
					print("Genned a spiking tetra", ele, "with", node_new, ".  New ele volume:", ele.volume)
			numIter += 1
			
			#Try to close up the space around edges
			for eg in egs_curFront:
				if sum(f.isActive for f in eg.faces) == 2:
					height = func_height(eg.c_cen) if callable(func_height) else func_height
					print("\nTry to close up eg space", eg, [f for f in eg.faces if f.isActive], height)
					self.try_closeupSpacearound_Eg(eg, height)
			print("Remaining cur front edges:", [eg for eg in egs_curFront if sum(f.isActive for f in eg.faces) > 1])
			
			#Try to close up the space around nodes
			for nd_cur in nds_curFront:
				self.try_closeupSpacearound_nd(nd_cur)
			
		print("Finished meshing domain\nElements created:", len(self.elements_3D), "Total volume:", round(sum(ele.volume for ele in self.elements_3D), DIGIT_HEIGHT))
		print("Nodes: {}. Edges: {}".format(len(self.nodes), len(self.edges)))
		print("Active faces left:", sum(f.isActive for f in self.elements_2D))
		
	def genaTetraElement(self, nodes):
		if not isinstance(nodes, list): nodes = list(nodes)
		self.elements_3D.append(ele := Ele_Tetrahedron(self, nodes))
		return ele
		
	def try_closeupSpacearound_Eg(self, edge, height, maxAgnle_directClosing=100):
		if fs := [f for f in edge.faces if f.isActive]:
			while fs:
				nd_other_2 = next(iter((f2 := fs[1]).nodes - (f1 := fs[0]).nodes))
				largeAngle = f1.angle_withFace_deg(f2) > maxAgnle_directClosing
				if not (node_new := self.try_budEle_Height(f1, height, npA([f.coors for f in self.elements_2D]),
														   preferNode=None if largeAngle else nd_other_2)[0]):
					print("Wrong while closing edge", edge)
					raise RuntimeError
				ele = self.genaTetraElement([node_new, *f1.nodes])
				print("Genned a tetra ele {} while closing edge {}".format(ele, edge))
				if fs := {f for f in edge.faces if f.isActive} - {f2}: fs = [f2, next(iter(fs))]
				
	def try_closeupSpacearound_nd(self, nd_cur, maxAngle_directClosing=90):
		if not (faces := {f for f in nd_cur.faces if f.isActive}): return
		if len(faces) != len({eg for f in faces for eg in f.edges} & nd_cur.edges): return
		print("\nChecking nd_cur:", nd_cur)
		
		#Check if nd_cur has active faces that form a closed polehedral cone. If yes and a pair of neighboring triangles form small angle, close it.
			#Loop until all such small angles are closed. If there are still active faces, pick the center of remaining nodes, and make all faces close toward this center
		while faces:
			nds_circ = {nd for f in faces for nd in f.nodes} - {nd_cur}
			tups_f1_nd2_ndAngle = [] #For each edge connected to nd_cur, check if its two connected opening faces form small enough angles
			for nd in nds_circ:
				(f1, f2) = nd.faces & faces
				nd1, nd2 = (f1.nodes | f2.nodes) - {nd_cur, nd}
				ndAngle = 180 + (1 if f1.angle_withFace_deg(f2) < 180 else -1) * (-180 + angle_btw_2vecs(nd1.coor - nd.coor, nd2.coor - nd.coor))
				tups_f1_nd2_ndAngle.append((f1, nd1 if nd2 in f1.nodes else nd2, ndAngle))
			#Try closing the pair of faces that form smallest angles.
				#After generating a closing tetra, finish if no more active faces left, otherwise start another loop
			cs_faces, genSuccess = npA([f.coors for f in self.elements_2D]), False
			for f1, nd2, angle in sorted(tups_f1_nd2_ndAngle, key=lambda t: t[2]):
				if angle > maxAngle_directClosing: continue
				if node_new := self.try_budEle_Height(f1, 0, cs_faces, preferNode=nd2)[0]:
					ele, genSuccess = self.genaTetraElement([*f1.nodes, node_new]), True
					print("Genned a tetra ele {} while closing node {}".format(ele, nd_cur))
					break
			
			if not (faces := {f for f in nd_cur.faces if f.isActive}): break
			elif not genSuccess:  # There is still opening faces, but they all form large angles.
				#No more small angles between faces, now try converging towards center.
				c_cen = np.average([nd.coor for nd in {nd for f in faces for nd in f.nodes} - {nd_cur}], axis=0)
				print("Use center to create tetras", c_cen)
				nd_cen, fs_preClosing = None, faces.copy()
				while faces:
					f, cs_faces = next(iter(faces & fs_preClosing)), npA([f.coors for f in self.elements_2D])
					nd_cen, collided = self.try_budEle_Height(f, 0, cs_faces, preferNode=nd_cen, preferCoor=c_cen)
					if not nd_cen: raise RuntimeError
					ele = self.genaTetraElement((nd_cen, *f.nodes))
					print("Genned a tetra ele {} while large space around closing node{}".format(ele, nd_cur))
					if collided:
						print("Closing large space around node {} encounters collision".format(nd_cur))
						break
					else: faces = {f for f in nd_cur.faces if f.isActive}
	
	def try_nearbyNodes(self, face, height, cs_faces,
						lateral=0.52, vertical_min=0.5, vertical_max=1.2):
		nodes, normal, c_cen, ownNodes = self.nodes, face.normal, face.c_cen, face.nodes
		vs_cen2Nodes = npA([nd.coor for nd in nodes]) - c_cen  # shape(n, 3)
		heights_fromface, horDist_fromCen = (normal @ vs_cen2Nodes.T).round(DIGIT_HEIGHT), npNorm(np.cross(normal, vs_cen2Nodes), axis=1)
		dist_fromCen, maxHorDist = npNorm(vs_cen2Nodes, axis=1), lateral * height
		idx_allowed = np.where((vertical_min * height < heights_fromface) & (heights_fromface < vertical_max * height) & (horDist_fromCen < maxHorDist))[0]
		nodes_ds = sorted(((nodes[i], dist_fromCen[i]) for i in idx_allowed), key=lambda tup: tup[1])
		return next((nd for nd, _ in nodes_ds if not self.tetra_cuts_anyTri(npA([*face.coors, nd.coor]), cs_faces)), None)
		
	def try_budEle_Height(self, face, height, cs_faces, preferNode=None, preferCoor=None,
							largerHeight_2Try=1.2):
		nodes, normal, ownNodes, c_cen = self.nodes, face.normal, face.nodes, face.c_cen
		c1, c2, c3 = face.coors
		
		if preferNode and preferNode in face.nodes:
			print(face, preferNode)
			raise ValueError
		if preferNode: cs_tetra_new = npA([c1, c2, c3, preferNode.coor])
		elif preferCoor is None: cs_tetra_new = npA([c1, c2, c3, c_cen + largerHeight_2Try * normal * height])
		else: cs_tetra_new = npA([c1, c2, c3, preferCoor])
		#欲生成新四面体，需要先检测其是否与已有faces相重叠。如没有，则可以直接生成，如果有，则找到所有有重叠的face
		if facesCut := self.getFaces_cutby_tetra(cs_tetra_new, cs_faces):
			nds_fromFacesCut = list({nd for ele in facesCut for nd in ele.nodes} - ownNodes)
			vs_cen_to_nds = npA([nd.coor for nd in nds_fromFacesCut]) - c_cen
			angles_from_normal = np.degrees(np.arccos(normal.dot(vs_cen_to_nds.T) / npNorm(vs_cen_to_nds, axis=1))).round(1)
			nds_angles = sorted(zip(nds_fromFacesCut, angles_from_normal), key=lambda t: t[1])
			return next((nd for nd, angle in nds_angles if angle < 90 and not self.tetra_cuts_anyTri(npA([c1, c2, c3, nd.coor]), cs_faces)), None), True
		elif preferNode:
			print("Bud with existing node", preferNode)
			return preferNode, False
		else:
			print("Bud with new node")
			return Node(self, len(nodes), (c_cen+normal*height) if preferCoor is None else preferCoor), False

	"""
	Save and load mesh for FEA solution.
	"""
	def saveMesh(self, filename="3D_Mesh.txt"):
		with open(filename, 'w') as fout:
			fout.write("Nodes:\n")
			for nd in self.nodes:
				fout.write("{}, [{},{},{}]\n".format(nd.i, *nd.coor.round(DIGIT_HEIGHT)))
			fout.write("Volumes:\n")
			for volume in self.elements_3D:
				n1, n2, n3, n4 = volume.nodes
				fout.write("{},{},{},{}\n".format(n1.i, n2.i, n3.i, n4.i))
				
	def loadMesh(self, filename="3D_Mesh.txt"):
		with open(filename, 'r') as fin:
			i = (lines := fin.readlines()).index("Volumes:\n")
			lines_nodes, lines_volumes = lines[1:i], lines[i + 1:]
			nodes, elements_3D = self.nodes, self.elements_3D = [], []
			
			for i, l in enumerate(lines_nodes):
				exec("Node(self, {}, np.array({}))".format(i, l[l.index(',') + 1:].strip("\n")))
			print("Nodes in mesh:\n", nodes)
			for l in lines_volumes:
				i0, i1, i2, i3 = l.strip("\n").split(',')
				i0, i1, i2, i3 = int(i0), int(i1), int(i2), int(i3)
				elements_3D.append(Ele_Tetrahedron(self, [nodes[i0], nodes[i1], nodes[i2], nodes[i3]], calcFEAMatrices=True))
			print("Volumes in mesh:\n", elements_3D)
	
	"""Calculate matrices neccesary for FEA."""
	def getArr_scalarProducts_IntoverVolume_Linear(self, ViVj_not_gradVigradVj=False):
		arr = np.zeros([n := len(self.nodes)] * 2)
		for ele in self.elements_3D:
			d, (nd0, nd1, nd2, nd3) = ele.dict_node_to_index, ele.nodes
			tups = ((nd0, d[nd0]), (nd1, d[nd1]), (nd2, d[nd2]), (nd3, d[nd3]))
			for (nd1, i1), (nd2, i2) in product(tups, tups): #4x4 pairs of ij
				if (i := nd1.i) <= (j := nd2.i):  # only consider diagonal and above
					if ViVj_not_gradVigradVj: arr[i, j] += ele.arr_Vi_Vj_IntoverVolume[i1, i2]
					else: arr[i, j] += ele.arr_gradVi_gradVj_IntoverVolume[i1, i2]
		for i in range(n):
			for j in range(n):
				if i > j: arr[i, j] = arr[j, i]
		return arr
		
	def getArr_Product_IntoverFace_Linear(self, VigradVj_dot_normal_not_ViVj=False):
		arr = np.zeros([len(self.nodes)] * 2)
		for vol, f in ((next(iter(face.volumes)), face) for face in self.elements_2D if len(face.volumes) == 1):
			(nd1, nd2, nd3), d = f.nodes, vol.dict_node_to_index
			if VigradVj_dot_normal_not_ViVj: arr_IntoverEdge = vol.dict_faceIdx_to_arr_Vi_gradVj_IntoverFace[(d[nd1], d[nd2], d[nd3])]
			else: arr_IntoverEdge = vol.dict_faceIdx_to_arr_Vi_Vj_IntoverFace[(d[nd1], d[nd2], d[nd3])]
			for nd1, nd2 in product(f.nodes, f.nodes): arr[nd1.i, nd2.i] += arr_IntoverEdge[d[nd1], d[nd2]]
		return arr
	
	
def create3DDomain_init2DBoundaries(ls_polies, height):
	domain_3D = Domain_3D()

	nodes_total = sorted((Node(domain_3D, pt.i, pt.coor) for pt in {pt for poly in ls_polies for pt in poly.pts_split}), key=lambda nd: nd.i)
	ls_edges, ls_nodes = [], [[nodes_total[pt.i] for pt in poly.pts_split] for poly in ls_polies]
	
	for poly in ls_polies:
		pts, edges_poly = poly.pts_split, []
		for pt1, pt2 in zip(pts, pts[1:] + [pts[0]]):
			nd1, nd2 = nodes_total[pt1.i], nodes_total[pt2.i]
			if egShared := nd1.edgeSharedwith(nd2): edges_poly.append(egShared)
			else: edges_poly.append(Edge(domain_3D, nd1, nd2))
		ls_edges.append(edges_poly)

	ls_domains_2D, dict_egBoundary_tofaces = [], {}
	for nodes, edges, poly in zip(ls_nodes, ls_edges, ls_polies):
		n_edges, n_face = poly.n_edges, poly.normal
		for edge, n_edge in zip(edges, n_edges): edge.normal = n_edge
		
		domain_2D = Domain_2D()
		domain_2D.prepare4Mesh(edges, nodes)
		try:
			domain_2D.genTriangleEle_Wavefront(height, maxIter=5)
		except Exception as e: print("Error.", e)
		domain_2D.dissolveSmallTriangles(height)
		for face in domain_2D.elements_2D: face.normal = n_face
		domain_3D.elements_2D += domain_2D.elements_2D
		for eg in edges:
			if eg in dict_egBoundary_tofaces: dict_egBoundary_tofaces[eg] |= eg.faces
			else: dict_egBoundary_tofaces[eg] = eg.faces.copy()
			eg.faces = set()
		ls_domains_2D.append(domain_2D)
		
	for eg, faces in dict_egBoundary_tofaces.items(): eg.faces = faces
	domain_3D.nodes = list({nd for domain_2D in ls_domains_2D for nd in domain_2D.nodes})
	domain_3D.edges = list({edge for domain_2D in ls_domains_2D for edge in domain_2D.edges})
	return domain_3D, ls_domains_2D