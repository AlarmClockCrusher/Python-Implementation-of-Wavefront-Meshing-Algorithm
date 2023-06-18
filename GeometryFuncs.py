import numpy as np
from numpy import array as npA
from numpy.linalg import norm as npNorm
from numpy.linalg import solve as npSolve
from numpy import newaxis as npNuax
from itertools import permutations, combinations, product
from math import ceil


IDX_3CHOOSE2 = ([0, 1], [0, 2], [1, 2])
IDX_4CHOOSE3 = ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])
arr_90deg = npA([[0, -1], [1, 0]])
arr_90deg_3D = npA([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
DIGIT_HEIGHT = 4
DIGIT_ACCU = 6

def angle_btw_2vecs(v1, v2):
	if npNorm(v1).round(3) == 0 or npNorm(v2).round(3) == 0:
		print("Wrong v1, v2:", v1, v2)
		raise ValueError
	return np.degrees(np.arccos(v1.dot(v2) / npNorm(v1) / npNorm(v2)))

def solidAngle_between_Coors(coors): #1st coor as the starting point
	v1, v2, v3 = vecs = coors[1:] - coors[0]
	norm1, norm2, norm3 = npNorm(vecs, axis=1)
	return np.degrees(2 * np.arctan(v1.dot(np.cross(v2, v3)) / (norm1 * norm2 * norm3 + norm1 * v2.dot(v3) + norm2 * v1.dot(v3) + norm3 * v1.dot(v2))))
	

def is_coorsonSamePlane(coors): #For dim < 3 or dim <= num of coors, then they are always on same plane
	if (n := len(coors[0])) < 3 or len(coors) <= n: return True
	for c1, c2, c3 in combinations(coors, 3):
		if (normal := np.cross(c2-c1, c3-c1).round(DIGIT_ACCU)).any():
			return not normal.dot((coors-c1).T).round(DIGIT_ACCU).any()
	return False


def getNormal_ofPlane(coors):
	normal = next((np.cross(c2-c1, c3-c1) for c1, c2, c3 in combinations(coors, 3) if np.cross(c2-c1, c3-c1).round(DIGIT_HEIGHT).any()), None)
	if normal is not None: return normal / npNorm(normal)
	else: return None

def getVecs_ifCanFormConvexPolyPlane(coors):
	#if (v := next((np.cross(c2-c1, c3-c1) for c1, c2, c3 in combinations(coors, 3) if np.cross(c2-c1, c3-c1).any()), None)) is None:
	if (normal := getNormal_ofPlane(coors)) is None: raise Exception
	dim, n, c0, c_cen = len(coors[0]), len(coors), min(coors, key=lambda tup: sum(tup)), np.average(coors, axis=0)
	if dim == 3 and not is_coorsonSamePlane(coors):
		print("Not on same same plane")
		return False
	# Picked a coordinate as a starting point. Find a permutation where all other nodes are on the same side of any edge.
	for cs in permutations(coors):
		if (c0 - cs[0]).any(): continue
		vecs = np.roll(cs := np.asarray(cs), -1, axis=0) - cs
		if not any((normal.dot(np.cross(vecs[i], cs - cs[i]).T).round(DIGIT_HEIGHT) < 0).any() for i in range(n)):
			if dim == 2:
				normals = np.matmul(arr_90deg, vecs.T).T
				if normals[0].dot(c_cen-(cs[0]+cs[1])/2) <= 0: normals = -normals
			else: #3D vectors
				normals = np.cross(vecs, np.cross(vecs, np.roll(vecs, -1, axis=0)))
				vecs_to_cen = np.average(cs, axis=0) - (cs + vecs)
				normals = normals * np.tile(np.sum(normals * vecs_to_cen, axis=1), (dim, 1)).T
			return cs, vecs, normals / np.tile(npNorm(normals, axis=1), (dim, 1)).T #CW or CCW, but normals are always pointing inwards.
	return (), (), ()

def getVecs_AlreadyConvexandSorted(coors, n_face=None):
	if n_face is None and (n_face := getNormal_ofPlane(coors)) is None: raise Exception
	dim, c_cen = len(coors[0]), np.average(coors, axis=0)
	print("Getting vecs: normal", n_face)
	vecs = np.roll(coors, -1, axis=0) - coors
	if dim == 2: normals = (arr_90deg @ vecs.T).T
	else:
		if n_face.dot(vecs.T).round(DIGIT_HEIGHT).any():
			print("Not on same same plane", n_face.dot(vecs.T))
			raise Exception
		else: normals = np.cross(n_face, vecs) #3D vectors
	#The n_edges should point inward the polygon
	if normals[0].dot(c_cen - (coors[0] + coors[1]) / 2) <= 0: normals = -normals
	return coors, vecs, normals / np.tile(npNorm(normals, axis=1), (dim, 1)).T
	
def splitConvexPoly(cs, vecs, normals, d_max):
	ls_cs, ls_vecs, ls_normals = [], [], []
	for c1, c2, v, normal in zip(cs, np.roll(cs, -1, axis=0), vecs, normals):
		n = ceil(npNorm(c2-c1) / d_max)
		ls_cs += np.linspace(c1, c2, n + 1)[:-1].tolist()
		ls_vecs += np.tile(v/n, (n, 1)).tolist()
		ls_normals += np.tile(normal, (n, 1)).tolist()
	return npA(ls_cs), npA(ls_vecs), npA(ls_normals)
	
def splitSegment_VarLength(c1, c2, height):
	#Always include c1 and c2 at index 0 and -1
	if not callable(height):
		return np.linspace(c1, c2, ceil(npNorm(c2 - c1) / height) + 1)
	v, l, cs = (c2 - c1) / (l_tot := npNorm(c2 - c1)), 0, [c1]
	while True:
		l0 = height(c1)
		if l + l0 >= l_tot: cs.append(c2); break
		else:
			l1 = (l0 + height(c1 + l0 * v)) / 2
			if l + l1 >= l_tot: cs.append(c2); break
			else:
				l += l1
				cs.append(c1 := c1 + v * l1)
	return npA(cs)

def getNormals_OutofTriangle(coors): #return normals to edges (outward), and normal to face (either direction)
	#ith n_edge is opposite to ith triangle vert
	if (dim:=len(coors[0])) != 3: coors = np.append(coors, np.zeros((3, 1)), axis=1)
	ns_edge = np.cross(coors[[2, 0, 1]] - coors[[1, 2, 0]], n_face := np.cross(*(coors[1:] - coors[0])))
	if dim == 2: ns_edge, n_face = ns_edge[:, :2], n_face[2]
	return ns_edge / np.tile(npNorm(ns_edge, axis=1), (dim, 1)).T, n_face / npNorm(n_face)
	
def getNormals_OutofTetra(coors):
	"""
	from c0: (c2-c1) x (c3-c1)  c0->c1
	from c1: (c3-c2) x (c0-c2)  c1->c2
	from c2: (c0-c3) x (c1-c3)  c2->c3
	from c3: (c1-c0) x (c2-c0)  c3->c0
	"""
	normals = np.cross(coors[[2, 3, 0, 1]]-coors[[1, 2, 3, 0]], coors[[3, 0, 1, 2]]-coors[[1, 2, 3, 0]])
	vecs = coors[[1, 2, 3, 0]] - coors
	normals = normals * np.tile((np.heaviside((normals*vecs).sum(axis=1), 1) * 2 - 1), (3, 1)).T
	return normals / np.tile(npNorm(normals, axis=1), (3, 1)).T
	
	
"""
Functions for loading an STL model and analyze the triangles it has
"""
#Process a STL model. Get the node coordinates, corresponding non-repetitive edges, normalized normals of each triangle.
def STLMesh_Verts_EdgeVecs_EdgeNodePairs_TriangleNormals(stlMesh, fullyClosed=True):
	vectors = stlMesh.vectors #(n, 3, 3) each (3, 3) is 3 coors that form a triangle
	cs_nodes = np.unique(stlMesh.vectors.reshape(-1, 3), axis=0)
	vec_edges = (stlMesh.vectors[:, [1, 2, 0], :] - stlMesh.vectors).reshape(-1, 3)  # (n*3, 3) there are n*3 vec_edges
	nodePairs = np.append(stlMesh.vectors, stlMesh.vectors[:, [1, 2, 0], :], axis=2).reshape(-1, 6) #(n*3, 6) each 3+3 is two nodes' coordinates
	if fullyClosed: # For bodies enclosed by triangles, each edge is shared by two triangles. We can filter out half of these vec_edges.
		x_dot_edges, y_dot_edges, z_dot_edges = vec_edges.T
		keepEdges = (x_dot_edges > 0) | ((x_dot_edges == 0) & ((y_dot_edges > 0) | ((y_dot_edges == 0) & (z_dot_edges > 0))))
		idx_keepEdges = np.nonzero(keepEdges)[0]
		if 3 * len(vectors) != 2 * len(idx_keepEdges):
			print("Failed to keep exactly half of vec_edges", len(vectors), len(idx_keepEdges))
			raise Exception
	else:
		d = {}
		for i, ndPair in enumerate(nodePairs):
			if tuple(ndPair) not in d and tuple(ndPair[[3, 4, 5, 0, 1, 2]]) not in d: d[tuple(ndPair)] = i
		idx_keepEdges = list(d.values())
	print("Num of triangles in model:", vectors.shape, "Edges found: ", nodePairs.shape, "Non-repetitive edges:", len(idx_keepEdges))
	edges_nonrepetitive, nodePairs_nonrepetitive = vec_edges[idx_keepEdges], nodePairs[idx_keepEdges]
	normals = stlMesh.normals
	return cs_nodes, edges_nonrepetitive, nodePairs_nonrepetitive, normals / npNorm(normals, axis=1)[:,npNuax]
