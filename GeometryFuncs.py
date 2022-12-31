import numpy as np
from numpy import array as npA
from itertools import permutations, combinations, product
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import mpl_toolkits.mplot3d as a3
import matplotlib.gridspec as gridspec


INDS_3CHOOSE2 = ([0, 1], [0, 2], [1, 2])
INDS_4CHOOSE3 = ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])

def ax_Init(ax, widgets):
	ax.quiver(*np.zeros((3, 3)), *np.diag([1, 1, 1]))
	ax.set(xlabel="x", ylabel="y", zlabel="z")
	#ax.set_position([0.08, 0.23, 0.8, 0.75])


def ax_plotDots_Vecs_Normals(ax, cs, vecs, normals):
	ax.scatter(*cs.T)
	for i, c in enumerate(cs): ax.text(*c, i)
	ax.quiver(*cs.T, *vecs.T)
	ax.quiver(*(cs + np.roll(cs, -1, axis=0)).T / 2, *normals.T, color='red')

def ax_plotArrows_alongCoors_2D(ax, arr_coors, head_width=0.02):
	N = len(arr_coors)
	for i, c in enumerate(arr_coors):
		ax.arrow(*c, *(arr_coors[(i+1)%N]-c), head_width=head_width)

def ax_plotPolygons(ax, coors, alpha=0.2, color="blue"):
	polyCol = a3.art3d.Poly3DCollection(coors, alpha=alpha, color=color)
	ax.add_collection3d(polyCol)
	return polyCol

def ax_plotTetra(ax, coors, alpha=0.2, color="blue"):
	tetraCol = a3.art3d.Poly3DCollection([coors[list(inds)] for inds in combinations(range(4), 3)],
							  			alpha=alpha, color=color)
	ax.add_collection3d(tetraCol)
	return tetraCol

def ax_textPoints(ax, coors, color="black"):
	for i, c in enumerate(coors):
		a = ax.text(*c, i, color=color)
		
half_sqrt3 = np.sqrt(3) / 2
arr_90deg = npA([[0, -1], [1, 0]])

def genXYZarray(x0, x1, Nx, y0, y1, Ny, z0, z1, Nz):
	xs, ys, zs = np.meshgrid(np.linspace(x0, x1, Nx), np.linspace(y0, y1, Ny), np.linspace(z0, z1, Nz))
	xs, ys, zs = xs.reshape(-1, ), ys.reshape(-1, ), zs.reshape(-1, )
	return npA([xs, ys, zs]).T

def genXYarray(x0, x1, Nx, y0, y1, Ny):
	xs, ys = np.meshgrid(np.linspace(x0, x1, Nx), np.linspace(y0, y1, Ny))
	xs, ys = xs.reshape(-1, ), ys.reshape(-1, )
	return npA([xs, ys]).T


def genXYarray_EquilateralTriangles(x0, x1, y0, y1, size):
	# Generated dot array that form equilateral triangles
	xs, ys = np.meshgrid(np.arange(x0, x1 + size, size), np.arange(y0, y1 + half_sqrt3 * size, half_sqrt3 * size))
	for i in range(1, len(xs), 2):
		xs[i] += size / 2
	return npA([xs.reshape(-1, ), ys.reshape(-1, )])

def genEquiDistCoors_Between2Coors(arrCoors, d_max, onlyBetween=False):
	#With two coordinates [(x1, y1), (x2, y2)] and maximum distance d, generate equally spaced dots along the segment (excluding the two ends)
	c1, c2 = arrCoors
	arr = np.linspace(c1, c2, ceil(np.linalg.norm(c2-c1) / d_max)+1)
	return arr[1:-1] if onlyBetween else arr

def genVariableDistCoors_Between2Coors(x0, x1, N, f=lambda x: 0.7**x):
	xs = np.linspace(x0, x1, N)
	ys = npA([f(x) for x in xs])[::-1]
	ys -= ys[0]
	scale = (xs[-1] - xs[0]) / (ys[-1] - ys[0])
	return (x0 + ys * scale)[1:-1]

def genXYZarray_EquilateralTetrahedrons(x0, x1, y0, y1, z0, z1, size):
	# Generated dot array that form equilateral triangles
	xs, ys = np.meshgrid(np.arange(x0, x1 + size, size), np.arange(y0, y1 + half_sqrt3 * size, half_sqrt3 * size))
	for i in range(1, len(xs), 2): xs[i] += size / 2
	d = np.sqrt(2/3); Nz = len(heights := np.arange(z0, z1+size*d, size*d))
	zs = npA([z*np.ones(xs.shape) for z in heights])
	xs, ys = npA([xs for i in range(Nz)]), npA([ys for i in range(Nz)])
	for i in range(1, Nz, 2): ys[i] -= 1 / np.sqrt(3)
	return npA([xs.reshape(-1, ), ys.reshape(-1, ), zs.reshape(-1, )])


DIGIT_ACCU = 7
ACCURACY = 1e-7
# For an mxn matrix A, equation set AX=B has solutions depending on the rank of coefficient matrix A and augmented matrix (A|B)
# If rank(A) < rank(A|B), there is no solution
# If rank(A) = rank(A|B) = n, there is one and only set of solution
# If rank(A) = rank(A|B) < n, there is unlimited number of solutions.
"""Get angle or solid angle formed between nodes"""
def angle_between_3coors(coor_cen, coor_1, coor_2):
	arr_Coors = npA([coor_cen, coor_1, coor_2])
	return angle_between_2vecs(arr_Coors[1] - arr_Coors[0], arr_Coors[2] - arr_Coors[0])

def angle_between_2vecs(v1, v2):
	return np.degrees(np.arccos(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))

def solidAngle_between_arrVecs(arr_Vecs):
	norm1, norm2, norm3 = np.linalg.norm(arr_Vecs, axis=1)
	v1, v2, v3 = arr_Vecs
	return np.degrees(2 * np.arctan(v1.dot(np.cross(v2, v3)) / (norm1 * norm2 * norm3 + norm1 * v2.dot(v3) + norm2 * v1.dot(v3) + norm3 * v1.dot(v2))))

def solidAngle_between_arrCoors(arr_Coors):
	return solidAngle_between_arrVecs(arr_Coors[1:] - arr_Coors[0])

def solidAngle_between_4coors(coor_cen, coor_1, coor_2, coor_3):
	arr_Coors = npA([coor_cen, coor_1, coor_2, coor_3])
	return solidAngle_between_3vecs(arr_Coors[1] - arr_Coors[0], arr_Coors[2] - arr_Coors[0], arr_Coors[3] - arr_Coors[0])

def solidAngle_between_3vecs(v1, v2, v3):
	norm1, norm2, norm3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
	return np.degrees(2 * np.arctan(v1.dot(np.cross(v2, v3)) / (norm1 * norm2 * norm3 + norm1 * v2.dot(v3) + norm2 * v1.dot(v3) + norm3 * v1.dot(v2))))


"""Get triangle area/tetrahedral volume formed among nodes"""
def area_between_arr3Coors(arr_coors):
	return abs(np.cross(arr_coors[1] - arr_coors[0], arr_coors[2] - arr_coors[0])) / 2

def area_between_3coors(*coors):
	arr_coors = npA(coors)
	return abs(np.cross(arr_coors[1] - arr_coors[0], arr_coors[2] - arr_coors[0])) / 2

def volume_between_arr4Coors(arr_coors):
	v01, v02, v03 = arr_coors[1:] - arr_coors[0]
	return abs(v01.dot(np.cross(v02, v03))) / 6
	
def volume_between_3vecs(*vecs):
	v01, v02, v03 = vecs
	return abs(v01.dot(np.cross(v02, v03))) / 6

"""Check the intersection between geometry elements"""
def is_coorsonSameLine(*coors):
	if (n := len(coors)) < 3: return True
	arr_coors = npA(coors)
	vecs = npA(arr_coors[1:]) - arr_coors[0]
	return (np.linalg.norm(np.cross(vecs[0], vecs[1:]), axis=len(coors[0]) - 2).round(DIGIT_ACCU) == 0).all()


def is_arrCoorsonSameLine(arr_coors):
	if (n := len(arr_coors)) < 3: return True
	vecs = npA(arr_coors[1:]) - arr_coors[0]
	return (np.linalg.norm(np.cross(vecs[0], vecs[1:]), axis=len(arr_coors[0]) - 2).round(DIGIT_ACCU) == 0).all()


# If c0 coincides with c1 or c2, return 1
# If c0 is on c1-c2, return 2; otherwise return 0
def check_coor_onSegment(c0, c1, c2):
	arr_coors = npA([c0, c1, c2])
	v1, v2 = arr_coors[1] - arr_coors[0], arr_coors[2] - arr_coors[0]
	if round(np.linalg.norm(v1), DIGIT_ACCU) and round(np.linalg.norm(v2), DIGIT_ACCU):
		return 2 if round(v1.dot(v2), DIGIT_ACCU) < 0 and not round(np.linalg.norm(np.cross(v1, v2)), DIGIT_ACCU) else 0
	return 1


def vec0_isLinCombof_vecs(v0, vecs):  # Check if vectors are in same plane.
	Av0 = np.concatenate([A := vecs.T, v0.reshape(-1, 1)], axis=1) #Augmented matrix
	return len(vecs) == np.linalg.matrix_rank(A) == np.linalg.matrix_rank(Av0)

def vec0_getLinCombof_vecs(v0, vecs):  # Check if vectors are in same plane.
	
	Av0 = np.concatenate([A := vecs.T, v0.reshape(-1, 1)], axis=1)
	if (n := len(vecs)) == np.linalg.matrix_rank(A) == np.linalg.matrix_rank(Av0):
		for inds in combinations(range(len(v0)), n):
			inds = list(inds)
			if np.linalg.det(A[inds]).round(DIGIT_ACCU): return np.linalg.solve(A[inds], v0[inds]).round(DIGIT_ACCU)
	return None


def is_coorsonSamePlane(coors): #For dim < 3 or dim <= num of coors, then they are always on same plane
	if (n := len(coors[0])) < 3 or len(coors) <= n: return True
	for c1, c2, c3 in combinations(coors, 3):
		if (normal := np.cross(c2-c1, c3-c1).round(DIGIT_ACCU)).any():
			return not normal.dot((coors-c1).T).round(DIGIT_ACCU).any()
	return False


def getNormal_arr3Coor_PerptoC0(arr_coors):
	if len(arr_coors) != 3: raise Exception
	v01, v02 = vecs = arr_coors[1:] - arr_coors[0]
	va_dot_vb, (modSquare_01, modSquare_02) = v01.dot(v02).round(DIGIT_ACCU), (np.linalg.norm(vecs, axis=1)**2).round(DIGIT_ACCU)
	if va_dot_vb == modSquare_01: normal = v01 #If 01 perp to 12, then v01 is already the normal
	elif va_dot_vb == modSquare_02: normal = v02
	else:
		normal = (modSquare_02 - va_dot_vb) / (modSquare_01 - va_dot_vb) * v01 + v02
		if normal.dot(v01).round(DIGIT_ACCU) < 0: normal = -normal
	return (normal / np.linalg.norm(normal)).round(DIGIT_ACCU)

def getNormal_arr4Coor_PerptoC0(arr_coors):
	if len(arr_coors) != 4: raise Exception
	normal = np.cross(*(arr_coors[2:] - arr_coors[1]))
	if not np.linalg.norm(normal).round(DIGIT_ACCU):
		print("Tetra wrong", arr_coors)
		raise Exception
	if normal.dot(np.average(arr_coors[1:], axis=0)-arr_coors[0]) < 0: normal = -normal
	return normal / np.linalg.norm(normal)

# Generate vecs that connect the coordinates head-to-tail
def getVecsinNodeChain(coors):
	if not isinstance(coors, np.ndarray): coors = np.asarray(coors)
	return np.roll(coors, -1, axis=0) - coors
	

def twoCoorsClose(c1, c2, d_thres):
	return (np.linalg.norm(c1 - c2) < d_thres).any()

# If coor0 and coors are not on the same plane, return 0
# If coor coincides with vertices, then return 1
# If coor is on edges, then return 2
# If coor is strictly within polygon, return 3; otherwise return 0
def is_coorinPoly_Convex(coor0, coors, arr_vecs=None):
	if arr_vecs is None: arr_vecs = getVecsinNodeChain(coors)
	"""YOU MUST MAKE SURE THAT COORS ARE DIFFERENT FROM EACH OTHER"""
	for i in range(N := len(coors)):  # If coordinate on vertices or edges, return 1 or 2
		if a := check_coor_onSegment(coor0, coors[i], coors[(i + 1) % N]): return a
	vecs_from0 = npA(coors) - npA(coor0)
	v1, v2 = vecs_from0[:2]
	if not is_coorsonSamePlane(np.concatenate([coor0.reshape(1, -1), coors], axis=0)): return 0
	# Possibilities: coor strictly outsie or strictly within
	v_normal = np.cross(vecs_from0[0], vecs_from0[1])
	r = v_normal.dot(np.cross(vecs_from0, arr_vecs).T).round(DIGIT_ACCU)
	return 3 if (r > 0).all() or (r < 0).all() else 0

def is_coorinPoly_Convex_2D(coor0, coors, arr_vecs=None):
	if arr_vecs is None: arr_vecs = getVecsinNodeChain(coors)
	"""YOU MUST MAKE SURE THAT COORS ARE DIFFERENT FROM EACH OTHER"""
	for i in range(N := len(coors)):  # If coordinate on vertices or edges, return 1 or 2
		if a := check_coor_onSegment(coor0, coors[i], coors[(i + 1) % N]): return a
	vecs_from0 = npA(coors) - npA(coor0)
	v1, v2 = vecs_from0[:2]
	# Possibilities: coor strictly outsie or strictly within
	v_normal = np.cross(vecs_from0[0], vecs_from0[1])
	r = v_normal.dot(np.cross(vecs_from0, arr_vecs).T).round(DIGIT_ACCU)
	return 3 if (r > 0).all() or (r < 0).all() else 0

def is_coorinTriangle(c0, coors):
	if not (v_coors_to_c0:=(c0-coors)).round(DIGIT_ACCU).any(axis=1).all(): return 1  # On one of the vertices
	vecs = (coors[1:] - coors[0])
	if len(c0) < 3:
		if not np.linalg.det(vecs).round(DIGIT_ACCU):
			print("Check det", np.linalg.det(vecs))
			print("triangle is flat!", coors)
			raise Exception
	if len(c0) < 3: vec = np.linalg.solve(vecs.T, v_coors_to_c0[0]).round(DIGIT_ACCU)
	elif (vec := vec0_getLinCombof_vecs(v_coors_to_c0[0], vecs)) is None: return 0
	s = round(sum(vec), DIGIT_ACCU-1)
	if (vec < 0).any() or s > 1: return 0  # Strictly outside tetrahedron
	elif (vec > 0).all() and s < 1: return 3  # Strictly inside tetrahedron
	else: return 2

def is_coorinTetrahedron(c0, coors):
	if not (v_coors_to_c0:=c0-coors).round(DIGIT_ACCU).any(axis=1).all(): return 1 #On one of the vertices
	#if not np.linalg.det((coors[1:]-coors[0])): print("Check coors of tetra\n", coors)
	s = round(sum(vec := np.linalg.solve((coors[1:]-coors[0]).T, v_coors_to_c0[0]).round(DIGIT_ACCU)), DIGIT_ACCU-1)
	if (vec < 0).any() or s > 1: return 0 #Strictly outside tetrahedron
	elif (vec > 0).all() and s < 1: return 4 #Strictly inside tetrahedron
	else: return 2 if (n := np.count_nonzero(vec == 0)) == 2 or (n == s == 1) else 3 #2 if on edge, 3 if on face
	
def getNormal_ofPlane(coors, normalize=False):
	normal = next((np.cross(c2-c1, c3-c1) for c1, c2, c3 in combinations(coors, 3) if np.cross(c2-c1, c3-c1).any()), None)
	if normal is not None and normalize: return normal / np.linalg.norm(normal)
	else: return normal

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
		vecs = getVecsinNodeChain(cs := np.asarray(cs))
		if not any((normal.dot(np.cross(vecs[i], cs - cs[i]).T).round(DIGIT_ACCU) < 0).any() for i in range(n)):
			if dim == 2:
				normals = np.matmul(arr_90deg, vecs.T).T
				if normals[0].dot(c_cen-(cs[0]+cs[1])/2) <= 0: normals = -normals
			else: #3D vectors
				normals = np.cross(vecs, np.cross(vecs, np.roll(vecs, -1, axis=0)))
				vecs_to_cen = np.average(cs, axis=0) - (cs + vecs)
				normals = normals * np.tile(np.sum(normals * vecs_to_cen, axis=1), (dim, 1)).T
			return cs, vecs, normals / np.tile(np.linalg.norm(normals, axis=1), (dim, 1)).T #CW or CCW, but normals are always pointing inwards.
	return (), (), ()

def getVecs_AlreadyConvexandSorted(coors):
	if (normal := getNormal_ofPlane(coors)) is None: raise Exception
	dim, n, c0, c_cen = len(coors[0]), len(coors), min(coors, key=lambda tup: sum(tup)), np.average(coors, axis=0)
	if dim == 3 and not is_coorsonSamePlane(coors):
		print("Not on same same plane")
		return False
	vecs = getVecsinNodeChain(coors)
	if dim == 2:
		normals = np.matmul(arr_90deg, vecs.T).T
		if normals[0].dot(c_cen-(coors[0]+coors[1])/2) <= 0: normals = -normals
	else: #3D vectors
		normals = np.cross(vecs, np.cross(vecs, np.roll(vecs, -1, axis=0)))
		vecs_to_cen = np.average(coors, axis=0) - (coors + vecs)
		normals = normals * np.tile(np.sum(normals * vecs_to_cen, axis=1), (dim, 1)).T
	return coors, vecs, normals / np.tile(np.linalg.norm(normals, axis=1), (dim, 1)).T
	
def splitConvexPoly(cs, vecs, normals, d_max):
	ls_cs, ls_vecs, ls_normals = [], [], []
	for c1, c2, v, normal in zip(cs, np.roll(cs, -1, axis=0), vecs, normals):
		n = ceil((length := np.linalg.norm(c2-c1)) / d_max)
		ls_cs += np.linspace(c1, c2, n + 1)[:-1].tolist()
		ls_vecs += np.tile(v/n, (n, 1)).tolist()
		ls_normals += np.tile(normal, (n, 1)).tolist()
	return npA(ls_cs), npA(ls_vecs), npA(ls_normals)
	
	
# Can handle parallel cases. coors[0] and coors[1] form a segment and coors[2] and coors[3] form the other
def check_2SegmentsIntersect_within_2D(coors1, coors2):
	if len(coors1) != 2 or len(coors2) != 2: raise Exception
	(a, b), (c, d) = v01, v23 = coors1[1] - coors1[0], coors2[1] - coors2[0]
	(x0, y0), ((x2, y2), (x3, y3)) = coors1[0], coors2
	if np.linalg.det(A := npA([[1, 0, -a, 0], [0, 1, -b, 0], [1, 0, 0, -c], [0, 1, 0, -d]])): #Non-parallel
		x, y, l, k = np.linalg.solve(A, npA([x0, y0, x2, y2])).round(DIGIT_ACCU)
		inter_on1st, inter_on2nd = (1 if l in (0, 1) else (2 if 0 < l < 1 else 0)), (1 if k in (0, 1) else (2 if 0 < k < 1 else 0))
		#print("inter_on1st, inter_on2nd", x, y, l, k, inter_on1st, inter_on2nd)
		#0: no touching, 1 if cross head-on, 2 if cross on one end and one inside, 3 if cross on both insides
		return max(inter_on1st + inter_on2nd - 1, 0)# if inter_on1st and inter_on2nd else 0
	else: #Two parallel segments
		if np.cross(v01, coors2[0] - coors1[0]).round(DIGIT_ACCU): return 0
		else: #Doesn't return 2. Only non-parallel segments return 2
			#0 if no overlapping at all, 1 if meet head-on or are same segment, 3 if partially overlap
			k02_to_01 = round(((x2 - x0) / a) if a else ((y2 - y0) / b), DIGIT_ACCU)
			k03_to_01 = round(((x3 - x0) / a) if a else ((y3 - y0) / b), DIGIT_ACCU)
			#2nd segment has both ends on same side of 1st segment (no touching)
			if (k02_to_01 > 1 and k03_to_01 > 1) or (k02_to_01 < 0 and k03_to_01 < 0): return 0
			#2nd segment has same ends as 1st (two identical segments)
			elif (k02_to_01==1 and k03_to_01==0) or (k02_to_01==0 and k03_to_01==1): return 1
			#head-on-head touch
			elif any((a == 1 and b > 1) or (a == 0 and b < 0) for a, b in ((k02_to_01, k03_to_01), (k03_to_01, k02_to_01))): return 1
			else: return 3
			

def check_2SegmentsIntersect_within_3D(coors1, coors2):
	if len(coors1) != 2 or len(coors2) != 2: raise Exception
	(a, b, c), (d, e, f) = v1, v2 = coors1[1] - coors1[0], coors2[1] - coors2[0]
	(x0, y0, z0), ((x2, y2, z2), (x3, y3, z3)) = coors1[0], coors2
	if not np.cross(v1, v2).round(DIGIT_ACCU).any():
		if not np.cross(v1, coors2[0]-coors1[0]).round(DIGIT_ACCU).any(): return 0 #Parallel but not on same line
		#Four points on same line.
		# 0 if no overlapping at all, 1 if meet head-on-head, 2 if are same segment, 3 if partially overlap
		k02_to_01 = round(((x2 - x0) / a) if a else (((y2 - y0) / b) if b else ((z2 - z0) / c)), DIGIT_ACCU)
		k03_to_01 = round(((x3 - x0) / a) if a else (((y3 - y0) / b) if b else ((z3 - z0) / c)), DIGIT_ACCU)
		# 2nd segment has both ends on same side of 1st segment (no touching)
		if (k02_to_01 > 1 and k03_to_01 > 1) or (k02_to_01 < 0 and k03_to_01 < 0): return 0
		# 2nd segment has same ends as 1st (two identical segments)
		elif (k02_to_01 == 1 and k03_to_01 == 0) or (k02_to_01 == 0 and k03_to_01 == 1): return 2
		# head-on touch
		elif any((a == 1 and b > 1) or (a == 0 and b < 0) for a, b in ((k02_to_01, k03_to_01), (k03_to_01, k02_to_01))): return 1
		else: return 3
	A = npA([[1, 0, 0, -a, 0], [0, 1, 0, -b, 0], [0, 0, 1, -c, 0], [1, 0, 0, 0, -d], [0, 1, 0, 0, -e], [0, 0, 1, 0, -f]])
	B = npA([x0, y0, z0, x2, y2, z2])
	for i in range(6):
		(inds := list(range(6))).remove(i)
		if np.linalg.det(A_sub := A[inds]).round(DIGIT_ACCU):
			sol = np.linalg.solve(A_sub, B[inds])
			if not (A[i].dot(sol) - B[i]).round(DIGIT_ACCU-2).any(): #There is one and only one solution (intersection)
				l, k = sol.round(DIGIT_ACCU-2)[-2:]
				inter_on1st, inter_on2nd = (1 if l in (0, 1) else (2 if 0 < l < 1 else 0)), (1 if k in (0, 1) else (2 if 0 < k < 1 else 0))
				return max(inter_on1st + inter_on2nd - 1, 0) #3 if cross on both insdies, 2 if one end is on the other's inside	return 0
	return 0
	

# Can handle parallel cases. coors[0] and coors[1] form a segment and coors[2] and coors[3] form the other
def check_segmentMeets_3CoorFace(c1, c2, coors):
	"""
	Return (inter, coor). inter -- intersection between a 2-node segment and a 3-node face
	coor -- coordinate (x, y, z, l) if infinite line and infinite plane meet, None if no intersection at all,  () if segment is same plane
		l is where the intersection is on the segment c0->c1, 0 < l < 1 if intersection is within segment
	"""
	if len(coors) != 3: raise Exception
	normal = None
	if is3D := (len(c1) == 3):
		##a line with vector (a, b, c) can be defined as x=x0+a*l, y=y0+b*l, c=c0+c*l
		# For a plane that passes (x2, y2, z2) with its normal being (d, e, f), the equation is d(x-x2)+e(y-y2)+f(z-z2)=0
		a, b, c = v12 = c2 - c1
		d, e, f = normal = np.cross(coors[1] - coors[0], coors[2] - coors[0])
		(x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = c1, c2, coors[0]
		if v12.dot(normal).round(DIGIT_ACCU): #Not parallel. Check the intersection relative to segment and triangle
			A = npA([[1, 0, 0, -a], [0, 1, 0, -b], [0, 0, 1, -c], [d, e, f, 0]])
			B = npA([x0, y0, z0, d * x2 + e * y2 + f * z2])
			x, y, z, l = vec = np.linalg.solve(A, B).round(DIGIT_ACCU)
			if l > 1 or l < 0: return False, None
			else: #Intersection is in the plane. #Intersection is segment's end #
				#Only False if intersection outside triangle or (intersection and segment end on vectices simultaneously)
				if (wrt_tri := is_coorinPoly_Convex(vec[:3], coors)) <= 1: return (wrt_tri == 1 and 0 < l < 1), vec
				else: return True, vec
		elif normal.dot(c1 - coors[0]).round(DIGIT_ACCU): return False, None #Parallel and off the triangle plane
	#Case is 2D or segment is in the triangle plane in 3D
	"""
					c2_wrt_tri
					0	1 	2	3	|	a: both ends outside triangle. Check intersection with each edge
	c1_wrt_tri	0	a	b	d	d	|	b: one end on a triangle vertex, the other outside. Check intersection with each edge
				1	b	c	d	d	|	c: both ends are triangle vertices. The segment is an edge
				2	d	d	d	d	|	d: any of the ends on edge or inside. This is overlapping
				3	d	d	d	d	|
	"""
	if normal is not None:
		(inds := [0, 1, 2]).remove(np.absolute(normal).argmax())
		c1, c2, coors = c1[inds], c2[inds], coors[:, inds]
	c1_wrt_tri, c2_wrt_tri = is_coorinTriangle(c1, coors), is_coorinTriangle(c2, coors)
	#print("wrt", c1_wrt_tri, c2_wrt_tri)
	if c1_wrt_tri >= 2 or c2_wrt_tri>= 2: return True, None
	elif c1_wrt_tri == c2_wrt_tri == 1: return False, None
	#case a and b. Need to check the intersection of the segment wrt each triangle edge.
	for inds in INDS_3CHOOSE2:
		#print("Check inter", npA([c1, c2]), "\n", coors[inds])
		#print(check_2SegmentsIntersect_within_2D(npA([c1, c2]), coors[inds]))
		if check_2SegmentsIntersect_within_2D(npA([c1, c2]), coors[inds]) > 1: return True, None
	return False, None
	#return any(check_2SegmentsIntersect_within_2D(npA([c1, c2]), coors[inds]) > 1 for inds in INDS_3CHOOSE2), None
	

def check_2Triangles_CutEachOther(coors1, coors2):
	if len(coors1) != 3 or len(coors2) != 3: raise Exception
	if any(not (coors1[list(inds)] - coors2).round(DIGIT_ACCU).any() for inds in permutations(range(3))): return False
	normal1 = normal2 = None
	if is3D := (len(coors1[0]) == 3):
		# For a plane that passes (x0, y0, z0) with its normal being (a, b, c), the equation is a(x-x0)+b(y-y0)+c(z-z0)=0
		# For a plane that passes (x3, y3, z3) with its normal being (d, e, f), the equation is d(x-x3)+e(y-y3)+f(z-z3)=0
		normal1, normal2 = np.cross(*(coors1[1:] - coors1[0])).round(DIGIT_ACCU), np.cross(*(coors2[1:] - coors2[0])).round(DIGIT_ACCU)
		if not normal1.any() or not normal2.any(): raise Exception
		A = npA([normal1, normal2])
		B = npA([normal1.dot(coors1[0]), normal2.dot(coors2[0])])
		if np.cross(normal1, normal2).round(DIGIT_ACCU).any(): #Two triangles not parallel
			tups = [] #pick two segments that can intersect with 2nd triangle, regardless of their pos wrt the 2nd triangle
			for i, j in INDS_3CHOOSE2:
				overlap, xyzl = check_segmentMeets_3CoorFace(coors1[i], coors1[j], coors2)
				if overlap: return True #If any intersection on 2nd plane
				elif xyzl is not None and 0 <= xyzl[3] <= 1: tups.append(tuple(xyzl[:3]))
			#print("Verify crossing", tups)
			if len(tups) < 2: return False #There must be at least 2 points that touch the 2nd plane.
			else:
				c_intersects = npA(list(set(tups))) #There are at most 2 different intersections
				(inds := [0, 1, 2]).remove(np.absolute(normal2).argmax())
				c_intersects, coors2 = c_intersects[:, inds], coors2[:, inds]
				#print("Verify triangle crossing", c_intersects, coors2)
				if len(c_intersects) <= 1: return is_coorinTriangle(c_intersects[0], coors2) > 1
				else: #Check if segment meets the triangle
					#print("check_segmentMeets_3CoorFace", c_intersects, coors2)
					#print(check_segmentMeets_3CoorFace(*c_intersects, coors2))
					return check_segmentMeets_3CoorFace(*c_intersects, coors2)[0]
		elif normal1.dot(coors2[0] - coors1[0]).round(DIGIT_ACCU): return False #Parallel but not same plane
	#2D case or two 3D triangles in same plane
	if normal2 is not None:
		(inds := [0, 1, 2]).remove(np.absolute(normal2).argmax())
		coors1, coors2 = coors1[:, inds], coors2[:, inds]
	if any(is_coorinTriangle(c, coors2) > 1 for c in coors1): return True
	elif any(is_coorinTriangle(c, coors1) > 1 for c in coors2): return True
	for inds1, inds2 in product(INDS_3CHOOSE2, INDS_3CHOOSE2):
		if check_2SegmentsIntersect_within_2D(coors1[inds1], coors2[inds2]) >= 2: return True
	return False
	
def check_Triangle_cuts_Tetrahedron(coors1, coors2):
	if len(coors1) != 3 or len(coors2) != 4: raise Exception
	if any(is_coorinTetrahedron(c, coors2) > 1 for c in coors1): return True
	#As long as triangle doesn't have vertex in tetra, then only need check 3 faces against the triangles
	#for inds2 in INDS_4CHOOSE3:
	#	print("Check face:", inds2)
	#	if check_2Triangles_CutEachOther(coors1, coors2[inds2]): return True
	#return False
	return any(check_2Triangles_CutEachOther(coors1, coors2[inds2]) for inds2 in INDS_4CHOOSE3)
	
def check_Tetras_Intersect(coors1, coors2):
	if len(coors1) != 4 or len(coors2) != 4: raise Exception
	if any(is_coorinTetrahedron(c, coors2) > 1 for c in coors1): return True
	elif any(is_coorinTetrahedron(c, coors1) > 1 for c in coors2): return True
	# Pick 3 out of 4 faces in coors1, and 3 out of 4 faces in coors2. Check if they cut each other
	for inds1, inds2 in product(INDS_4CHOOSE3, INDS_4CHOOSE3):
		if check_2Triangles_CutEachOther(coors1[inds1], coors2[inds2]): return True
	return False
		
def parallelVec_PlaneOAB_to_3Coors(arr_coors1, arr_coors2):
	normal = np.cross(*(arr_coors2[:2] - arr_coors2[2]))
	n_dot_OA, n_dot_OB = normal.dot((arr_coors1[1:] - arr_coors1[0]).T)
	print(normal, (arr_coors1[1:] - arr_coors1[0]), n_dot_OA, n_dot_OB)
	#
	
def rotateArr_betweenZandArbitrayVec(v):
	#Return two matrices. 1st mat is to rotate z+ to along v, 2nd is to rotate v to along z
	# First rotate Vec, so that Vec is in xy plane
	# Second rotate Vec, so that Vec is along z+ axis
	theta_1st_toXZ = np.arccos(v[0] / np.linalg.norm(v[:2])) * (1 if v[1] > 0 else -1)
	theta_2nd_toZ = np.arccos(v[2] / np.linalg.norm(v))
	a, b, c, d = np.cos(theta_1st_toXZ), np.sin(theta_1st_toXZ), np.cos(theta_2nd_toZ), np.sin(theta_2nd_toZ)
	arr_1st_toXZ, arr_back_1st_fromXZ = npA([[a, b, 0], [-b, a, 0], [0, 0, 1]]), npA([[a, -b, 0], [b, a, 0], [0, 0, 1]])
	arr_2nd_toZ, arr_back_2nd_fromZ = npA([[c, 0, -d], [0, 1, 0], [d, 0, c]]), npA([[c, 0, d], [0, 1, 0], [-d, 0, c]])
	return np.matmul(arr_back_1st_fromXZ, arr_back_2nd_fromZ), np.matmul(arr_2nd_toZ, arr_1st_toXZ)

def rotateArr_aboutArbitraryVec(v, theta):
	e, f = np.cos(theta), np.sin(theta)
	arr_3rd_aboutZ = npA([[e, -f, 0], [f, e, 0], [0, 0, 1]])
	if v[:2].any():
		arr_ztov, arr_vtoz = rotateArr_betweenZandArbitrayVec(v)
		return np.matmul(arr_ztov, np.matmul(arr_3rd_aboutZ, arr_vtoz))
	else: return arr_3rd_aboutZ if v[2] > 0 else arr_3rd_aboutZ.T
	
	
def arr_rotateabout2Coors(coors, theta):
	c0 = coors[0]
	arr_rotate = rotateArr_aboutArbitraryVec(coors[1]-c0, theta)
	return arr_rotate, -np.matmul(arr_rotate, c0) + c0


def getNormals_OutofTriangle(coors, normalize=False): #return normals to edges (outward), and normal to face (either direction)
	#ith n_edge is opposite to ith triangle vert
	if (dim:=len(coors[0])) != 3: coors = np.append(coors, np.zeros((3, 1)), axis=1)
	ns_edge = np.cross(coors[[2, 0, 1]] - coors[[1, 2, 0]], n_face := np.cross(*(coors[1:] - coors[0])))
	if dim == 2: ns_edge = ns_edge[:, :2]
	if normalize: return ns_edge / np.tile(np.linalg.norm(ns_edge, axis=1), (dim, 1)).T, n_face
	return ns_edge, n_face

def getNormals_OutofTetra(coors, normalize=False): #Get
	"""
	from c0: (c2-c1) x (c3-c1)  c0->c1
	from c1: (c3-c2) x (c0-c2)  c1->c2
	from c2: (c0-c3) x (c1-c3)  c2->c3
	from c3: (c1-c0) x (c2-c0)  c3->c0
	"""
	normals = np.cross(coors[[2, 3, 0, 1]]-coors[[1, 2, 3, 0]], coors[[3, 0, 1, 2]]-coors[[1, 2, 3, 0]])
	vecs = coors[[1, 2, 3, 0]] - coors
	normals = normals * np.tile((np.heaviside((normals*vecs).sum(axis=1), 1) * 2 - 1), (3, 1)).T
	if normalize: return normals / np.tile(np.linalg.norm(normals, axis=1), (3, 1)).T
	return normals
	
def check_coorinTetra(c0, coors):
	normals = np.cross(coors[[2, 3, 0, 1]] - coors[[1, 2, 3, 0]], coors[[3, 0, 1, 2]] - coors[[1, 2, 3, 0]])
	vecs = coors[[1, 2, 3, 0]] - coors
	normals = normals * np.tile((np.heaviside((normals * vecs).sum(axis=1), 1) * 2 - 1), (3, 1)).T
	v_verts_to_c = c0 - coors[[1, 2, 3, 0]]
	wrt = (normals*v_verts_to_c).sum(axis=1).round(DIGIT_ACCU)
	return not ((wrt > 0).any() or np.count_nonzero(wrt == 0) == 3)
	
	
def is_origin_in_Verts_2D(coors):
	#coors is (n, 2), n 2D vectors from origin. Get a 4x4 matrix, with each element being ith vector x jth vector
		#np.cross(np.tile(coors.reshape(-1, 1, 2), (1, len(coors), 1)), coors).round(DIGIT_ACCU)
	#Make sure origin (0, 0) is not included in the max convex polygon formed by the coors. (coors in the polygon allowed)
		#If origin is strictly outside, then matrix has a row with all positive values, but vi x vi (self cross product)
		#If origin is on a vertex, then matrix has an all-zero row
		#If origin is on an edge, then matrix has a row with >=2 zeros
	#Count the number of positive vi x vj on each row. If strictly outside, then num must be n-1
	mat_vi_x_vj = np.cross(np.tile(coors.reshape(-1, 1, 2), (1, n:=len(coors), 1)), coors).round(DIGIT_ACCU)
	if (np.count_nonzero(mat_vi_x_vj > 0, axis=1) == (n - 1)).any(): return 0
	elif not (mat_vi_x_vj >= 0).all(axis=1).any(): return 3
	elif (mat_vi_x_vj == 0).all(axis=1).any(): return 1
	else: return 2
	
#def verify_segmentMeets_3CoorFace(cs_seg, cs_tri, ns_edge=None, n_face=None):


def check_negX_cuts_SegmentsbetweenVerts_2D(cs):
	#Two vectors va and vb starting from origin. vc is -x axis. For segment ab to cut -x axis, they have to be on opposite sides of x axis,
		#If any vert is on -x axis(including origin), then simply return True
		#Rule out any vert that is +x axis (excluding origin)
	if ((cs[:,1] == 0) & (cs[:,0] <= 0)).any(): return True #If any c on -x axis (including origin)
	if len(cs:=cs[np.where(cs[:,1]!=0)]) < 2: return False
	inds1, inds2 = np.array(list(combinations(range(len(cs)), 2))).T
	if (a_b_acrossX := ((cs_a:=cs[inds1])[:,1] * (cs_b:=cs[inds2])[:,1] <= 0)).any(): #There must be a pair of coors on two side of x axis
		return ((cs_a[:,1] * np.cross(cs_a, cs_b) >= 0) & a_b_acrossX).any()
	return False

def allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra, ns_edge=None, n_face=None, inds_Edge2Check=None):
	if ns_edge is None or n_face is None: #With triangle 0->1->2, ns_edge will the normal to 1->2, 2->0, 0->1 (opposite to ith vert)
		ns_edge = np.cross(cs_tri[[2, 0, 1]] - cs_tri[[1, 2, 0]], n_face := np.cross(*(cs_tri[1:] - cs_tri[0])))
	if inds_Edge2Check is None: inds_Edge2Check = [0, 1, 2]
	#E.g., for inds_Edge2Check=[0, 1], check the projection of cs_tetra verts along edges opposite to tri vert 0&1
	#for n_edge, c0 in zip(ns_edge[inds_Edge2Check], cs_tri[npA([1, 2, 0])[inds_Edge2Check]]):
	#	print("??")
	#	print(wrt:=np.array([n_edge.dot((cs_tetra_wrt := cs_tetra - c0).T), n_face.dot(cs_tetra_wrt.T)]).T)
	#	print(check_negX_cuts_SegmentsbetweenVerts_2D(wrt))
	return all(check_negX_cuts_SegmentsbetweenVerts_2D(np.array([n_edge.dot((cs_tetra_wrt := cs_tetra - c0).T), n_face.dot(cs_tetra_wrt.T)]).T)
			   for n_edge, c0 in zip(ns_edge[inds_Edge2Check], cs_tri[npA([1, 2, 0])[inds_Edge2Check]]))

def verify_Triangle_cuts_Tetra(cs_tri, cs_tetra, ns_edge=None, n_face=None, ns_tetraFace=None):
	# Exam overlapping of verts. If 0 tri vert different from tetra verts, then simply True (tri is a face of tetra)
	# v_tetraVerts_2_triVerts is (3, 4, 3). Each ij is from vector from jth tetra vert to ith tri vert
	v_ithTriV_from_jthTetraV = (np.tile(cs_tri.reshape(-1, 1, 3), (1, 4, 1)) - cs_tetra).round(DIGIT_ACCU - 1)
	isDiff_ithTri_jthTetra = v_ithTriV_from_jthTetraV.any(axis=2) #matrix ij True if ith tri vert is same as jth tetra vert
	if isDiff_ithTri_jthTetra.all(axis=1).any(): #There are tri verts that don't coincide with tetra verts
		#Prepare the normals of triangle's edges and face
		if ns_edge is None or n_face is None:
			ns_edge = np.cross(cs_tri[[2, 0, 1]] - cs_tri[[1, 2, 0]], n_face := np.cross(*(cs_tri[1:] - cs_tri[0])))
		inds_tri_diff = np.nonzero(isDiff_ithTri_jthTetra.all(axis=1))[0]  # Get the inds of tri verts that are not also tetra verts
		inds_tetra_diff = np.nonzero(isDiff_ithTri_jthTetra.all(axis=0))[0]  # Get the inds of tetra verts that are not also tri verts
		# If tri shares 1 EDGE with tetra, then only check if THIS SINGLE TRI EDGE's half plane cuts the segment between tetra's other two verts
		if (n := len(inds_tri_diff)) == 1:
			return allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra[inds_tetra_diff], ns_edge, n_face, inds_Edge2Check=inds_tri_diff)
		elif n == 2: #Tri and tetra share 1 vert. Need to check which octant two vects are
			i_tri_same = next(i for i in (0, 1, 2) if i not in inds_tri_diff)
			i_tetra_same = next(i for i in (0, 1, 2, 3) if i not in inds_tetra_diff)
			sol1, sol2 = sol = np.linalg.solve((offsets_2otherTetraVs := (cs_tetra[inds_tetra_diff] - cs_tetra[i_tetra_same])).T,
								  (cs_tri[inds_tri_diff] - cs_tri[i_tri_same]).T).T.round(DIGIT_ACCU)
			if (sol >= 0).all(axis=1).any(): return True #If either vert is in 1st octant (including face)
			elif (sol < 0).all(axis=0).any(): return False #If two verts avoid 1st octant completely
			# At this point, segment must cross xy, xz or yz plane. Get the corresponding intersection coor,
				#and make sure the 2-d coor is in 1st quadrant (including axes).
			elif not (crossing:=(sol1 * sol2 < 0)).any(): return False
			i_cross = np.nonzero(crossing)[0][0]
			return (np.delete(sol1 - (sol1[i_cross] / (sol2-sol1)[i_cross]) * (sol2-sol1), i_cross) >= 0).all()
		else: #There is not shared verts. Any touching is seens as cutting
			# Prepare normals of tetras 4 faces. Get the distance of each tri vert wrt each of the normals
			if ns_tetraFace is None: ns_tetraFace = getNormals_OutofTetra(cs_tetra)
			triV_out_tetraN = (np.roll(v_ithTriV_from_jthTetraV, -1, axis=1) * np.tile(ns_tetraFace, (3, 1, 1))).sum(axis=2).round(DIGIT_ACCU)
			#To avoid cutting, tri should be outside one tetra face plane or tetra is not cut by >=1 tri's edge's half plane.
			#print("triV_out_tetraN > 0", triV_out_tetraN > 0, (triV_out_tetraN > 0).all(axis=0).any())
			#if (triV_out_tetraN > 0).all(axis=0).any(): return False
			#return allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra[inds_tetra_diff], ns_edge, n_face, inds_Edge2Check=inds_tri_diff)
			return not (triV_out_tetraN > 0).all(axis=0).any() \
				   and allTriHalfPlanes_touch_Tetra(cs_tri, cs_tetra[inds_tetra_diff], ns_edge, n_face, inds_Edge2Check=inds_tri_diff)
	return False #If all tri verts are also tetra verts, then tri is simply a face of tetra
	
	
def verify_triV_to_tetraN(v_ithTriV_from_jthTetraV, cs_tetra, ns_tetraFace):
	if ns_tetraFace is None: ns_tetraFace = getNormals_OutofTetra(cs_tetra)
	wrt_triV_to_tetraN = (np.roll(v_ithTriV_from_jthTetraV, -1, axis=1) * np.tile(ns_tetraFace, (3, 1, 1))).sum(axis=2).round(DIGIT_ACCU)
	if (wrt_triV_to_tetraN > 0).all(axis=0).any(): return False