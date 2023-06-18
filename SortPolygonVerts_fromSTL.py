from GeometryFuncs import *

def checkif2D_getXYZIdx(stl_mesh):
	#return the dimensionality of mesh, and the idx of x,y,z dim used.
	xs, ys, zs = stl_mesh.vectors.reshape(-1, 3).round(5).T
	allSame_xyz = ((xs == xs[0]).all(), (ys == ys[0]).all(), (zs == zs[0]).all())
	if (n := sum(allSame_xyz)) == 0: return 3, [0, 1, 2]
	elif n == 1:
		dim, idx = 2, [0, 1, 2]
		idx.remove(allSame_xyz.index(True))
		return dim, idx
	else:
		print("Nodes must be either on xy/yz/xz planes or distributed in 3D")
		raise ValueError


class Poly_CircumExtraction:
	def __init__(self, pts, conns, tri):
		# pts: list, conns: set. Get copies of pts & conns, they will be modified later
		self.pts, self.conns = pts[:], conns.copy()
		self.normal, self.tris = tri.normal / npNorm(tri.normal), {tri}  # conns, tris are both sets
		self.dim = tri.coors.shape[-1]
		self.pts_split, self.n_edges = [], []
		
	def __repr__(self):
		pts = sorted(self.pts, key=lambda pt: pt.i)
		return "Poly-" + '-'.join(str(pt.i) for pt in pts)
		
	def sharesConn_inSamePlane(self, tri):
		conn = next(iter(self.conns & tri.conns))
		if self.dim == 2: return True
		(pt1, pt2), pt_other = conn.pts, next(iter(tri.pts - conn.pts))
		return np.round(self.normal @ (pt_other.coor - pt1.coor), 5) == 0
		
	def consumeaTri(self, tri, conn=None):
		if not conn:
			conn = next(iter(tri.conns & self.conns))
		self.conns |= tri.conns
		self.conns.discard(conn)
		
		(p1, p2), p3 = conn.pts, next(iter(tri.pts - conn.pts))
		p1.conns.discard(conn)
		p2.conns.discard(conn)
		i, j = self.pts.index(p1), self.pts.index(p2)
		if i == j + 1: self.pts.insert(i, p3)
		elif i == j - 1: self.pts.insert(j, p3)
		else: self.pts.append(p3)
		self.tris.add(tri)
	
class Point_CircumExtraction:
	def __init__(self, i, coor):
		self.i, self.coor = i, coor
		self.conns, self.triangles = set(), set()
	
	def conn_Sharedwith(self, pt):
		return next(iter(self.conns & pt.conns), None)
	
	@property
	def nextPt_Connected(self):
		return next(iter(pt for conn in self.conns for pt in conn.pts if pt is not self))
	
	def __repr__(self): return "Pt%d" % self.i


"""
Take in array of 2D triangles, shape(n, 3, 2/3). These triangles come from stl reading
sort the vertices of the polygon into 1 direction (clockwise or counter-clockwise).
return (m, 2/3) array
"""
def extract_sort_verts_boundaries(cs_tris, onlyExtractVertsTris=False):
	#Poorly-designed stl models can have 0-area triangles, which typically arise from unnecessary verts within edges or planes
	vecs_Tri = cs_tris[:, 1:] - cs_tris[:, 0, np.newaxis] #(m, 2, 2/3) v12 & v13 in each triangle
	dim = vecs_Tri.shape[-1]
	bools_zeroTriArea = npNorm(np.cross(vecs_Tri[:,0], vecs_Tri[:,1]), axis=dim-2).round(5) == 0
	if (idx_zeroTriArea := np.where(bools_zeroTriArea)[0]).size:
		print("There are triangles with zero area:", idx_zeroTriArea, "\nTERMINATE")
		raise ValueError
	
	#Establish the data structure: triangles->connections->points
	#Later let one triangle expand and consume neighboring triangles, and eventually form the polygon.
	class Connection_CircumExtraction:
		def __init__(self, pt1, pt2):
			self.pts = {pt1, pt2}
			self.triangles = set()
			self.polies = []
			
		def __repr__(self):
			pt1, pt2 = self.pts
			if pt1.i > pt2.i: pt1, pt2 = pt2, pt1
			return "C%d-%d" % (pt1.i, pt2.i)
	
	
	class Triangle_CircumExtraction:
		def __init__(self, pt1, pt2, pt3):
			self.pts = {pt1, pt2, pt3}
			self.conns = set()
			for pt in self.pts: pt.triangles.add(self)
			self.coors = np.array([pt1.coor, pt2.coor, pt3.coor])
			self.normal = np.cross(pt2.coor - pt1.coor, pt3.coor - pt1.coor)
			
		@property
		def tris_sharingConn(self):
			return {tri for conn in self.conns for tri in conn.triangles} - {self}
		
		def __repr__(self):
			pt1, pt2, pt3 = sorted(self.pts, key=lambda pt: pt.i)
			return "Tri%d-%d-%d" % (pt1.i, pt2.i, pt3.i)
	
	dim = cs_tris.shape[-1]
	cs_verts = np.unique(cs_tris.reshape(-1, dim), axis=0)
	tris, conns = [], set()
	dict_c_to_points = {tuple(c): Point_CircumExtraction(i, c) for i, c in enumerate(cs_verts)}
	for c1, c2, c3 in cs_tris:
		pt1, pt2, pt3 = dict_c_to_points[tuple(c1)], dict_c_to_points[tuple(c2)], dict_c_to_points[tuple(c3)]
		tris.append(tri := Triangle_CircumExtraction(pt1, pt2, pt3))
		for p1, p2 in ((pt1, pt2), (pt2, pt3), (pt1, pt3)):
			if conn := p1.conn_Sharedwith(p2):  # Existing conn between p1 and p2
				tri.conns.add(conn)
			else:
				tri.conns.add(conn := Connection_CircumExtraction(p1, p2))
				p1.conns.add(conn)
				p2.conns.add(conn)
			conn.triangles.add(tri)
			conns.add(conn)
	
	if onlyExtractVertsTris:
		#Only return distinct points and triangles (no sorting)
		return [list(dict_c_to_points.values())], tris
	
	
	"""
	Start merging all triangles into one polygon. Pick one as the seed.
	Constantly check the polygon's edges and consume neighboring triangles.
	"""
	tris_all, tri_2Grow = tris[:], tris.pop()
	poly = Poly_CircumExtraction(list(tri_2Grow.pts), tri_2Grow.conns, tri_2Grow)
	while tris:
		for conn in poly.conns.copy():
			if set_otherTri := conn.triangles - poly.tris:
				tri = next(iter(set_otherTri))
				poly.consumeaTri(tri, conn=conn)
				tris.remove(tri)
	
	"""
	If there is hollow interior, there would be points with >2 connections. Detach connect at these pts
	Try to separate exterior and interior
	"""
	counts, pts = {}, poly.pts
	for pt in pts:
		if pt in counts: counts[pt] += 1
		else: counts[pt] = 1
	pts_overlap = [pt for pt, num in counts.items() if num > 1]
	for pt in pts_overlap:
		print("Touch point: {} with {} conns".format(pt.i, len(pt.conns)))
	#Generally pts with >2 connections will be connected to >=1 pts with >2 connections
	#Interior circles will only have 1 such pt, but exterior can have multiple of these also connected to each other.
	#Such connected
	for pt in pts_overlap:
		pts_2remove = {p for conn in pt.conns for p in conn.pts if p is not pt and len(p.conns) > 2}
		if len(pts_2remove) > 1:
			print(pt, "Shouldn't exceed length 1:", pts_2remove)
		elif pts_2remove:
			p_other = next(iter(pts_2remove))
			pt.conns.discard(conn := pt.conn_Sharedwith(p_other))
			p_other.conns.discard(conn)
	
	"""Separated exterior and interior will be saved in different lists."""
	c_cen = np.average(cs_verts := npA([pt.coor for pt in pts]), axis=0)
	pt_0 = pts[npNorm(cs_verts - c_cen, axis=1).argmax()]
	i, ls_pts_onBoundaries, pts = 0, [], set(pts)
	while pts:
		if i: ls = [pt_0 := next(iter(pts))]
		else: ls = [pt_0]
		pt_cur = pt_0
		print("Start from pt_cur", pt_cur, pt_cur.conns, [pt for conn in pt_cur.conns for pt in conn.pts])
		while (pt_next := pt_cur.nextPt_Connected) is not pt_0:
			pt_cur.conns.discard(conn := pt_cur.conn_Sharedwith(pt_next))
			pt_next.conns.discard(conn)
			ls.append(pt_next)
			pt_cur = pt_next
		ls_pts_onBoundaries.append(ls)
		pts -= set(ls)
		i += 1
	# first list is the exterior, e.g. [exterior, interior1, interior2...]
	# tris_all is all triangles included in the STL model
	return ls_pts_onBoundaries, tris_all


def getVecs_for1Plane_sortedOuterInner(ls_pts_onBoundaries, height=None):
	ls_cs, ls_normals = [], []
	
	for i, pts in enumerate(ls_pts_onBoundaries):
		pt1_ext, pt2_ext = pts[:2]
		tri = next(iter(pt1_ext.triangles & pt2_ext.triangles))
		pt3 = next(iter(tri.pts - {pt1_ext, pt2_ext}))
		
		ls_cs.append(cs_ext := npA([p.coor for p in pts]))
		vs_ext = np.roll(cs_ext, shift=-1, axis=0) - cs_ext # 0->1, 1->2 ... -1->0
		v12, v13 = vs_ext[0], pt3.coor - pt1_ext.coor
		
		if (dim := v12.shape[0]) == 2:
			arr = np.array([[0, -1], [1, 0]])
			if (arr @ v12) @ v13 < 0: arr = -arr
			normals = (arr @ vs_ext.T).T
		else:
			normal_face = np.cross(v12, v13)
			if np.cross(normal_face, v12) @ v13 < 0: normal_face = -normal_face
			normals = np.cross(normal_face, vs_ext)
		normals /= np.tile(npNorm(normals, axis=1)[:, np.newaxis], (1, dim))
		ls_normals.append(normals)
	
	if height is None: return ls_cs, ls_normals
	else:
		ls_cs_split, ls_normals_split = [], []
		for cs, normals in zip(ls_cs, ls_normals):
			cs_split, normals_split = [], []
			for c1, c2, normal in zip(cs, np.roll(cs, -1, axis=0), normals):
				c1c2_split = splitSegment_VarLength(c1, c2, height) #array
				cs_split.append(c1c2_split[:-1])
				normals_split += [normal] * (len(c1c2_split) - 1)
			ls_cs_split.append(npA([c for coors in cs_split for c in coors]))
			ls_normals_split.append(npA(normals_split))
		return ls_cs_split, ls_normals_split

#Return the coordinates, vectors, normals of stl of 3D model
def getVecs_sorted_3D(cs_tris):
	ls_pts_onBoundaries, tris = extract_sort_verts_boundaries(cs_tris, onlyExtractVertsTris=True)
	ls_polies, pts = [], ls_pts_onBoundaries[0]
	
	while tris:
		tri_2Grow = tris.pop()
		poly = Poly_CircumExtraction(list(tri_2Grow.pts), tri_2Grow.conns, tri_2Grow)
		#print("Start as poly", poly)
		while True:
			tris_neighbors = {tri for conn in poly.conns for tri in conn.triangles} - poly.tris
			found1Tri_2consume = False
			for tri in tris_neighbors:
				if poly.sharesConn_inSamePlane(tri):
					found1Tri_2consume = True
					poly.consumeaTri(tri)
					tris.remove(tri)
			if not found1Tri_2consume: break
		ls_polies.append(poly)
	
	for poly in ls_polies:
		pts = poly.pts
		c_cen = np.average(cs_verts := npA([pt.coor for pt in pts]), axis=0)
		pt_0 = pts[npNorm(cs_verts - c_cen, axis=1).argmax()]
		pt_cur = next(iter(pt for conn in pt_0.conns for pt in conn.pts if pt in pts and pt is not pt_0))
		ls = [pt_0]
		
		while True:
			tups = (pt_0, pt_cur, ls[-1])
			if pt_next := next(iter(pt for conn in pt_cur.conns for pt in conn.pts if pt in pts and pt not in tups), None):
				ls.append(pt_cur)
				pt_cur = pt_next
			else: break
		ls.append(pt_cur)
		
	ls_cs, ls_vecs, ls_normals = [], [], []
	for poly in ls_polies:
		pts = poly.pts
		pt1_ext, pt2_ext = pts[:2]
		tri = next(iter(pt1_ext.triangles & pt2_ext.triangles & poly.tris))
		pt3 = next(iter(tri.pts - {pt1_ext, pt2_ext}))
		cs_ext = npA([p.coor for p in pts])
		vs_ext = np.roll(cs_ext, shift=-1, axis=0) - cs_ext  # 0->1, 1->2 ... -1->0
		v12, v13 = vs_ext[0], pt3.coor - pt1_ext.coor
		
		normal_face = np.cross(v12, v13)
		if np.cross(normal_face, v12) @ v13 < 0: normal_face = -normal_face
		normals = np.cross(normal_face, vs_ext)
		normals /= np.tile(npNorm(normals, axis=1)[:, np.newaxis], (1, 3))
		ls_normals.append(normals)
	
	return ls_normals, ls_polies
	

def getVecsPolies_splitConns_fromSTL(cs_tris, height=None):
	ls_normals, ls_polies = getVecs_sorted_3D(cs_tris)
	if height is None:
		for poly, normals in zip(ls_polies, ls_normals): poly.n_edges = normals
		return ls_polies
	
	i, dict_conn_to_csMiddle = len({pt for poly in ls_polies for pt in poly.pts}), {}
	for poly, n_edges in zip(ls_polies, ls_normals):
		pts = poly.pts
		for pt1, pt2, n_edge in zip(pts, pts[1:] + [pts[0]], n_edges):
			if (pt1, pt2) in dict_conn_to_csMiddle:
				pts_middle = dict_conn_to_csMiddle[(pt1, pt2)]
			else:
				conn, c1, c2 = pt1.conn_Sharedwith(pt2), pt1.coor, pt2.coor
				cs = splitSegment_VarLength(c1, c2, height)
				pts_middle = [Point_CircumExtraction(i + j, c) for j, c in enumerate(cs[1:-1])]
				dict_conn_to_csMiddle[(pt1, pt2)] = pts_middle
				dict_conn_to_csMiddle[(pt2, pt1)] = pts_middle[::-1]
				i += len(pts_middle)
			poly.pts_split += [pt1] + pts_middle
			poly.n_edges += [n_edge] * (len(pts_middle) + 1)
		poly.n_edges = npA(poly.n_edges)
		
	return ls_polies


"""
Check splitting 2D layout.
"""
if __name__ == "__main__":
	from PlotMethods import *
	import stl
	
	stl_mesh = stl.mesh.Mesh.from_file('2DModel.stl')
	dim, idx = checkif2D_getXYZIdx(stl_mesh)  # [0, 1]: x-y plane, [0, 2]: x-z plane, [1, 2]: y-z plane, [0, 1, 2]: 3D
	cs_tris = stl_mesh.vectors[...,:dim]
	
	ls_pts_onBoundaries, tris = extract_sort_verts_boundaries(cs_tris)
	ls_cs, ls_normals = getVecs_for1Plane_sortedOuterInner(ls_pts_onBoundaries, height=0.3)
	
	if (dim := cs_tris.shape[-1]) == 2:
		fig, ax = plt.subplots(dpi=200)
	else: ax = plt.figure(dpi=200).add_subplot(111, projection='3d')
	
	for i, ls_pts in enumerate(ls_pts_onBoundaries):
		if i:
			for pt in ls_pts:
				ax.text(*pt.coor, str(pt.i))
				ax.scatter(*pt.coor, s=2)
		else:
			for pt in ls_pts:
				ax.text(*pt.coor, str(pt.i), color="black")
				ax.scatter(*pt.coor, s=2, color="black")
	
	if dim == 2:
		for cs, normals in zip(ls_cs, ls_normals):
			for n, c1, c2 in zip(normals, cs, np.roll(cs, -1, axis=0)):
				ax.arrow(*(c1+c2)/2, *n/20)
			ax.set_aspect(1)
	else:
		for cs, normals in zip(ls_cs, ls_normals):
			ax.quiver(*((cs+np.roll(cs, -1, axis=0))/2).T, *(normals/8).T)
			
	plt.show()