from GeometryFuncs import *

def getNormalsof1Vec_3D(vec, first_both_comb=1):
	n1 = vec[[1, 2, 0]] - vec[[2, 0, 1]] #Would return [0, 0, 0] for a=b=c. In that case, get (-b, a, 0)
	if not n1.round(DIGIT_HEIGHT).any(): n1 = arr_90deg_3D @ vec
	if first_both_comb:
		n1, n2 = n1 / npNorm(n1), (n2 := np.cross(vec, n1)) / npNorm(n2)
		return (n1, n2) if first_both_comb == 1 else np.append(n1[npNuax], n2[npNuax], axis=0)
	else: return n1 / npNorm(n1)
	
def getNormalsofVecs_3D(vecs, first_both_comb=1):
	n1s = vecs[:,[1, 2, 0]] - vecs[:,[2, 0, 1]] #Would return [0, 0, 0] for a=b=c. In that case, get (-b, a, 0)
	if (idx_retry := np.nonzero(~n1s.round(DIGIT_HEIGHT).any(axis=1))[0]).size:
		n1s[idx_retry] = (arr_90deg_3D @ vecs[idx_retry].T).T
	if first_both_comb:
		n1s, n2s = n1s / npNorm(n1s, axis=1)[:,npNuax], (n2s := np.cross(vecs, n1s)) / npNorm(n2s, axis=1)[:,npNuax]
		return (n1s, n2s) if first_both_comb == 1 else np.append(n1s[:,npNuax], n2s[:,npNuax], axis=1)
	else: return n1s / npNorm(n1s, axis=1)[:,npNuax]
	
idx1_2C2, idx2_2C2 = [0], [1]
idx1_3C2, idx2_3C2 = [0, 0, 1], [1, 2, 2]
idx1_4C2, idx2_4C2 = [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]

def verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs, idx1, idx2, xs_ijk=None, ys_ijk=None):
	if len(projs.shape) != 4: raise Exception #projs.shape (m, nV, nHalfPlane, 2)
	if xs_ijk is None: xs_ijk = projs[...,0]
	if ys_ijk is None: ys_ijk = projs[...,1]
	#c is the index of connecting segs combinations
	ys_ijk = ys_ijk.round(DIGIT_HEIGHT)
	anyOnNegX_i = ((xs_ijk.round(DIGIT_HEIGHT) <= 0) & (ys_ijk == 0)).all(axis=2).any(axis=1)  # (m, )
	segCrossX_ic = ys_ijk[:,idx1,0] * ys_ijk[:,idx2,0] < 0 # any connecting seg crosses x axis (m, len(idx1))
	anyMightCrossNegX_ik = ys_ijk[:, idx1] * np.cross(projs[:, idx1], projs[:, idx2]).round(DIGIT_HEIGHT) >= 0  # (m, len(idx1), nEgs)
	return anyOnNegX_i | (segCrossX_ic[..., npNuax] & anyMightCrossNegX_ik).all(axis=2).any(axis=1)
	
def verify_origininPolygon_projs_ijkl_iObj_jV_lxy(projs):
	if len(projs.shape) != 3: raise Exception #projs.shape (m, nV, 2)
	crossProds_ijj = np.cross(projs[:,npNuax], projs[:,:,npNuax]).round(DIGIT_HEIGHT) # (m, nV, nV) doesn't matter whether it's transposed or not
	dotProds_ijj = (projs[:,np.newaxis] * projs[:,:,np.newaxis]).sum(axis=3).round(DIGIT_HEIGHT)
	return ~((crossProds_ijj > 0) | ((crossProds_ijj == 0) & (dotProds_ijj > 0))).all(axis=2).any(axis=1)
	
"""Check if single segment cuts multiple other objects"""
def check_Seg_cuts_Segs(cs_seg0, cs_segs, c_cen_segs=None, r_segs=None, any_not_idx=True):
	# cs_segs.shape (n, 2, 3) n segments, each with 2 ends, 2D or 3D, cs_seg.shape (2, 3)
	c_cen_seg, r = np.average(cs_seg0, axis=0), npNorm(cs_seg0[1] - cs_seg0[0]) / 2
	if c_cen_segs is None or r_segs is None:
		c_cen_segs, r_segs = np.average(cs_segs, axis=1), npNorm(cs_segs[:, 0] - cs_segs[:, 1], axis=1) / 2
	
	idx_closeEnough = np.nonzero(npNorm(c_cen_segs - c_cen_seg, axis=1) <= 1.01 * (r + r_segs))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	dim, cs_Segs_close = len(cs_seg0[0]), cs_segs[idx_closeEnough]
	#v_ijk_SegSegVSegV is the displacement from kth seg0V to jth segV of ith seg (m, 2, 2, 2/3)
	v_ijk_SegSegVSeg0V = cs_Segs_close[:, :, npNuax] - cs_seg0
	diff_ijk_SegSegVSeg0V = v_ijk_SegSegVSeg0V.round(DIGIT_HEIGHT).any(axis=3)  # (m, 2, 2)
	nDiffV_ithSeg = np.count_nonzero(diff_ijk_SegSegVSeg0V.all(axis=2), axis=1)  # (m, ) how many different verts ith seg has
	if (nDiffV_ithSeg == 0).all(): return False if any_not_idx else []
	
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share1V := np.nonzero(nDiffV_ithSeg == 1)[0]).size:
		idx_segV_diff = np.nonzero(diff_ijk_SegSegVSeg0V[idx_Share1V].all(axis=2))[1] #(m, )
		idx_seg0V_diff = np.nonzero(diff_ijk_SegSegVSeg0V[idx_Share1V].all(axis=1))[1] #(m, )
		v_seg0_i = cs_seg0[idx_seg0V_diff] - cs_seg0[1-idx_seg0V_diff] #(m, 3)
		v_seg0_i = v_seg0_i / npNorm(v_seg0_i, axis=1)[:,npNuax] #(m, 3) normalized
		v_seg_i = cs_Segs_close[idx_Share1V, idx_segV_diff] - cs_seg0[1-idx_seg0V_diff] #(m, 3)
		#Segs share 1V must be pointing in same direction in order to cut each other
		crossProds_eq0 = np.cross(v_seg0_i, v_seg_i).round(DIGIT_HEIGHT) == 0 #(m, ) if 2D else (m, 3)
		dotProds_pos = (v_seg0_i * v_seg_i).sum(axis=1).round(DIGIT_HEIGHT) > 0 #(m, )
		cutSeg0_i = (crossProds_eq0 & dotProds_pos) if dim == 2 else (crossProds_eq0.all(axis=1) & dotProds_pos)
		if any_not_idx:
			if cutSeg0_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][np.nonzero(cutSeg0_i)[0]])
	
	if (idx_0Share := np.nonzero(nDiffV_ithSeg == 2)[0]).size:
		cs_segs_0Share = cs_Segs_close[idx_0Share]
		#ijk: ith seg, jth segV, kth seg0V. Two segs project onto each other
		v_seg0, v_segs = cs_seg0[1] - cs_seg0[0], cs_segs_0Share[:, 1] - cs_segs_0Share[:, 0]
		if dim == 2:
			n1_seg0, n1_segs = arr_90deg @ v_seg0, (arr_90deg @ v_segs.T).T #n1_seg0.shape (2,) , n1_segs.shape (m, 2)
			n1_seg0, n1_segs = n1_seg0 / npNorm(n1_seg0), n1_segs / npNorm(n1_segs)
			xs_onSeg0_ij = (v_ijk_SegSegVSeg0V[idx_0Share,:,0] * n1_seg0).sum(axis=2).round(DIGIT_HEIGHT)
			xs_onSeg_ij = (-v_ijk_SegSegVSeg0V[idx_0Share,0] * np.tile(n1_segs[:,npNuax], (1, 2, 1))).sum(axis=2).round(DIGIT_HEIGHT)
			#xs_onSeg0_ij.shape (m, 2) xs_onSeg_ij.shape(m, 2)
			onSameLine_i = (xs_onSeg0_ij == 0).all(axis=1) #(m, )
			wrt_seg0_ij = (v_seg0 / npNorm(v_seg0) * v_ijk_SegSegVSeg0V[idx_0Share]).sum(axis=3) #(m, 2, 2) Check if jth segV of ith seg is along v_seg0
			twoSegsOutEachOther = (xs_onSeg0_ij > 0).all(axis=1) | (xs_onSeg0_ij < 0).all(axis=1) | (xs_onSeg_ij > 0).all(axis=1) | (xs_onSeg_ij < 0).all(axis=1)
			cutSeg0_i = twoSegsOutEachOther | (onSameLine_i & ((wrt_seg0_ij[:,1] > 0).all(axis=1) | (wrt_seg0_ij[:,0] < 0).all(axis=1)))
			if any_not_idx:
				if cutSeg0_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][np.nonzero(cutSeg0_i)[0]])
		else:
			n12_seg0 = getNormalsof1Vec_3D(v_seg0, first_both_comb=2) # (2, 3)
			n12_segs = getNormalsofVecs_3D(v_segs, first_both_comb=2) #(m, 2, 3)
			#projs_onSeg0_ij.shape (m, 2, 2) projection of ith seg jth segV on seg0
			#projs_ofSeg0_ik.shape (m, 2, 2) projection of kth seg0V on ith seg
			projs_onSeg0_ij = (np.tile(v_ijk_SegSegVSeg0V[idx_0Share,:,0][:,:,npNuax], (1, 1, 2, 1)) * n12_seg0).sum(axis=3)
			projs_ofSeg0_ik = (np.tile(v_ijk_SegSegVSeg0V[idx_0Share,  0][:,:,npNuax], (1, 1, 2, 1)) * np.tile(n12_segs[:,npNuax], (1, 2, 1, 1))).sum(axis=3)
			#crossProds_onSeg0.shape (m, ), dotProds_onSeg0.shape (m ,)
			crossProds_onSeg0, dotProds_onSeg0 = np.cross(projs_onSeg0_ij[:,0], projs_onSeg0_ij[:,1]).round(DIGIT_HEIGHT), (projs_onSeg0_ij[:, 0] * projs_onSeg0_ij[:, 1]).sum(axis=1).round(DIGIT_HEIGHT)
			crossProds_ofSeg0, dotProds_ofSeg0 = np.cross(projs_ofSeg0_ik[:,0], projs_ofSeg0_ik[:,1]).round(DIGIT_HEIGHT), (projs_ofSeg0_ik[:, 0] * projs_ofSeg0_ik[:, 1]).sum(axis=1).round(DIGIT_HEIGHT)
			onSameLine_i = (projs_onSeg0_ij.round(DIGIT_HEIGHT) == 0).all(axis=(1, 2))  # (m, )
			twoSegLinesTouch_i = (crossProds_onSeg0 == 0) & (dotProds_onSeg0 <= 0) & (crossProds_ofSeg0 == 0) & (dotProds_ofSeg0 <= 0)
			wrt_seg0_ij = (v_seg0 / npNorm(v_seg0) * v_ijk_SegSegVSeg0V[idx_0Share]).sum(axis=3).round(DIGIT_HEIGHT)  # (m, 2, 2) Check if jth segV of ith seg is along v_seg0
			onSameLine_NoOverlap_i = (onSameLine_i & ((wrt_seg0_ij[:,1] > 0).all(axis=1) | (wrt_seg0_ij[:,0] < 0).all(axis=1)))
			cutSeg0_i = ~onSameLine_NoOverlap_i & twoSegLinesTouch_i
			if any_not_idx:
				if cutSeg0_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][np.nonzero(cutSeg0_i)[0]])
	return False if any_not_idx else idx_cut_rootLevel


def check_Seg_cuts_Tris(cs_seg, cs_tris, c_cen_tris=None, r_tris=None, ns_edges=None, ns_face=None, any_not_idx=True):
	# cs_seg.shape (2, 2/3), each with 2 ends, 2D or 3D, cs_tri.shape (n, 3, 2/3)
	c_cen_seg, r = np.average(cs_seg, axis=0), npNorm(cs_seg[1] - cs_seg[0]) / 2
	if c_cen_tris is None or r_tris is None:
		c_cen_tris = np.average(cs_tris, axis=1)  # (m, 2/3)
		r_tris = npNorm(cs_tris - np.tile(c_cen_tris[:, npNuax], (1, 3, 1)), axis=2).max(axis=1)  # (m, )
	idx_closeEnough = np.nonzero(npNorm(c_cen_tris - c_cen_seg, axis=1) <= 1.01 * (r + r_tris))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	dim, cs_Tris_close = len(cs_seg[0]), cs_tris[idx_closeEnough]  # (m, 3, 2/3)
	# v_ijk_TriTriVSegV is (m, 3, 2, 2/3). Each ijk is displacement from kth seg vert to jth tri vert of ith tri
	v_ijk_TriTriVSegV = np.tile(cs_Tris_close[:, :, npNuax], (1, 1, 2, 1)) - cs_seg
	diff_ijk_TriTriVSegV = v_ijk_TriTriVSegV.round(DIGIT_HEIGHT).any(axis=3)  # (m, 3, 2)
	nDiffV_ithTri = np.count_nonzero(diff_ijk_TriTriVSegV.all(axis=2), axis=1)  # (m, ) how many different verts ith triangle has
	if (nDiffV_ithTri == 1).all(): return False if any_not_idx else []
	
	if ns_edges is None or ns_face is None:
		vs_tri = cs_Tris_close[:, 1:] - cs_Tris_close[:, [0, 0]]  # (m, 2, 2/3)
		ns_face = np.cross(vs_tri[:, 0], vs_tri[:, 1])  # (m, 3)
		ns_edges = np.cross(cs_Tris_close[:, [2, 0, 1]] - cs_Tris_close[:, [1, 2, 0]], np.tile(ns_face[:, npNuax], (1, 3, 1)))
		ns_face, ns_edges = ns_face / npNorm(ns_face, axis=1)[:,npNuax], ns_edges / np.tile(npNorm(ns_edges, axis=2)[..., npNuax], (1, 1, dim))
	
	if dim == 3: wrt_triN_ij_TriSegV = (v_ijk_TriTriVSegV[:, 0] * np.tile(ns_face[:, npNuax], (1, 2, 1))).sum(axis=2).round(DIGIT_HEIGHT)  # (m, 2)
	else: wrt_triN_ij_TriSegV = np.zeros((len(cs_tris), 2))  # For 2D, segVs are always in the same plane as triangles
	# Exam those that share 1 vert with tri. Segs that have verts out of tri plane are seen as not cutting
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share1V_IP := np.nonzero((nDiffV_ithTri == 2) & (wrt_triN_ij_TriSegV == 0).all(axis=1))[0]).size:
		# idx_ithSeg_diff get the indices of seg verts that are't shared for each segment, will be a 1-D array of 0/1
		# idx_triV_diff is the indices of tri verts that aren't shared for each segment, (m, 2)
		idx_segV_diff = np.nonzero(diff_ijk_TriTriVSegV[idx_Share1V_IP].all(axis=1))[1]  # (m,)
		idx_triV_diff = np.nonzero(diff_ijk_TriTriVSegV[idx_Share1V_IP].all(axis=2))[1].reshape(-1, 2)  # (m, 2)
		idx_triv_same = np.nonzero((~diff_ijk_TriTriVSegV[idx_Share1V_IP]).any(axis=2))[1]  # (m, )
		# If seg in same plane, then seg vector must be in same direction to n_edge and between adjacent edges
		v_Seg_i = cs_seg[idx_segV_diff] - cs_seg[1 - idx_segV_diff]  # (m, 2/3)
		vs_tri_i = cs_Tris_close[idx_Share1V_IP[:,npNuax], idx_triV_diff] - cs_Tris_close[idx_Share1V_IP, idx_triv_same][:,npNuax]  # (m, 2, 2/3)
		vs = np.cross(np.tile(v_Seg_i[:, npNuax], (1, 2, 1)), vs_tri_i)  # (m, 2, 3)
		segLine_betweenTriEg = 0 >= ((vs[:, 0] * vs[:, 1]).sum(axis=1) if dim == 3 else vs[:, 0] * vs[:, 1])
		cut_ithSeg = ((ns_edges[idx_Share1V_IP, idx_triv_same] * v_Seg_i).sum(axis=1) > 0) & segLine_betweenTriEg
		if any_not_idx:
			if cut_ithSeg.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V_IP][np.nonzero(cut_ithSeg)[0]])
	# Handle those that don't share verts
	v_seg = cs_seg[1] - cs_seg[0]  # (3, ) vector of the segment
	if (idx_0Share_OOP := np.nonzero((nDiffV_ithTri == 3) & (wrt_triN_ij_TriSegV != 0).any(axis=1))[0]).size:
		n12_seg = getNormalsof1Vec_3D(v_seg, first_both_comb=2)
		# ij: ith tri, jth triV. v_ijk_TriTriVSegV.shape (m, 3, 2, 3) projs_ij.shape (m, 3, 2)
		projs_ij = (v_ijk_TriTriVSegV[idx_0Share_OOP][:,:,[0,0]] * n12_seg).sum(axis=3)
		segLine_cutsTris_i = verify_origininPolygon_projs_ijkl_iObj_jV_lxy(projs_ij)
		cuts_i = (wrt_triN_ij_TriSegV[idx_0Share_OOP, 0] * wrt_triN_ij_TriSegV[idx_0Share_OOP, 1] <= 0) & segLine_cutsTris_i
		if any_not_idx:
			if cuts_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share_OOP][np.nonzero(cuts_i)[0]])
	
	if (idx_0Share_IP := np.nonzero((nDiffV_ithTri == 3) & (wrt_triN_ij_TriSegV == 0).all(axis=1))[0]).size:
		# While tri and seg are IP, they intersect when seg line cuts tri and seg not on half plane of a tri edge
		xs_segV_ij_TriTriEg = (-v_ijk_TriTriVSegV[idx_0Share_IP][:, [1, 2, 0]]
							   * np.tile(ns_edges[idx_0Share_IP, :, npNuax], (1, 1, 2, 1))).sum(axis=3).round(DIGIT_HEIGHT)  # (m, 3, 2)
		n1s_seg = (arr_90deg @ v_seg.T).T if dim == 2 else np.cross(ns_face[idx_0Share_IP], v_seg)  # (m, 2/3)
		n1s_seg = n1s_seg / npNorm(n1s_seg, axis=1)[:, npNuax]
		xs_triV_ij_TriTriV = (v_ijk_TriTriVSegV[idx_0Share_IP, :, 0] * n1s_seg[:, npNuax]).sum(axis=2).round(DIGIT_HEIGHT)  # (m, 3)
		# xs_segV_ij_TriTriEg.shape (m, 3, 2), xs_triV_ij_TriTriV.shape (m, 3)
		allSegVsOutanyEdge_ithTri = (xs_segV_ij_TriTriEg > 0).all(axis=2).any(axis=1)  # (m,)
		segLineOutTri_ithTri = (xs_triV_ij_TriTriV < 0).all(axis=1) | (xs_triV_ij_TriTriV > 0).all(axis=1)  # (m,)
		cut_i = ~(allSegVsOutanyEdge_ithTri | segLineOutTri_ithTri)
		if any_not_idx:
			if cut_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share_IP][np.nonzero(cut_i)[0]])
	return False if any_not_idx else idx_cut_rootLevel


"""Check if single triangle cuts multiple other objects: segs, tris"""
def check_Tri_cuts_Segs(cs_tri, cs_segs, c_cen_segs=None, r_segs=None, ns_edge=None, n_face=None, any_not_idx=True):
	# cs_segs.shape (n, 2, 2/3) n segments, each with 2 ends, 2D or 3D, cs_tri.shape (3, 3)
	r = max(npNorm(cs_tri - (c_cen_tri := np.average(cs_tri, axis=0)), axis=1))
	if c_cen_segs is None or r_segs is None:
		c_cen_segs, r_segs = np.average(cs_segs, axis=1), npNorm(cs_segs[:, 0] - cs_segs[:, 1], axis=1) / 2
	idx_closeEnough = np.nonzero(npNorm(c_cen_segs - c_cen_tri, axis=1) <= 1.01 * (r + r_segs))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	dim, cs_Segs_close = len(cs_tri[0]), cs_segs[idx_closeEnough]
	# ithSeg_jthSegV_from_kthTriV is (m, 2, 3, 2/3). Each jk is from vector from kth tri vert to jth seg vert of ith seg
	v_ijk_SegSegVTriV = np.tile(cs_Segs_close[:, :, npNuax], (1, 1, 3, 1)) - cs_tri
	diff_ijk_SegSegVTriV = v_ijk_SegSegVTriV.round(DIGIT_HEIGHT).any(axis=3)  # (m, 2, 3)
	nDiffV_ithSeg = np.count_nonzero(diff_ijk_SegSegVTriV.all(axis=2), axis=1)  # (m, ) how many different verts ith triangle has
	if (nDiffV_ithSeg == 0).all(): return False if any_not_idx else []
	
	if ns_edge is None or n_face is None:
		ns_edge, n_face = getNormals_OutofTriangle(cs_tri)
	# Get the height of ith segment's jth vert wrt triangle #(m, 2) m is num of segs
	if dim == 3: wrt_triN_ij_SegSegV = ((cs_Segs_close - cs_tri[0]) @ n_face).round(DIGIT_HEIGHT)
	else: wrt_triN_ij_SegSegV = np.zeros(diff_ijk_SegSegVTriV.shape[:2])
	# Exam those that share 1 vert with tri. Segs that have verts out of tri plane are seen as not cutting
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share1V_IP := np.nonzero((nDiffV_ithSeg == 1) & (wrt_triN_ij_SegSegV == 0).all(axis=1))[0]).size:
		cs_segs_Share1V_IP = cs_Segs_close[idx_Share1V_IP]
		# idx_ithSeg_diff get the indices of seg verts that are't shared for each segment, will be a 1-D array of 0/1
		# idx_triV_diff is the indices of tri verts that aren't shared for each segment, (m, 2)
		idx_segV_diff = np.nonzero(diff_ijk_SegSegVTriV[idx_Share1V_IP].all(axis=2))[1]
		idx_triV_diff = np.nonzero(diff_ijk_SegSegVTriV[idx_Share1V_IP].all(axis=1))[1].reshape(-1, 2)
		idx_triv_same = np.nonzero((~diff_ijk_SegSegVTriV[idx_Share1V_IP]).any(axis=1))[1]
		# If seg in same plane, then seg vector must be in same direction to n_edge and between adjacent edges
		it = range(len(idx_segV_diff))
		v_ithSeg = cs_segs_Share1V_IP[it, idx_segV_diff] - cs_segs_Share1V_IP[it, 1 - idx_segV_diff]
		v_tri_ithSeg = npA([cs_tri[idx] - cs_tri[i] for idx, i in zip(idx_triV_diff, idx_triv_same)])
		vs = np.cross(np.tile(v_ithSeg[:, npNuax], (1, 2, 1)), v_tri_ithSeg)  # (m, 2, 3)
		segLine_betweenTriEg = (vs[:, 0] * vs[:, 1]).sum(axis=1) if dim == 3 else vs[:, 0] * vs[:, 1]
		cut_ithSeg = ((ns_edge[idx_triv_same] * v_ithSeg).sum(axis=1) > 0) & (0 >= segLine_betweenTriEg)
		if any_not_idx:
			if cut_ithSeg.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V_IP][np.nonzero(cut_ithSeg)[0]])
	# Handle those that don't share verts
	if (idx_0Share := np.nonzero(nDiffV_ithSeg == 2)[0]).size:
		cs_segs_0Share = cs_Segs_close[idx_0Share]
		if (idx_0Share_OOP := np.nonzero((wrt_triN_ij_SegSegV[idx_0Share] != 0).any(axis=1))[0]).size:
			# ijk: ith seg, jth segV, kth triEg. Check if all tri edges cut segms
			projs_ijk = (v_ijk_SegSegVTriV[idx_0Share][idx_0Share_OOP][:,:,[[1, 0], [2, 0], [0, 0]]]
						* np.append(ns_edge[:,npNuax], np.tile(n_face, (3, 1, 1)), axis=1)).sum(axis=4)
			#projs_ijk (m, 2, 3, 2) check if there is a triEg whose half plane the segment doesn't cross
			cut_i = verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs_ijk, idx1_2C2, idx2_2C2)
			if any_not_idx:
				if cut_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][idx_0Share_OOP][np.nonzero(cut_i)[0]])
		
		if (idx_0Share_IP := np.delete(range(len(idx_0Share)), idx_0Share_OOP)).size:
			xs_ijk_SegSegVTriEg = (v_ijk_SegSegVTriV[idx_0Share][idx_0Share_IP][:, :, [1, 2, 0]] * ns_edge).sum(axis=3).round(DIGIT_HEIGHT)  # (m, 2, 3)
			cs_segs_0Share_IP = cs_segs_0Share[idx_0Share_IP]
			v_segs_Share_IP = cs_segs_0Share[idx_0Share_IP, 1] - cs_segs_0Share[idx_0Share_IP, 0]
			ns_seg = (arr_90deg @ v_segs_Share_IP.T).T if dim == 2 else np.cross(n_face, v_segs_Share_IP)
			ns_seg = ns_seg / np.tile(npNorm(ns_seg, axis=1), (dim, 1)).T
			v_tri_wrt_ithSeg0 = np.tile(cs_tri[npNuax], (len(cs_segs_0Share_IP), 1, 1)) - cs_segs_0Share_IP[:, [0, 0, 0]]
			xs_ij_SegSegN = (v_tri_wrt_ithSeg0 * ns_seg[:, npNuax]).sum(axis=2).round(DIGIT_HEIGHT)  # (m, 3)
			# For any seg, the following means it cuts tri: seg isn't outside any tri edge and tri verts are not on same side of seg line
			cut_ithSeg = (xs_ijk_SegSegVTriEg <= 0).any(axis=1).all(axis=1) & ~((xs_ij_SegSegN < 0).all(axis=1) | (xs_ij_SegSegN > 0).all(axis=1))
			if any_not_idx:
				if cut_ithSeg.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][idx_0Share_IP][np.nonzero(cut_ithSeg)[0]])
	return False if any_not_idx else idx_cut_rootLevel


def check_Tri_cuts_Tris(cs_tri0, cs_tris, c_cen_tris=None, r_tris=None, ns_triEgs=None, ns_triF=None, any_not_idx=True):
	# cs_tris.shape (n, 3, 2/3) n tris, cs_tri.shape (3, 3)
	r = max(npNorm(cs_tri0 - (c_cen_tri := np.average(cs_tri0, axis=0)), axis=1)) #c_cen_tri (3,)
	if c_cen_tris is None or r_tris is None:
		r_tris = npNorm(cs_tris - np.tile((c_cen_tris := np.average(cs_tris, axis=1))[:, npNuax], (1, 3, 1)), axis=2).max(axis=1)  # (m,)
	idx_closeEnough = np.nonzero(npNorm(c_cen_tris - c_cen_tri, axis=1) <= 1.01 * (r + r_tris))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	dim, cs_Tris_close = len(cs_tri0[0]), cs_tris[idx_closeEnough]
	
	# v_ijk_TriTriVTri0V is (m, 3, 3, 2/3). Each jk is from vector from kth tri0 vert to jth tri vert of ith tri
	v_ijk_TriTriVTri0V = cs_Tris_close[:, :, npNuax] - cs_tri0
	diff_ijk_TriTriVTri0V = v_ijk_TriTriVTri0V.round(DIGIT_HEIGHT).any(axis=3)  # (m, 3, 3)
	nDiffV_ithTri = np.count_nonzero(diff_ijk_TriTriVTri0V.all(axis=2), axis=1)  # (m, ) how many different verts ith triangle has
	if (nDiffV_ithTri == 0).all(): return False if any_not_idx else []
	
	if ns_triEgs is None or ns_triF is None:
		v01_02_tris = cs_Tris_close[:, 1:] - np.tile(cs_Tris_close[:, npNuax, 0], (1, 2, 1))  # (m, 2, 3)
		ns_triF = np.cross(v01_02_tris[:, 0], v01_02_tris[:, 1])  # (m, 3)
		ns_triEgs = np.cross(cs_Tris_close[:, [2, 0, 1]] - cs_Tris_close[:, [1, 2, 0]], np.tile(ns_triF[:, npNuax], (1, 3, 1)))  # (m, 3, 3)
		ns_triF = ns_triF / np.tile(npNorm(ns_triF, axis=1), (3, 1)).T
		ns_triEgs = ns_triEgs / np.tile(npNorm(ns_triEgs, axis=2)[..., npNuax], (1, 1, 3))
	else: ns_triEgs, ns_triF = ns_triEgs[idx_closeEnough], ns_triF[idx_closeEnough]
	#Get the normals of cs_tri0
	ns_edge, n_face = getNormals_OutofTriangle(cs_tri0)
	#wrt_triN_ik is kth tri0V wrt the plane of ith tri. (m, 3)
	if dim == 3: wrt_triN_ik = (v_ijk_TriTriVTri0V[:,0] * ns_triF[:,npNuax]).sum(axis=2).round(DIGIT_HEIGHT)
	else: wrt_triN_ik = np.zeros((idx_closeEnough.size, 3))
	
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share2V_IP := np.nonzero((nDiffV_ithTri == 1) & (wrt_triN_ik == 0).all(axis=1))[0]).size:
		idx_tri0_diff = np.nonzero(diff_ijk_TriTriVTri0V[idx_Share2V_IP].all(axis=1))[1]  # (m, )
		idx_tri_diff = np.nonzero(diff_ijk_TriTriVTri0V[idx_Share2V_IP].all(axis=2))[1]  # (m, )
		cut_i = (ns_edge[idx_tri0_diff] * ns_triEgs[idx_Share2V_IP, idx_tri_diff]).sum(axis=1) > 0
		if any_not_idx:
			if cut_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share2V_IP][np.nonzero(cut_i)[0]])
			
	if (idx_Share1V := np.nonzero(nDiffV_ithTri == 2)[0]).size:
		# idx_ithSeg_diff get the indices of seg verts that are't shared for each segment, will be a 1-D array of 0/1
		# idx_triV_diff is the indices of tri verts that aren't shared for each segment, (m, 2)
		idx_tri0V_diff = np.nonzero(diff_ijk_TriTriVTri0V[idx_Share1V].all(axis=1))[1].reshape(-1, 2) #(m, 2)
		idx_tri0V_same = np.nonzero((~diff_ijk_TriTriVTri0V[idx_Share1V]).any(axis=1))[1] #(m, )
		idx_triV_diff = np.nonzero(diff_ijk_TriTriVTri0V[idx_Share1V].all(axis=2))[1].reshape(-1, 2) #(m, 2)
		idx_triV_same = np.nonzero((~diff_ijk_TriTriVTri0V[idx_Share1V]).any(axis=2))[1] #(m, )
		IP_i = (wrt_triN_ik[idx_Share1V] == 0).all(axis=1)
		idx_Share1V_IP, idx_Share1V_OOP = np.nonzero(IP_i)[0], np.nonzero(~IP_i)[0]
		if idx_Share1V_IP.size: #Check if there are tris that are in same plane as tri0
			cs_Tris_Share1V_IP = cs_Tris_close[idx_Share1V][idx_Share1V_IP]
			idx_tri0V_diff_IP, idx_tri0V_same_IP = idx_tri0V_diff[idx_Share1V_IP], idx_tri0V_same[idx_Share1V_IP]
			idx_triV_diff_IP, idx_triV_same_IP = idx_triV_diff[idx_Share1V_IP], idx_triV_same[idx_Share1V_IP]
			#If seg in same plane, then seg vector must be in same direction to n_edge and between adjacent edges
			idx_m_2 = np.tile(np.arange(idx_Share1V_IP.size), (2, 1)).T
			vs_tri0_i = cs_tri0[idx_tri0V_diff_IP] - cs_tri0[np.tile(idx_tri0V_same_IP, (2, 1)).T]
			vs_tri_i = cs_Tris_Share1V_IP[idx_m_2, idx_triV_diff_IP] - cs_Tris_Share1V_IP[idx_m_2, np.tile(idx_triV_same_IP, (2, 1)).T]
			# vs_tri0_i.shape (m, 2, 3) vs_tri_i.shape (m, 2, 3) The vectors that start from shared verts
			arr_i = (vs_tri0_i[:, [0, 0, 1, 1]] * vs_tri0_i[:, [0, 1, 0, 1]]).sum(axis=2).reshape(-1, 2, 2)
			vTri1_dot_vsTri0, vTri2_dot_vsTri0 = (vs_tri_i[:, [0, 0]] * vs_tri0_i).sum(axis=2), (vs_tri_i[:, [1, 1]] * vs_tri0_i).sum(axis=2)
			# vTri_1&2 become linear combination of vTri0_1&2.  arr_i (m, 2, 2) vTri1_dot_vsTri0 (m, 2), sol_vTri1_i (m, 2), sol_vTri2_i (m, 2)
			sol_vTri1_i, sol_vTri2_i = npSolve(arr_i, vTri1_dot_vsTri0).round(DIGIT_HEIGHT), npSolve(arr_i, vTri2_dot_vsTri0).round(DIGIT_HEIGHT)
			anyVTri_in1stQuad = (sol_vTri1_i >= 0).all(axis=1) | (sol_vTri2_i >= 0).all(axis=1)  # (m, )
			crossXY_and_1stQuad = (sol_vTri1_i[:, 1] * sol_vTri2_i[:, 1] < 0) & (sol_vTri1_i[:, 0] * sol_vTri2_i[:, 0] < 0) & (
						sol_vTri1_i[:, 1] * np.cross(sol_vTri1_i, sol_vTri2_i) < 0)  # (m, )
			cut_i = anyVTri_in1stQuad | crossXY_and_1stQuad
			if any_not_idx:
				if cut_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][idx_Share1V_IP][np.nonzero(cut_i)[0]])
		if idx_Share1V_OOP.size:
			#ijk. ith tri, jth triV, kth tri0Eg. Only need to make sure the opposing triEg doesn't adjacent tri0Egs
			idx_tri0V_diff_OOP, idx_triV_diff_OOP = idx_tri0V_diff[idx_Share1V_OOP], idx_triV_diff[idx_Share1V_OOP]
			#xs_ijk & ys_ijk (m, 2, 2). ys_ij (m, 2)
			idx_m_2 = np.tile(idx_Share1V[idx_Share1V_OOP], (4, 1)).T
			displacements = v_ijk_TriTriVTri0V[idx_m_2,np.tile(idx_triV_diff_OOP, (1, 2))[:,[0, 2, 1, 3]],
											   np.tile((idx_tri0V_diff_OOP+1)%3, (1, 2))].reshape(-1, 2, 2, 3)
			xs_ijk = (displacements * ns_edge[idx_tri0V_diff_OOP][:,npNuax]).sum(axis=3) #(m, 2, 2)
			ys_ijk = (displacements @ n_face) #(m, 2, 2)
			projs_ijk = np.append(xs_ijk[...,npNuax], ys_ijk[...,npNuax], axis=3)
			cut_i = verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs_ijk, idx1_2C2, idx2_2C2, xs_ijk, ys_ijk)
			if any_not_idx:
				if cut_i.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][idx_Share1V_OOP][np.nonzero(cut_i)[0]])
			
	if (idx_0Share_OOP := np.nonzero((nDiffV_ithTri == 3) & (wrt_triN_ik != 0).any(axis=1))[0]).size:
		allTri0VOutTriF_i = (wrt_triN_ik[idx_0Share_OOP] > 0).all(axis=1) | (wrt_triN_ik[idx_0Share_OOP] < 0).all(axis=1)
		# ijk: ith tri, jth triV, kth triEg. Check if all tri0 edges cut tris. Get projections of triVs along tri0Egs
		projs_ijk = (v_ijk_TriTriVTri0V[idx_0Share_OOP][:,:,[[1, 0], [2, 0], [0, 0]]]
					* np.append(ns_edge[:,npNuax], np.tile(n_face, (3, 1, 1)), axis=1)).sum(axis=4)
		cut_i = ~allTri0VOutTriF_i & verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs_ijk, idx1_3C2, idx2_3C2)
		if any_not_idx:
			if cut_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share_OOP][np.nonzero(cut_i)[0]])
		
	if (idx_0Share_IP := np.nonzero((nDiffV_ithTri == 3) & (wrt_triN_ik == 0).all(axis=1))[0]).size:
		# ijk: ith tri, jth triV, kth tri0Eg/tri0V. Check if tri0 is outside a tri's half plane or vice versa. Only need x projection
		xs_ijk_ofTri = (v_ijk_TriTriVTri0V[idx_0Share_IP][:, :, [1, 2, 0]] * ns_edge).sum(axis=3).round(DIGIT_HEIGHT)  # (m, 3, 3)
		xs_ikj_ofTri0 = (-v_ijk_TriTriVTri0V[idx_0Share_IP][:, [1, 2, 0]].transpose((0, 2, 1, 3))
						 * ns_triEgs[idx_0Share_IP,npNuax]).sum(axis=3).round(DIGIT_HEIGHT)  # (m, 3, 3)
		cut_i = ~((xs_ijk_ofTri > 0).all(axis=1).any(axis=1) | (xs_ikj_ofTri0 > 0).all(axis=1).any(axis=1))
		if any_not_idx:
			if cut_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share_IP][np.nonzero(cut_i)[0]])
	
	return False if any_not_idx else idx_cut_rootLevel


"""Check if a single tetra cuts multiple triangles"""
def check_Tetra_cuts_Tris(cs_tetra, cs_tris, ns_triEgs=None, ns_triF=None, any_not_idx=True):
	# cs_Tris (n, 3, 3): each (3, 3) sub-array is the coordinates of a triangle to check against
	r = max(npNorm(cs_tetra - (c_cen_tetra := np.average(cs_tetra, axis=0)), axis=1))
	r_tris = npNorm(cs_tris - (c_cen_tris := np.average(cs_tris, axis=1))[:, npNuax], axis=2).max(axis=1)  # (m,)
	idx_closeEnough = np.nonzero(npNorm(c_cen_tris - c_cen_tetra, axis=1) <= 1.01 * (r + r_tris))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	cs_Tris_close = cs_tris[idx_closeEnough]
	# v_ijk_TriTriVTetraV is (m, 3, 4, 3). Each jk is from vector from kth tetra vert to jth tri vert of ith tri
	v_ijk_TriTriVTetraV = (np.tile(cs_Tris_close[:, :, npNuax], (1, 1, 4, 1)) - cs_tetra)
	diff_ijk_TriTriVTetraV = v_ijk_TriTriVTetraV.round(DIGIT_HEIGHT).any(axis=3)  # (m, 3, 4)
	nDiffV_ithTri = np.count_nonzero(diff_ijk_TriTriVTetraV.all(axis=2), axis=1)  # (m, ) how many different verts ith triangle has
	if (nDiffV_ithTri == 0).all(): return False if any_not_idx else []
	
	if ns_triEgs is None or ns_triF is None:
		v01_02_tris = cs_Tris_close[:, 1:] - np.tile(cs_Tris_close[:, npNuax, 0], (1, 2, 1))  # (m, 2, 3)
		ns_triF = np.cross(v01_02_tris[:, 0], v01_02_tris[:, 1])  # (m, 3)
		ns_triEgs = np.cross(cs_Tris_close[:, [2, 0, 1]] - cs_Tris_close[:, [1, 2, 0]], np.tile(ns_triF[:, npNuax], (1, 3, 1)))  # (m, 3, 3)
		ns_triF = ns_triF / npNorm(ns_triF, axis=1)[:,npNuax]
		ns_triEgs = ns_triEgs / npNorm(ns_triEgs, axis=2)[..., npNuax]
	else: ns_triEgs, ns_triF = ns_triEgs[idx_closeEnough], ns_triF[idx_closeEnough]
	
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share2V := np.nonzero(nDiffV_ithTri == 1)[0]).size:
		# Tris that share 1 edge with the tetra. Check whether the half plane cuts the tetra opposing seg
		# Identify the shared edges and oppo edges
		idx_tetra_diff = np.nonzero(diff_ijk_TriTriVTetraV[idx_Share2V].all(axis=1))[1].reshape(-1, 2)  # (m, 2)
		idx_tri_diff = np.nonzero(diff_ijk_TriTriVTetraV[idx_Share2V].all(axis=2))[1]  # (m, )
		idx_tri_same = np.nonzero((~diff_ijk_TriTriVTetraV[idx_Share2V]).any(axis=2))[1].reshape(-1, 2)  # (m, 2)
		vs_tetraOppoSegs = -v_ijk_TriTriVTetraV[idx_Share2V, idx_tri_same[:, 0]][np.tile(np.arange(len(idx_Share2V)), (2, 1)).T, idx_tetra_diff]
		ns_1triEg, ns_triF_2Share = ns_triEgs[idx_Share2V, idx_tri_diff], ns_triF[idx_Share2V]  # (m, 3) (m, 3)
		# ij: ith tetra oppo seg/tri edge, jth jth segV
		xs_ij = (vs_tetraOppoSegs * np.tile(ns_1triEg[:, npNuax], (1, 2, 1))).sum(axis=2)  # (m, 2)
		ys_ij = (vs_tetraOppoSegs * np.tile(ns_triF_2Share[:, npNuax], (1, 2, 1))).sum(axis=2)  # (m, 2)
		projs_ij = np.append(xs_ij[:, npNuax], ys_ij[:, npNuax], axis=1).transpose((0, 2, 1))  # (m, 2, 2)
		cut_i = verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs_ij[:,:,npNuax], idx1_2C2, idx2_2C2)
		#anyOnNegx_i = ((xs_ij <= 0) & (ys_ij == 0)).any(axis=1)  # any seg on tri edge half plane (m, )
		#segCrossX_i = ys_ij[:, 0] * ys_ij[:, 1] < 0  # any seg crosses tri half plane (m, )
		#anyMightCrossNegX_i = ys_ij[:, 0] * np.cross(projs_ij[:, :, 0], projs_ij[:, :, 1]) >= 0  # (m, )
		#cutsOppoEg_i = anyOnNegx_i | (segCrossX_i & anyMightCrossNegX_i)
		if any_not_idx:
			if cut_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share2V][np.nonzero(cut_i)[0]])
	
	if (idx_Share1V := np.nonzero(nDiffV_ithTri == 2)[0]).size:
		cs_Tris_Share1V = cs_Tris_close[idx_Share1V]
		idx_triV_diff = np.nonzero(diff_ijk_TriTriVTetraV[idx_Share1V].all(axis=2))[1].reshape(-1, 2)  # (m, 2)
		idx_triV_same = np.nonzero((~diff_ijk_TriTriVTetraV[idx_Share1V]).any(axis=2))[1]  # (m, )
		idx_tetraV_diff = np.nonzero(diff_ijk_TriTriVTetraV[idx_Share1V].all(axis=1))[1].reshape(-1, 3)  # (m, 3)
		idx_tetraV_same = np.nonzero((~diff_ijk_TriTriVTetraV[idx_Share1V]).any(axis=1))[1]  # (m, )
		rows = np.arange(len(idx_triV_diff))
		
		vs_tri_i = cs_Tris_Share1V[np.tile(rows, (2, 1)).T, idx_triV_diff] - np.tile(cs_Tris_Share1V[rows, npNuax, idx_triV_same], (1, 2, 1))  # (m, 2, 3)
		vs_tetra_i = npA([cs_tetra[idx] - cs_tetra[i] for idx, i in zip(idx_tetraV_diff, idx_tetraV_same)])  # (m, 3, 3)
		sols_i = npSolve(vs_tetra_i.transpose((0, 2, 1)), vs_tri_i.transpose((0, 2, 1))).transpose((0, 2, 1)).round(DIGIT_HEIGHT)  # (m, 2, 3)
		anySolin1stOcta_i = (sols_i >= 0).all(axis=2).any(axis=1)  # (m, )
		if (idx_anySolin1stOcta := np.nonzero(anySolin1stOcta_i)[0]).size:
			if any_not_idx: return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][idx_anySolin1stOcta])
		twoSols_safeHalfSpace = (sols_i < 0).all(axis=1).any(axis=1)
		crossing_i = sols_i[:, 0] * sols_i[:, 1] < 0  # (m, 3) ith v1-v2 cross x/y/z plane
		if (idx_lastCheck := np.nonzero(~anySolin1stOcta_i & ~twoSols_safeHalfSpace & crossing_i.any(axis=1))[0]).size:
			it, sols1, sols2 = range(len(idx_lastCheck)), sols_i[idx_lastCheck, 0], sols_i[idx_lastCheck, 1]  # (m, 3)
			idx_cross = crossing_i[idx_lastCheck].argmax(axis=1)  # (m, )
			crossing_1stQuadron = ((sols1 - (sols2 - sols1) * np.tile(
				sols1[it, idx_cross] / (sols2 - sols1)[it, idx_cross], (3, 1)).T).round(DIGIT_HEIGHT) >= 0).all(axis=1)
			if any_not_idx:
				if crossing_1stQuadron.any(): return True
			else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][idx_lastCheck][np.nonzero(crossing_1stQuadron)[0]])
	if (idx_0Share := np.nonzero(nDiffV_ithTri == 3)[0]).size:
		ns_tetraFace = getNormals_OutofTetra(cs_tetra)
		triV_out_tetraN = (np.roll(v_ijk_TriTriVTetraV[idx_0Share], -1, axis=2) * ns_tetraFace).sum(axis=3).round(DIGIT_HEIGHT)
		allTriVOutAnyF_i = (triV_out_tetraN > 0).all(axis=1).any(axis=1)  # (m,)
		# ijk: ith tri, jth triEg, kth tetraV|lth tetra eg
		xs_ijk = (-v_ijk_TriTriVTetraV[idx_0Share][:, [1, 2, 0]] * ns_triEgs[idx_0Share, :, npNuax]).sum(axis=3)  # (m, 3, 4)
		ys_ik = (-v_ijk_TriTriVTetraV[idx_0Share, 0] * ns_triF[idx_0Share, npNuax]).sum(axis=2)  # (m, 4)
		projs_ikj = np.append(xs_ijk.transpose(0, 2, 1)[..., npNuax], np.tile(ys_ik[...,npNuax,npNuax], (1, 1, 3, 1)), axis=3)  # (m, 4, 3, 2)
		cut_i = ~allTriVOutAnyF_i & verify_allHalfPlanesCutSegs_projs_ijkl_iObj_jV_kEg_lxy(projs_ikj, idx1_4C2, idx2_4C2)
		if any_not_idx:
			if cut_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][np.nonzero(cut_i)[0]])
	return False if any_not_idx else idx_cut_rootLevel


def check_Tetra_cuts_Segs(cs_tetra, cs_segs, ns_tetraFace=None, any_not_idx=True):
	# cs_Tris (n, 2, 3): each (2, 3) sub-array is the coordinates of a segment to check against
	r = max(npNorm(cs_tetra - (c_cen_tetra := np.average(cs_tetra, axis=0)), axis=1))
	r_segs = npNorm(cs_segs[:, 1] - cs_segs[:, 0], axis=1)  # (m, )
	idx_closeEnough = np.nonzero(npNorm(np.average(cs_segs, axis=1) - c_cen_tetra, axis=1) <= 1.01 * (r + r_segs))[0]
	if not idx_closeEnough.size: return False if any_not_idx else []
	cs_Segs_close = cs_segs[idx_closeEnough]  # (m, 2, 3)
	# v_ijk_SegSegVTetraV is (m, 2, 4, 3). Each jk is from vector from kth tetra vert to jth seg vert of ith seg
	v_ijk_SegSegVTetraV = (np.tile(cs_Segs_close[:, :, npNuax], (1, 1, 4, 1)) - cs_tetra)  # (m, 2, 4, 3)
	diff_ijk_SegSegVTetraV = v_ijk_SegSegVTetraV.round(DIGIT_HEIGHT).any(axis=3)  # (m, 2, 4)
	nDiffV_ithSeg = np.count_nonzero(diff_ijk_SegSegVTetraV.all(axis=2), axis=1)  # (m, ) how many different verts ith seg has
	if (nDiffV_ithSeg == 0).all(): return False if any_not_idx else []
	
	idx_cut_rootLevel = np.empty(0, dtype=int)
	if (idx_Share1V := np.nonzero(nDiffV_ithSeg == 1)[0]).size:  # If seg shares a vertex with tetra, it must not point inward the tetra
		cs_Segs_Share1V = cs_Segs_close[idx_Share1V]  # (m, 2, 3)
		idx_segV_diff = np.nonzero(diff_ijk_SegSegVTetraV[idx_Share1V].all(axis=2))[1]  # (m, )
		idx_segV_same = np.nonzero((~diff_ijk_SegSegVTetraV[idx_Share1V]).any(axis=2))[1]  # (m, )
		idx_tetraV_diff = np.nonzero(diff_ijk_SegSegVTetraV[idx_Share1V].all(axis=1))[1].reshape(-1, 3)  # (m, 3)
		idx_tetraV_same = np.nonzero((~diff_ijk_SegSegVTetraV[idx_Share1V]).any(axis=1))[1]  # (m, )
		rows = range(idx_segV_diff.size)
		
		v_seg_i = cs_Segs_Share1V[rows, idx_segV_diff] - cs_Segs_Share1V[rows, idx_segV_same]  # (m, 3)
		vs_tetra_i = cs_tetra[idx_tetraV_diff,:] - cs_tetra[idx_tetraV_same,npNuax] #(m, 3, 3) needs to be transposed
		solin1stOcta_i = (npSolve(vs_tetra_i.transpose((0, 2, 1)), v_seg_i).round(DIGIT_HEIGHT) >= 0).all(axis=1)  # (m, )
		if any_not_idx:
			if solin1stOcta_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_Share1V][np.nonzero(solin1stOcta_i)[0]])
	# For segs that don't share vertices. If all segVs outside a tetra half-space or segment line doesn't cut through tetra, then no cutting
	if (idx_0Share := np.nonzero(nDiffV_ithSeg == 2)[0]).size:
		ns_tetraFace = getNormals_OutofTetra(cs_tetra)  # (4, 3)
		segV_out_tetraN = (v_ijk_SegSegVTetraV[idx_0Share][:,:,[1, 2, 3, 0]] * ns_tetraFace).sum(axis=3).round(DIGIT_HEIGHT)
		allSegVOutAnyF_i = (segV_out_tetraN > 0).all(axis=1).any(axis=1)  # (m, )
		v_segs_i = cs_Segs_close[idx_0Share, 1] - cs_Segs_close[idx_0Share, 0]  # (m, 3)
		n12_segs_i = getNormalsofVecs_3D(v_segs_i, first_both_comb=2) #(m, 2, 3)
		# ij: ith seg, jth tetraV|kth tetra. v_ijk_SegSegVTetraV.shape (m, 2, 4, 3)
		projs_ik = (-v_ijk_SegSegVTetraV[idx_0Share, 0][:,:,npNuax] * n12_segs_i[:,npNuax]).sum(axis=3) #(m, 4, 2)
		segLine_cutsTris_i = verify_origininPolygon_projs_ijkl_iObj_jV_lxy(projs_ik)
		cuts_i = ~allSegVOutAnyF_i & segLine_cutsTris_i
		if any_not_idx:
			if cuts_i.any(): return True
		else: idx_cut_rootLevel = np.append(idx_cut_rootLevel, idx_closeEnough[idx_0Share][np.nonzero(cuts_i)[0]])
	
	return False if any_not_idx else idx_cut_rootLevel
