# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tier 1: turn a flattened LDraw part into one closed, oriented triangle mesh.

LDraw parts are assembled from pieces that touch without sharing edges: a stud
stands on the top face, a tube ends against the underside of the top plate.
`solidify` closes every such piece with a cap over its open loops, which gives a
set of closed solids, and unions them with manifold3d.  Caps lie on the surface
they rest against, so the union removes them again and cuts the matching hole
into that surface.  A cap that survives the union marks a place where the input
did not touch anything and the result should not be trusted.

Every output triangle keeps the index of the `Flat` triangle it came from
(`tri_source`, -1 for a surviving cap), so provenance survives.  A cap carries
the instance id of the loop it closes in `tri_instance`.

Welding, edge counting and the export helpers are numpy-only.  manifold3d is
imported by `solidify` alone.
"""
from __future__ import annotations

import io
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .parser import Flat, LDRAW_TO_ZUP, LDU_TO_MM

WELD_TOL = 1e-3   # LDU


# ----------------------------------------------------------------------------
# Welding and topology
# ----------------------------------------------------------------------------
def weld(tris: np.ndarray, tol: float = WELD_TOL):
    """Merge vertices closer than tol/2 along every axis.

    A single rounding grid splits pairs that straddle a cell boundary, however
    close (6377 has vertices 0.0005 LDU apart in different cells).  Eight grids
    shifted by half a cell cover every such pair; clusters are joined across
    grids, so a few vertices up to tol*sqrt(3) apart may merge as well.
    -> (verts, faces, keep).  `verts` keeps the first original position of each
    welded vertex, `keep` marks triangles still non-degenerate after welding.
    """
    pts = tris.reshape(-1, 3)
    upts, first, inv = np.unique(pts, axis=0, return_index=True, return_inverse=True)
    label = np.arange(len(upts))
    shifts = [np.array([i, j, k]) * tol / 2 for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    changed = True
    while changed:                       # propagate the smallest label through each cell
        changed = False
        for s in shifts:
            _, cell = np.unique(np.floor((upts + s) / tol).astype(np.int64), axis=0, return_inverse=True)
            cell = cell.reshape(-1)
            low = np.full(cell.max() + 1, len(upts))
            np.minimum.at(low, cell, label)
            new = low[cell]
            if (new != label).any():
                label, changed = new, True
    # Number clusters by their first occurrence in the input.
    rep_first = np.full(len(upts), len(pts))
    np.minimum.at(rep_first, label, first)
    order, cluster = np.unique(rep_first[label], return_inverse=True)
    faces = cluster.reshape(-1)[inv.reshape(-1)].reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return pts[order], faces, keep


def edge_use(faces: np.ndarray):
    """-> (open, shared by 2, shared by 3+) undirected edge counts."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts == 2).sum()), int((counts > 2).sum())


def topology(tris: np.ndarray, tol: float = WELD_TOL):
    """Weld vertices on a grid and count how many faces meet at each edge."""
    _, faces, keep = weld(tris, tol)
    return edge_use(faces[keep])


def open_half_edges(faces: np.ndarray):
    """Directed edges a->b whose reverse b->a is missing.

    -> (edges (K, 2), owning triangle (K,), position of a in that triangle (K,)).
    """
    he = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
    present = set(map(tuple, he.tolist()))
    is_open = np.array([(b, a) not in present for a, b in he.tolist()], dtype=bool)
    tri = np.repeat(np.arange(len(faces)), 3)
    pos = np.tile(np.arange(3), len(faces))
    return he[is_open], tri[is_open], pos[is_open]


def tri_normals_area(t: np.ndarray):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n, np.linalg.norm(n, axis=1) / 2


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    t = verts[faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


# ----------------------------------------------------------------------------
# Repairs before the union
# ----------------------------------------------------------------------------
def split_t_junctions(verts, faces, src, tol: float = WELD_TOL, max_passes: int = 8):
    """Split open edges that have another open edge's vertex lying on them.

    The triangle owning the edge is fanned from its opposite vertex through the
    inserted vertices, so orientation and `src` carry over.  One edge per
    triangle per pass; passes repeat until nothing changes.
    -> (faces, src, number of vertices inserted)
    """
    inserted = 0
    for _ in range(max_passes):
        he, tri, pos = open_half_edges(faces)
        if not len(he):
            break
        cand = np.unique(he)
        q = verts[cand]
        drop = np.zeros(len(faces), dtype=bool)
        new_f, new_s = [], []
        for (a, b), t, k in zip(he.tolist(), tri.tolist(), pos.tolist()):
            if drop[t]:
                continue
            pa, d = verts[a], verts[b] - verts[a]
            length = float(np.linalg.norm(d))
            s = (q - pa) @ d / (length * length)
            dist = np.linalg.norm(pa + s[:, None] * d - q, axis=1)
            hit = (dist < tol) & (s * length > tol) & ((1 - s) * length > tol)
            if not hit.any():
                continue
            chain = [a, *cand[hit][np.argsort(s[hit])].tolist(), b]
            c = faces[t][(k + 2) % 3]
            new_f += [(x, y, c) for x, y in zip(chain, chain[1:])]
            new_s += [src[t]] * (len(chain) - 1)
            inserted += len(chain) - 2
            drop[t] = True
        if not drop.any():
            break
        faces = np.concatenate([faces[~drop], np.array(new_f, dtype=faces.dtype)])
        src = np.concatenate([src[~drop], np.array(new_s, dtype=src.dtype)])
    return faces, src, inserted


def pair_half_edges(verts: np.ndarray, faces: np.ndarray):
    """Match each half-edge with the one that continues the same surface.

    Half-edge h runs from faces[h // 3, h % 3] to the next corner.  An edge with
    exactly two opposite half-edges is paired directly.  Where more faces meet
    (a rib whose top edge also touches a wall), the faces are sorted by angle
    around the edge and neighbours are paired only when the sector between
    them is material for both.  Unmatched half-edges stay open and are capped
    with the piece they belong to.
    -> (partner (3T,) half-edge index or -1, number of edges that needed sorting)
    """
    he = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
    groups = defaultdict(list)
    for h, (a, b) in enumerate(np.sort(he, axis=1).tolist()):
        groups[(a, b)].append(h)
    partner = np.full(len(he), -1, dtype=np.int64)
    sorted_edges = 0
    for (a, b), hs in groups.items():
        if len(hs) == 1:
            continue
        if len(hs) == 2 and he[hs[0], 0] == he[hs[1], 1]:
            partner[hs[0]], partner[hs[1]] = hs[1], hs[0]
            continue
        sorted_edges += 1
        u = verts[b] - verts[a]
        u /= np.linalg.norm(u)
        x = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(u, x)                     # x, y, u right-handed: angles run CCW about u
        ang, sign = [], []
        for h in hs:
            c = faces[h // 3, (h % 3 + 2) % 3]
            w = verts[c] - verts[a]
            ang.append(np.arctan2(w @ y, w @ x))
            sign.append(1 if he[h, 0] == a else -1)
        # A face running a->b has its normal on its CCW side, so material lies
        # clockwise of it; a face running b->a has material counter-clockwise.
        order = np.argsort(ang)
        used = set()
        for i in range(len(hs)):
            p, q = order[i], order[(i + 1) % len(hs)]
            if p in used or q in used or p == q:
                continue
            if sign[p] == -1 and sign[q] == 1:
                partner[hs[p]], partner[hs[q]] = hs[q], hs[p]
                used |= {p, q}
    return partner, sorted_edges


def face_components(n_tris: int, partner: np.ndarray) -> list:
    """Groups of triangle indices connected through paired half-edges."""
    parent = list(range(n_tris))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for h, p in enumerate(partner.tolist()):
        if p > h:
            parent[find(p // 3)] = find(h // 3)
    roots = np.array([find(t) for t in range(n_tris)])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


def boundary_loops(edges: np.ndarray) -> list:
    """Chains of open half-edges (K, 2), each a simple closed list of vertex indices.

    Loops may touch at a vertex (two stud bases meeting at a corner).  The open
    edges are walked as closed paths and each path is cut wherever it passes a
    vertex twice, which gives the same simple loops whichever way it went.
    """
    nxt = defaultdict(list)
    balance = defaultdict(int)
    for a, b in edges.tolist():
        nxt[a].append(b)
        balance[a] += 1
        balance[b] -= 1
    bad = [v for v, d in balance.items() if d]
    if bad:
        raise SolidifyError(f"open edges at vertex {bad[0]} do not close into loops")
    loops = []
    for start in list(nxt):
        while nxt[start]:
            walk, v = [start], nxt[start].pop()
            while v != start:
                walk.append(v)
                v = nxt[v].pop()
            stack, at = [], {}
            for v in walk:
                if v in at:                       # back at a vertex: cut off the loop since then
                    i = at[v]
                    loops.append(stack[i:])
                    for u in stack[i + 1:]:
                        del at[u]
                    del stack[i + 1:]
                else:
                    at[v] = len(stack)
                    stack.append(v)
            loops.append(stack)
    return loops


def _is_planar(p: np.ndarray, tol: float) -> bool:
    c = p.mean(0)
    s = np.linalg.svd(p - c)
    n = s[2][2]
    return s[1][1] > tol and np.abs((p - c) @ n).max() <= tol    # 2D extent, flat


def _planar_pieces(verts, loop, tol, max_len: int = 200):
    """Split a loop along chords into planar loops, or None if that fails.

    A box missing two adjacent faces leaves one L-shaped loop; the chord along
    their missing common edge splits it back into the two faces.  Greedy: cut
    off the largest planar piece, then repeat on the rest.  Each chord appears
    in both pieces in opposite directions, so their caps share it.
    """
    if _is_planar(verts[loop], tol):
        return [loop]
    n = len(loop)
    if n > max_len:
        return None
    best = None
    for i in range(n):
        for j in range(i + 2, n if i else n - 1):
            for piece, rest in ((loop[i:j + 1], loop[j:] + loop[:i + 1]),
                                (loop[j:] + loop[:i + 1], loop[i:j + 1])):
                if len(piece) >= 3 and (best is None or len(piece) > len(best[0])) \
                        and _is_planar(verts[piece], tol):
                    best = (piece, rest)
    if best is None:
        return None
    rest = _planar_pieces(verts, best[1], tol, max_len)
    return None if rest is None else [best[0]] + rest


def _clean_triangulate(poly: list, idx: list, tol: float):
    """Triangulate planar cap loops after cleaning them with a 2D union.

    Split loops can leave chords of two nested loops on one line, running in
    opposite directions (6377: a stepped tube's inner and outer cap at y = 4).
    That zero-width slit is not a valid polygon: the triangulator returns
    overlapping triangles and manifold3d then invents faces around them.  The
    union removes slits; its points are mapped back to the loop vertices.  It is
    used only when the plain triangulation overlaps itself (its area exceeds the
    region's) or has zero-height triangles, which manifold3d removes by swapping
    edges into faces off the surface.  If a point matches no loop vertex the
    plain triangulation is kept.
    -> (triangles into the returned vertex ids, vertex ids)
    """
    import manifold3d as mf

    flat_idx = np.concatenate(idx)
    flat_pts = np.concatenate(poly)
    raw = np.asarray(mf.triangulate(poly))
    t = flat_pts[raw]
    areas = np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                   - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])) / 2
    longest = np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2).max(axis=1)
    region = mf.CrossSection(poly, mf.FillRule.Positive)
    # A slit shows as overlapping triangles or as zero-height ones along it.
    if areas.sum() <= region.area() + tol * tol and (areas > tol * longest).all():
        return raw, flat_idx                       # clean polygons: keep them as they are
    clean = [p for p in region.to_polygons() if len(p) >= 3]
    ids = []
    for p in clean:
        d = np.linalg.norm(p[:, None, :] - flat_pts[None, :, :], axis=2)
        near = d.argmin(axis=1)
        if (d[np.arange(len(p)), near] > tol).any():
            return raw, flat_idx
        ids.append(flat_idx[near])
    if not clean:
        return np.zeros((0, 3), dtype=np.int64), flat_idx
    return np.asarray(mf.triangulate(clean)), np.concatenate(ids)


def cap_loops(verts, loops, tol: float = WELD_TOL, fan=()):
    """Faces that close the given loops, running against their open edges.

    Non-planar loops are split along chords into planar pieces, unless their
    index is in `fan`.  Planar loops on one plane are triangulated together, so
    nested loops give a ring.  The rest are fanned to their centroid (one new
    vertex each).
    -> (faces (C, 3), extra vertices (E, 3), loop index per face (C,),
        indices of the loops that were split)
    """
    import manifold3d as mf

    groups, fanned, split = [], [], set()     # groups: [centre, normal, [(loop, i), ...]]
    pieces = []
    for i, loop in enumerate(loops):
        q = None if i in fan else _planar_pieces(verts, loop, tol)
        if q is None:
            fanned.append((loop, i))
            continue
        if len(q) > 1:
            split.add(i)
        pieces += [(piece, i) for piece in q]
    for loop, i in pieces:
        p = verts[loop]
        c = p.mean(0)
        n = np.linalg.svd(p - c)[2][2]
        for g in groups:
            if abs(abs(g[1] @ n) - 1) < 1e-6 and abs((c - g[0]) @ g[1]) < tol:
                g[2].append((loop, i))
                break
        else:
            groups.append([c, n, [(loop, i)]])

    faces, face_loop, extra = [], [], []
    for c, n, group in groups:
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        idx = [np.array(loop[::-1]) for loop, _ in group]    # cap runs against the open edges
        poly = [np.c_[(verts[k] - c) @ e1, (verts[k] - c) @ e2] for k in idx]
        area = sum((p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]).sum()
                   for p in poly)
        if area < 0:                                       # triangulate wants outer loops CCW
            poly = [p * [1.0, -1.0] for p in poly]
        tri, ids = _clean_triangulate(poly, idx, tol)
        faces.append(ids[tri])
        loop_of = {v: i for loop, i in group for v in loop}
        face_loop.append(np.array([loop_of[x] for x in ids[tri[:, 0]]], dtype=np.int64))
    for loop, i in fanned:
        k = len(verts) + len(extra)
        extra.append(verts[loop].mean(0))
        faces.append(np.array([(k, b, a) for a, b in zip(loop, loop[1:] + loop[:1])]))
        face_loop.append(np.full(len(loop), i))
    faces = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
    face_loop = np.concatenate(face_loop) if face_loop else np.zeros(0, dtype=np.int64)
    extra = np.array(extra, dtype=float).reshape(-1, 3)
    return faces, extra, face_loop, split


def _plane_frame(n: np.ndarray):
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _cancel_self_contact(V, F, ids, tol: float = WELD_TOL):
    """Remove zero-thickness double walls inside one closed piece.

    A piece can rest against itself: 6377's rail segments end against a
    cross-wall they are joined to elsewhere, so the segment's cap lies flat on
    the wall, facing the other way.  manifold3d assumes a piece never overlaps
    itself and invents faces around such a contact.  Here, in each plane that
    holds faces of both orientations, their overlap is cut out of both sides
    and the rest re-triangulated.  T-junctions this leaves on neighbouring
    edges are split afterwards.  A new triangle keeps the id of the original
    triangle it lies in.
    -> (V, F, ids, number of planes changed); the input comes back unchanged
       when the result would not be closed.
    """
    import manifold3d as mf

    n, area = tri_normals_area(V[F])
    good = area > 1e-12
    nu = np.zeros_like(n)
    nu[good] = n[good] / np.linalg.norm(n[good], axis=1, keepdims=True)
    lead = np.take_along_axis(nu, np.argmax(np.abs(nu) > 1e-6, axis=1)[:, None], axis=1)[:, 0]
    side = np.where(lead < 0, -1, 1)
    key_n = nu * side[:, None]
    d = np.einsum("ij,ij->i", key_n, V[F[:, 0]])
    groups = defaultdict(list)
    for t in np.flatnonzero(good):
        groups[(*np.round(key_n[t], 4).tolist(), round(float(d[t]) / tol))].append(t)

    drop, new_f, new_ids, new_v = set(), [], [], []
    changed = 0
    for key, ts in groups.items():
        ts = np.array(ts)
        if len(set(side[ts].tolist())) < 2:
            continue
        kn = np.array(key[:3]) / np.linalg.norm(key[:3])
        e1, e2 = _plane_frame(kn)
        origin = kn * d[ts[0]]

        def flat2d(t, flip):
            q = (V[F[t]] - origin) @ np.c_[e1, e2]
            return q[::-1] if flip else q

        plus = [t for t in ts if side[t] > 0]
        minus = [t for t in ts if side[t] < 0]
        reg_p = mf.CrossSection([flat2d(t, False) for t in plus], mf.FillRule.Positive)
        reg_m = mf.CrossSection([flat2d(t, True) for t in minus], mf.FillRule.Positive)
        overlap = reg_p ^ reg_m
        if overlap.area() < tol * tol:
            continue
        changed += 1
        drop.update(ts.tolist())
        local = {tuple(np.round((V[x] - origin) @ np.c_[e1, e2], 6)): x for x in np.unique(F[ts])}
        for region, src_tris, flip in ((reg_p - overlap, plus, False), (reg_m - overlap, minus, True)):
            polys = [p for p in region.to_polygons() if len(p) >= 3]
            if not polys:
                continue
            tri = np.asarray(mf.triangulate(polys))
            pts = np.concatenate(polys)
            idx = []
            for q in pts:
                k = local.get(tuple(np.round(q, 6)))
                if k is None:                      # nearest existing vertex, else a new one
                    cand = np.array(list(local.values()))
                    dist = np.linalg.norm((V[cand] - origin) @ np.c_[e1, e2] - q, axis=1)
                    if dist.min() <= tol:
                        k = cand[np.argmin(dist)]
                    else:
                        k = len(V) + len(new_v)
                        new_v.append(origin + q[0] * e1 + q[1] * e2)
                        local[tuple(np.round(q, 6))] = k
                idx.append(k)
            faces2 = np.array(idx)[tri]
            if flip:
                faces2 = faces2[:, ::-1]
            # provenance: the original triangle of this orientation holding the centroid
            for f2, c2 in zip(faces2, pts[tri].mean(1)):
                owner = src_tris[0]
                for t in src_tris:
                    a, b, c = flat2d(t, flip)
                    s1 = (b[0] - a[0]) * (c2[1] - a[1]) - (b[1] - a[1]) * (c2[0] - a[0])
                    s2 = (c[0] - b[0]) * (c2[1] - b[1]) - (c[1] - b[1]) * (c2[0] - b[0])
                    s3 = (a[0] - c[0]) * (c2[1] - c[1]) - (a[1] - c[1]) * (c2[0] - c[0])
                    if min(s1, s2, s3) >= -1e-9:
                        owner = t
                        break
                new_f.append(f2)
                new_ids.append(ids[owner])
    if not changed:
        return V, F, ids, 0
    keep = np.array([t not in drop for t in range(len(F))])
    V2 = np.concatenate([V, np.array(new_v).reshape(-1, 3)])
    F2 = np.concatenate([F[keep], np.array(new_f, dtype=F.dtype).reshape(-1, 3)])
    ids2 = np.concatenate([ids[keep], np.array(new_ids, dtype=ids.dtype)])
    F2, ids2, _ = split_t_junctions(V2, F2, ids2, tol)
    o, _, more = edge_use(F2)
    if o or more:
        return V, F, ids, 0
    return V2, F2, ids2, changed


# ----------------------------------------------------------------------------
# Tier 1 entry point
# ----------------------------------------------------------------------------
class SolidifyError(RuntimeError):
    pass


@dataclass
class Solid:
    verts: np.ndarray          # (V, 3) LDU, LDraw axes
    faces: np.ndarray          # (T, 3) CCW seen from outside
    tri_source: np.ndarray     # (T,) index into Flat.tris, -1 for a cap
    tri_instance: np.ndarray   # (T,) Flat instance id; for a cap, the instance it closes
    stats: dict = field(default_factory=dict)

    def to_print_frame(self) -> np.ndarray:
        """Vertices in millimetres, Z up, winding preserved."""
        return (self.verts @ LDRAW_TO_ZUP.T) * LDU_TO_MM


def solidify(flat: Flat, tol: float = WELD_TOL) -> Solid:
    """Weld, split T-junctions, cap open loops and union the closed pieces.

    `tol` is both the weld distance and the precision given to the union.
    A non-planar loop is first capped with planar pieces.  If any of its caps
    survive the union, it is capped again with a fan, and the union with fewer
    leftover caps is kept: a correct cap disappears into the surface it rests on.
    """
    verts, faces, keep = weld(flat.tris, tol)
    src = np.flatnonzero(keep)
    faces = faces[keep]
    faces, src, n_tj = split_t_junctions(verts, faces, src, tol)
    partner, n_sorted = pair_half_edges(verts, faces)
    he = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
    comps = []
    for comp in face_components(len(faces), partner):
        hid = (comp[:, None] * 3 + np.arange(3)).ravel()
        open_h = hid[partner[hid] < 0]
        comps.append((comp, open_h, boundary_loops(he[open_h])))

    first = _union(flat, verts, faces, src, he, comps, tol, fan=set())
    retry = {k for k in first["left"] if k in first["split"]}
    best = first
    if retry:
        second = _union(flat, verts, faces, src, he, comps, tol, fan=retry)
        if second["left_area"] < first["left_area"]:
            best = second
    solid = best["solid"]
    if best["left"]:
        solid = _cut_double_walls(best, tol)
    solid = _drop_needles(solid, tol)
    solid.stats.update({"welded_tris": int(keep.sum()), "tjunction_verts": n_tj,
                        "sorted_edges": n_sorted, "loops": sum(len(c[2]) for c in comps),
                        "loops_retried": len(retry) if best is not first else 0})
    return solid


def _cut_double_walls(u: dict, tol: float, max_planes: int = 64) -> Solid:
    """Remove zero-thickness double walls left where a piece rests against itself.

    A rail's open end can rest against a cross-wall it is joined to elsewhere.
    The union only merges separate pieces, so the cap and the wall survive as
    two coincident sheets with material on both sides.  Cutting the result along
    each leftover cap's plane and joining the halves removes them.  Faces with
    material on one side only are untouched, so a cap that survives this really
    is exposed.
    """
    result, allsrc, allinst = u["manifold"], u["allsrc"], u["allinst"]
    v, f, fid = _mesh_of(result)
    caps = np.flatnonzero(allsrc[fid] < 0)
    n, _ = tri_normals_area(v[f[caps]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = np.einsum("ij,ij->i", n, v[f[caps, 0]])
    planes = np.unique(np.c_[np.round(n, 6), np.round(d / tol) * tol], axis=0)[:max_planes]
    for p in planes:
        a, b = result.split_by_plane(p[:3].tolist(), float(p[3]))
        result = a + b
    v, f, fid = _mesh_of(result)
    stats = dict(u["solid"].stats, double_wall_cuts=len(planes))
    return Solid(v, f, allsrc[fid], allinst[fid], stats)


def _drop_needles(solid: Solid, tol: float) -> Solid:
    """Weld the union result and drop the triangles that collapse.

    The union can place a new vertex 1e-6..1e-5 LDU from an existing one.  The
    mesh is valid by index, but in a float32 STL the two become one point, so a
    slicer sees collapsed triangles and edges with four faces.  Welding at the
    same tolerance as the input removes each needle together with its partner
    across the short edge.
    """
    v, f, keep = weld(solid.verts[solid.faces], tol)
    stats = dict(solid.stats, needles_dropped=int((~keep).sum()))
    return Solid(v, f[keep], solid.tri_source[keep], solid.tri_instance[keep], stats)


def _mesh_of(man):
    """(vertices, faces, face ids) of a manifold3d result."""
    out = man.to_mesh64()
    return (np.asarray(out.vert_properties)[:, :3].astype(float),
            np.asarray(out.tri_verts).astype(np.int64),
            np.asarray(out.face_id).astype(np.int64))


def _union(flat, verts, faces, src, he, comps, tol, fan):
    """Cap every piece (loops in `fan` as fans), union, and report leftover caps."""
    import manifold3d as mf

    # Each piece is capped on its own and gets its own vertices, so pieces that
    # touch along an edge or at a vertex stay separate until the union.
    allf, allsrc, allinst, n_all = [faces], [src], [flat.tri_instance[src]], len(faces)
    cap_key = [np.full(len(faces), -1)]                 # global loop number per face
    keys, split_keys = [], set()
    pieces, piece_vol, n_split, n_fan, n_contact = [], [], 0, 0, 0
    for ci, (comp, open_h, loops) in enumerate(comps):
        base = len(keys)
        keys += [(ci, li) for li in range(len(loops))]
        cap_f, extra, face_loop, split = cap_loops(
            verts, loops, tol, fan={li for (c, li) in fan if c == ci})
        split_keys |= {(ci, li) for li in split}
        n_split += len(split)
        n_fan += len(extra)                              # one centroid per fanned loop
        verts = np.concatenate([verts, extra])
        allf.append(cap_f)
        allsrc.append(np.full(len(cap_f), -1, dtype=src.dtype))
        cap_key.append(base + face_loop)
        # A cap belongs to the instance whose open edge it closes.
        opener = dict(zip(he[open_h, 0].tolist(), (open_h // 3).tolist()))
        owner = [next(opener[x] for x in tri if x in opener) for tri in cap_f.tolist()]
        allinst.append(flat.tri_instance[src[np.array(owner, dtype=np.int64)]] if owner
                       else np.zeros(0, dtype=flat.tri_instance.dtype))
        ids = np.concatenate([comp, np.arange(n_all, n_all + len(cap_f))])
        n_all += len(cap_f)
        piece = np.concatenate([faces[comp], cap_f])
        verts, piece, ids, k = _cancel_self_contact(verts, piece, ids, tol)
        n_contact += k
        used, local = np.unique(piece, return_inverse=True)
        mesh = mf.Mesh64(np.array(verts[used], dtype=np.float64, order="C"),
                         np.array(local.reshape(-1, 3), dtype=np.uint64, order="C"),
                         face_id=np.array(ids, dtype=np.uint64))
        man = mf.Manifold(mesh)
        if man.status() != mf.Error.NoError:
            o, _, m = edge_use(piece)
            raise SolidifyError(f"piece of {len(piece)} triangles is not closed "
                                f"({man.status()}, {o} open edges, {m} shared by 3+)")
        piece_vol.append(man.volume())
        # Library coordinates have 4 decimals, so a cap can poke ~1e-4 LDU out of
        # the surface it rests on.  Treat that as touching, not as a sliver.
        pieces.append(man.set_tolerance(tol))
    if min(piece_vol) <= 0:
        raise SolidifyError(f"a closed piece has non-positive volume: {sorted(piece_vol)[:3]}")

    allsrc, allinst = np.concatenate(allsrc), np.concatenate(allinst)
    cap_key = np.concatenate(cap_key)
    result = mf.Manifold.batch_boolean(pieces, mf.OpType.Add)
    out_v, out_f, fid = _mesh_of(result)
    left = cap_key[fid][allsrc[fid] < 0]
    _, area = tri_normals_area(out_v[out_f])
    stats = {"loops_split": n_split, "loops_fanned": n_fan, "double_wall_cuts": 0,
             "self_contact_planes": n_contact,
             "cap_tris": int(sum(len(f) for f in allf[1:])), "pieces": len(pieces)}
    return {"solid": Solid(out_v, out_f, allsrc[fid], allinst[fid], stats),
            "left": {keys[k] for k in left.tolist()}, "split": split_keys,
            "left_area": float(area[allsrc[fid] < 0].sum()),
            "manifold": result, "allsrc": allsrc, "allinst": allinst}


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _orient(a, b, c, d):
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)


def self_intersections(verts: np.ndarray, faces: np.ndarray) -> int:
    """Pairs of triangles, sharing no vertex, where an edge of one pierces the other.

    Only proper crossings count; coplanar overlaps and touching are not detected.
    """
    t = verts[faces]
    lo, hi = t.min(1), t.max(1)
    eps = 1e-12 * float(np.ptp(verts, axis=0).max() or 1.0) ** 3
    count = 0
    for i in range(len(t) - 1):
        j = np.arange(i + 1, len(t))
        j = j[np.all((lo[j] <= hi[i]) & (hi[j] >= lo[i]), axis=1)]
        j = j[~np.isin(faces[j], faces[i]).any(axis=1)]
        if not len(j):
            continue
        A, B = np.broadcast_to(t[i], (len(j), 3, 3)), t[j]
        hit = np.zeros(len(j), dtype=bool)
        for X, Y in ((A, B), (B, A)):                 # edges of X against triangle Y
            a, b, c = Y[:, 0], Y[:, 1], Y[:, 2]
            for k in range(3):
                p, q = X[:, k], X[:, (k + 1) % 3]
                d1, d2 = _orient(a, b, c, p), _orient(a, b, c, q)
                s1, s2, s3 = _orient(p, q, a, b), _orient(p, q, b, c), _orient(p, q, c, a)
                hit |= (d1 * d2 < -eps * eps) & (((s1 > eps) & (s2 > eps) & (s3 > eps))
                                                 | ((s1 < -eps) & (s2 < -eps) & (s3 < -eps)))
        count += int(hit.sum())
    return count


def pinch_fans(faces: np.ndarray) -> dict:
    """{vertex: [face groups]} for vertices where the surface touches itself.

    Around a normal vertex the faces form one fan linked through shared edges;
    two or more fans mean separate sheets meeting at a single point.
    """
    inc = defaultdict(list)
    for t, tri in enumerate(faces.tolist()):
        for x in tri:
            inc[x].append(t)
    out = {}
    for x, ts in inc.items():
        parent = {t: t for t in ts}

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        by_neighbour = defaultdict(list)
        for t in ts:
            for y in faces[t].tolist():
                if y != x:
                    by_neighbour[y].append(t)
        for tt in by_neighbour.values():
            for a in tt[1:]:
                parent[find(a)] = find(tt[0])
        groups = defaultdict(list)
        for t in ts:
            groups[find(t)].append(t)
        if len(groups) > 1:
            out[x] = list(groups.values())
    return out


def separate_pinches(solid: Solid, nudge: float = WELD_TOL) -> Solid:
    """Give each sheet meeting at a pinch vertex its own copy, moved `nudge` LDU into its sheet.

    Slicers accept a surface that touches itself at a point, and so does OCC in
    memory, but OCC's STEP import splits such a vertex and leaves wires open.
    Moving each copy into its own fan makes the solid manifold.  The move has to
    exceed OCC's vertex tolerances (up to ~1.7e-4 mm here): at 1e-5 LDU the
    mesher merged some copies and not others, leaving slits a slicer reports
    as open edges.
    """
    pins = pinch_fans(solid.faces)
    if not pins:
        return solid
    v, f = solid.verts.copy(), solid.faces.copy()
    for x, groups in pins.items():
        base = v[x].copy()
        moved = []
        for g in groups:
            d = v[np.unique(f[g][f[g] != x])].mean(0) - base
            moved.append(base + nudge * d / np.linalg.norm(d))
        v[x] = moved[0]
        for g, p in zip(groups[1:], moved[1:]):
            v = np.vstack([v, p])
            for t in g:
                f[t][f[t] == x] = len(v) - 1
    return Solid(v, f, solid.tri_source, solid.tri_instance, dict(solid.stats, pinches_separated=len(pins)))


def _point_tri_distance(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Distance from point p to each triangle in T (m, 3, 3)."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n) + 1e-30
    h = np.einsum("ij,ij->i", p - a, n) / nn
    q = p - h[:, None] * n
    w = q - a
    d00, d01, d11 = (np.einsum("ij,ij->i", ab, ab), np.einsum("ij,ij->i", ab, ac),
                     np.einsum("ij,ij->i", ac, ac))
    d20, d21 = np.einsum("ij,ij->i", w, ab), np.einsum("ij,ij->i", w, ac)
    den = d00 * d11 - d01 * d01 + 1e-30
    v = (d11 * d20 - d01 * d21) / den
    u = (d00 * d21 - d01 * d20) / den
    dist = np.where((v >= 0) & (u >= 0) & (v + u <= 1), np.abs(h) * np.sqrt(nn), np.inf)
    for x, y in ((a, b), (b, c), (c, a)):
        e = y - x
        t = np.clip(np.einsum("ij,ij->i", p - x, e) / (np.einsum("ij,ij->i", e, e) + 1e-30), 0, 1)
        dist = np.minimum(dist, np.linalg.norm(x + t[:, None] * e - p, axis=1))
    return dist


def off_surface(solid: Solid, flat: Flat, tol: float = WELD_TOL) -> int:
    """Triangles (caps aside) with a point farther than tol from every LDraw triangle.

    The union may cut and re-triangulate faces, but every point it outputs
    must lie on the input surface.  manifold3d invents faces when a piece
    overlaps itself (6377's rails and stud strips); this catches that.
    """
    src = flat.tris
    lo, hi = src.min(1) - tol, src.max(1) + tol
    bad = 0
    for t in np.flatnonzero(solid.tri_source >= 0):
        tri = solid.verts[solid.faces[t]]
        m = np.all((lo <= tri.max(0)) & (hi >= tri.min(0)), axis=1)
        cand = src[m]
        pts = np.vstack([tri.mean(0), (tri + np.roll(tri, -1, axis=0)) / 2])
        if not len(cand) or any(_point_tri_distance(p, cand).min() > tol for p in pts):
            bad += 1
    return bad


def export_check(solid: Solid) -> dict:
    """The mesh as a slicer reads it: float32 mm, vertices merged by position."""
    t = solid.to_print_frame()[solid.faces].astype(np.float32)
    _, inv = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    collapsed = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    f = f[~collapsed]
    o, _, more = edge_use(f)
    he = np.stack([f, np.roll(f, -1, axis=1)], axis=2).reshape(-1, 2)
    return {"collapsed": int(collapsed.sum()), "open_edges": o, "edges_3plus": more,
            "flipped_edges": len(he) - len(np.unique(he, axis=0))}


def validate(solid: Solid, flat: Flat) -> dict:
    """Checks a Tier 1 solid. `ok` is true only when every check passes.

    `export` repeats the edge checks on the float32 STL a slicer would read.
    """
    f, v = solid.faces, solid.verts
    o, two, more = edge_use(f)
    he = np.stack([f, np.roll(f, -1, axis=1)], axis=2).reshape(-1, 2)
    flipped = len(he) - len(np.unique(he, axis=0))    # same directed edge used twice
    lo, hi = flat.bounds()
    bounds_err = float(max(np.abs(v.min(0) - lo).max(), np.abs(v.max(0) - hi).max()))
    _, area = tri_normals_area(v[f])
    caps = solid.tri_source < 0
    r = {"tris": len(f), "open_edges": o, "edges_3plus": more, "flipped_edges": flipped,
         "volume_ldu3": signed_volume(v, f), "bounds_err_ldu": bounds_err,
         "cap_tris_left": int(caps.sum()), "cap_area_left": float(area[caps].sum()),
         "self_intersections": self_intersections(v, f), "export": export_check(solid),
         "pinch_vertices": len(pinch_fans(f)),          # reported, not an error
         "off_surface": off_surface(solid, flat)}
    r["ok"] = (o == 0 and more == 0 and flipped == 0 and r["volume_ldu3"] > 0
               and bounds_err < WELD_TOL and r["cap_tris_left"] == 0 and r["self_intersections"] == 0
               and r["off_surface"] == 0
               and not any(r["export"].values()))
    return r


# ----------------------------------------------------------------------------
# Export (millimetres, Z up)
# ----------------------------------------------------------------------------
def write_stl(path: str, tris: np.ndarray, header: bytes = b"ldraw2solid"):
    """Binary STL from (T, 3, 3) triangles."""
    n, _ = tri_normals_area(tris)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    with open(path, "wb") as fh:
        fh.write(header[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        fh.write(rec.tobytes())


def write_3mf(path: str, verts: np.ndarray, faces: np.ndarray, name: str = "part"):
    """Minimal 3MF with one object, units in millimetres."""
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
              f'<resources><object id="1" type="model" name="{name}"><mesh><vertices>\n')
    for x, y, z in verts.tolist():
        buf.write(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>\n')
    buf.write("</vertices><triangles>\n")
    for a, b, c in faces.tolist():
        buf.write(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>\n')
    buf.write('</triangles></mesh></object></resources>\n'
              '<build><item objectid="1"/></build></model>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
                   '</Relationships>')
        z.writestr("3D/3dmodel.model", buf.getvalue())
