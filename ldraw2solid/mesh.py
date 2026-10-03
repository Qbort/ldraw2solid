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
        tri = np.asarray(mf.triangulate(poly))
        faces.append(np.concatenate(idx)[tri])
        face_loop.append(np.concatenate([[i] * len(loop) for loop, i in group])[tri[:, 0]])
    for loop, i in fanned:
        k = len(verts) + len(extra)
        extra.append(verts[loop].mean(0))
        faces.append(np.array([(k, b, a) for a, b in zip(loop, loop[1:] + loop[:1])]))
        face_loop.append(np.full(len(loop), i))
    faces = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
    face_loop = np.concatenate(face_loop) if face_loop else np.zeros(0, dtype=np.int64)
    extra = np.array(extra, dtype=float).reshape(-1, 3)
    return faces, extra, face_loop, split


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
    pieces, piece_vol, n_split, n_fan = [], [], 0, 0
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


def validate(solid: Solid, flat: Flat) -> dict:
    """Checks a Tier 1 solid. `ok` is true only when every check passes."""
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
         "self_intersections": self_intersections(v, f)}
    r["ok"] = (o == 0 and more == 0 and flipped == 0 and r["volume_ldu3"] > 0
               and bounds_err < WELD_TOL and r["cap_tris_left"] == 0 and r["self_intersections"] == 0)
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
