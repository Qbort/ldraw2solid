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
(`tri_source`, -1 for a surviving cap), so provenance survives.

Welding, edge counting and the export helpers are numpy-only.  manifold3d is
imported by `solidify` alone.
"""
from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .parser import Flat, LDRAW_TO_ZUP, LDU_TO_MM

WELD_TOL = 1e-3   # LDU


# ----------------------------------------------------------------------------
# Welding and topology
# ----------------------------------------------------------------------------
def weld(tris: np.ndarray, tol: float = WELD_TOL):
    """Merge vertices that round to the same `tol` grid point.

    -> (verts, faces, keep).  `verts` keeps the first original position of each
    welded vertex, `keep` marks triangles still non-degenerate after welding.
    """
    pts = tris.reshape(-1, 3)
    key = np.round(pts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    faces = inv.reshape(-1, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return pts[first], faces, keep


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


def boundary_loops(faces: np.ndarray) -> list:
    """Chains of open half-edges, each a closed list of vertex indices."""
    he, _, _ = open_half_edges(faces)
    nxt = {}
    for a, b in he.tolist():
        if a in nxt:
            raise SolidifyError(f"vertex {a} starts two open edges; boundary is not a set of simple loops")
        nxt[a] = b
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
            if v is None:
                raise SolidifyError(f"open edges starting at vertex {start} do not form a loop")
        if v != start:
            raise SolidifyError(f"open edges starting at vertex {start} run into another loop")
        loops.append(loop)
    return loops


def cap_loops(verts, loops, tol: float = WELD_TOL):
    """Faces that close the given loops, running against their open edges.

    Loops on one plane are triangulated together, so nested loops give a ring.
    Non-planar loops are fanned to their centroid, which adds one vertex each.
    -> (faces (C, 3), extra vertices (E, 3), {'planar': n, 'nonplanar': n})
    """
    import manifold3d as mf

    groups, nonplanar = [], []          # groups: [centre, normal, [loop, ...]]
    for loop in loops:
        p = verts[loop]
        c = p.mean(0)
        n = np.linalg.svd(p - c)[2][2]
        if np.abs((p - c) @ n).max() > tol:
            nonplanar.append(loop)
            continue
        for g in groups:
            if abs(abs(g[1] @ n) - 1) < 1e-6 and abs((c - g[0]) @ g[1]) < tol:
                g[2].append(loop)
                break
        else:
            groups.append([c, n, [loop]])

    faces, extra = [], []
    for c, n, group in groups:
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        idx = [np.array(loop[::-1]) for loop in group]    # cap runs against the open edges
        poly = [np.c_[(verts[i] - c) @ e1, (verts[i] - c) @ e2] for i in idx]
        area = sum((p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]).sum()
                   for p in poly)
        if area < 0:                                       # triangulate wants outer loops CCW
            poly = [p * [1.0, -1.0] for p in poly]
        tri = np.asarray(mf.triangulate(poly))
        faces.append(np.concatenate(idx)[tri])
    for loop in nonplanar:
        k = len(verts) + len(extra)
        extra.append(verts[loop].mean(0))
        faces.append(np.array([(k, b, a) for a, b in zip(loop, loop[1:] + loop[:1])]))
    faces = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
    extra = np.array(extra, dtype=float).reshape(-1, 3)
    return faces, extra, {"planar": sum(len(g[2]) for g in groups), "nonplanar": len(nonplanar)}


def components(faces: np.ndarray) -> list:
    """Groups of face indices connected through shared vertices."""
    parent = list(range(int(faces.max()) + 1 if len(faces) else 0))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(f[0]) for f in faces.tolist()])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


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
    tri_instance: np.ndarray   # (T,) Flat instance id, -1 for a cap
    stats: dict = field(default_factory=dict)

    def to_print_frame(self) -> np.ndarray:
        """Vertices in millimetres, Z up, winding preserved."""
        return (self.verts @ LDRAW_TO_ZUP.T) * LDU_TO_MM


def solidify(flat: Flat, tol: float = WELD_TOL) -> Solid:
    """Weld, split T-junctions, cap open loops and union the closed pieces.

    `tol` is both the weld distance and the precision given to the union.
    """
    import manifold3d as mf

    verts, faces, keep = weld(flat.tris, tol)
    src = np.flatnonzero(keep)
    faces = faces[keep]
    faces, src, n_tj = split_t_junctions(verts, faces, src, tol)
    loops = boundary_loops(faces)
    cap_f, extra, n_loops = cap_loops(verts, loops, tol)
    verts = np.concatenate([verts, extra])
    allf = np.concatenate([faces, cap_f])
    allsrc = np.concatenate([src, np.full(len(cap_f), -1, dtype=src.dtype)])

    pieces, piece_vol = [], []
    for comp in components(allf):
        used, local = np.unique(allf[comp], return_inverse=True)
        mesh = mf.Mesh64(np.array(verts[used], dtype=np.float64, order="C"),
                         np.array(local.reshape(-1, 3), dtype=np.uint64, order="C"),
                         face_id=np.array(comp, dtype=np.uint64))
        man = mf.Manifold(mesh)
        if man.status() != mf.Error.NoError:
            o, _, m = edge_use(allf[comp])
            raise SolidifyError(f"piece of {len(comp)} triangles is not closed "
                                f"({man.status()}, {o} open edges, {m} shared by 3+)")
        piece_vol.append(man.volume())
        # Library coordinates have 4 decimals, so a cap can poke ~1e-4 LDU out of
        # the surface it rests on.  Treat that as touching, not as a sliver.
        pieces.append(man.set_tolerance(tol))
    if min(piece_vol) <= 0:
        raise SolidifyError(f"a closed piece has non-positive volume: {sorted(piece_vol)[:3]}")

    result = mf.Manifold.batch_boolean(pieces, mf.OpType.Add)
    out = result.to_mesh64()
    out_v = np.asarray(out.vert_properties)[:, :3].astype(float)
    out_f = np.asarray(out.tri_verts).astype(np.int64)
    tri_source = allsrc[np.asarray(out.face_id).astype(np.int64)]
    tri_instance = np.where(tri_source >= 0, flat.tri_instance[np.maximum(tri_source, 0)], -1)
    stats = {"welded_tris": int(keep.sum()), "tjunction_verts": n_tj,
             "loops_planar": n_loops["planar"], "loops_nonplanar": n_loops["nonplanar"],
             "cap_tris": len(cap_f), "pieces": len(pieces)}
    return Solid(out_v, out_f, tri_source, tri_instance, stats)


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
