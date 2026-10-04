# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tiers 2 and 3: build an OpenCASCADE solid from a Tier 1 mesh and write STEP.

Tier 2 merges connected coplanar triangles into planar faces.  Their boundary
loops come straight from the closed mesh, so neighbouring faces share every
boundary vertex exactly and sewing only has to match identical edges.

Everything here works in the print frame (millimetres, Z up).  The mesh is
converted once, through `Solid.to_print_frame()`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid,
                                BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Wire
from OCP.Geom import Geom_CylindricalSurface
from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Pln, gp_Pnt

from .mesh import Solid, separate_pinches, signed_volume, tri_normals_area
from .parser import Flat, LDRAW_TO_ZUP, LDU_TO_MM
from .primitives import surfaces

PLANE_TOL = 1e-3 * 0.4   # mm: a vertex this close to a face's plane counts as on it

# OCC prints a transfer banner on every STEP write; keep only failures.
for _p in Message.DefaultMessenger_s().Printers():
    _p.SetTraceLevel(Message_Gravity.Message_Fail)


class BrepError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Mesh -> face regions
# ----------------------------------------------------------------------------
def _edge_map(faces: np.ndarray) -> dict:
    """Directed edge (a, b) -> triangle that owns it."""
    out = {}
    for t, (a, b, c) in enumerate(faces.tolist()):
        out[(a, b)], out[(b, c)], out[(c, a)] = t, t, t
    return out


def planar_regions(verts: np.ndarray, faces: np.ndarray, tol: float = PLANE_TOL,
                   keys: list | None = None) -> np.ndarray:
    """Label connected groups of coplanar triangles.  -> (T,) region index.

    With `keys`, a triangle whose key is not None joins only neighbours with
    the same key (one analytic surface), whatever their planes.
    """
    n, _ = tri_normals_area(verts[faces])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    em = _edge_map(faces)
    for (a, b), t in em.items():
        u = em.get((b, a))
        if u is None or u < t:
            continue
        if keys is not None and (keys[t] is not None or keys[u] is not None):
            if keys[t] == keys[u]:
                parent[find(u)] = find(t)
            continue
        far = [v for v in faces[u] if v not in (a, b)][0]
        if n[t] @ n[u] > 0.999 and abs((verts[far] - verts[a]) @ n[t]) < tol:
            parent[find(u)] = find(t)
    roots = np.array([find(t) for t in range(len(faces))])
    return np.unique(roots, return_inverse=True)[1].reshape(-1)


def region_loops(faces: np.ndarray, labels: np.ndarray, em: dict | None = None,
                 verts: np.ndarray | None = None) -> dict:
    """{region: [loop, ...]}: boundary vertex loops, in the triangles' winding.

    A face can touch itself at a vertex (6377: two cavities meeting at a
    corner).  There, with `verts`, the walk takes the outgoing edge first
    clockwise from the way back, seen from the face's normal, which keeps
    following the same patch of face; the loop then passes that vertex twice,
    which OCC accepts.
    """
    em = _edge_map(faces) if em is None else em
    nxt = defaultdict(lambda: defaultdict(list))         # region -> vertex -> next vertices
    for (a, b), t in em.items():
        if labels[em[(b, a)]] != labels[t]:
            nxt[labels[t]][a].append(b)
    normal = {}
    if verts is not None:
        n, _ = tri_normals_area(verts[faces])
        for r in nxt:
            m = n[labels == r].sum(0)
            normal[r] = m / (np.linalg.norm(m) or 1)
    out = {}
    for r, step in nxt.items():
        # Successor of each boundary edge (u, x): the edge leaving x picked by
        # the turn rule.  Loops are the cycles of this map.
        succ = {(u, x): (x, _choose(verts, normal.get(r), u, x, step[x], r))
                for u, outs in list(step.items()) for x in outs}
        loops, seen = [], set()
        for h in succ:
            if h in seen:
                continue
            loop = []
            while h not in seen:
                seen.add(h)
                loop.append(h[0])
                h = succ[h]
            loops.append(loop)
        out[int(r)] = loops
    return out


def _choose(verts, n, u, x, outs, r):
    """Outgoing edge from x after arriving from u: the first clockwise from x->u."""
    if len(outs) == 1:
        return outs[0]
    if verts is None:
        raise BrepError(f"region {r} touches itself at vertex {x}")
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    def ang(w):
        d = verts[w] - verts[x]
        return np.arctan2(d @ e2, d @ e1)

    back = ang(u)
    return min(outs, key=lambda w: (back - ang(w)) % (2 * np.pi) or 2 * np.pi)


def touch_points(v: np.ndarray, loops: dict, tol: float = PLANE_TOL) -> np.ndarray:
    """Boundary vertices that lie inside another boundary edge.

    6377 has a connector hole tangent to a cavity wall: the hole's polygon has a
    vertex on the middle of the wall's edge.  OCC rejects a face whose boundary
    touches an edge like that, but accepts a wire that passes the same vertex
    twice, so these vertices are inserted into the edges they touch.
    """
    ids = np.unique(np.concatenate([np.concatenate(ls) for ls in loops.values() if ls]))
    P = v[ids]
    hits = set()
    for ls in loops.values():
        for loop in ls:
            for a, b in zip(loop, loop[1:] + loop[:1]):
                m = _on_segment(v[a], v[b], P, tol)
                hits.update(ids[m].tolist())
    return np.array(sorted(hits), dtype=np.int64)


def _on_segment(pa, pb, P, tol):
    """Mask of points P strictly inside segment pa-pb (within tol)."""
    lo, hi = np.minimum(pa, pb) - tol, np.maximum(pa, pb) + tol
    m = np.all((P >= lo) & (P <= hi), axis=1)
    if not m.any():
        return m
    d = pb - pa
    length = float(np.linalg.norm(d))
    q = P[m]
    t = (q - pa) @ d / (length * length)
    near = (np.linalg.norm(pa + t[:, None] * d - q, axis=1) < tol) & (t * length > tol) & ((1 - t) * length > tol)
    m[np.flatnonzero(m)[~near]] = False
    return m


def _with_pins(v, seq, pins, tol, closed=True):
    """`seq` with any pin vertex lying inside one of its edges inserted there.

    A closed loop also gets its closing edge checked; an open run keeps its end.
    """
    if not len(pins):
        return list(seq)
    pairs = zip(seq, seq[1:] + seq[:1]) if closed else zip(seq[:-1], seq[1:])
    out = []
    for a, b in pairs:
        out.append(a)
        m = _on_segment(v[a], v[b], v[pins], tol)
        if m.any():
            d = v[b] - v[a]
            out += pins[m][np.argsort((v[pins[m]] - v[a]) @ d)].tolist()
    return out if closed else out + [seq[-1]]


# ----------------------------------------------------------------------------
# OCC construction
# ----------------------------------------------------------------------------
def _pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _wire(points: np.ndarray):
    poly = BRepBuilderAPI_MakePolygon()
    for p in points:
        poly.Add(_pnt(p))
    poly.Close()
    if not poly.IsDone():
        raise BrepError("could not build a polygon wire")
    return poly.Wire()


def _planar_face(verts: np.ndarray, loops: list, normal: np.ndarray, pins=()):
    """Face on the best-fit plane; the loop with positive area is the outer one."""
    loops = [_with_pins(verts, l, np.asarray(pins, dtype=np.int64), PLANE_TOL) for l in loops]
    pts = np.concatenate([verts[l] for l in loops])
    c = pts.mean(0)
    e1 = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    def area(loop):
        p = verts[loop] - c
        x, y = p @ e1, p @ e2
        return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)

    areas = [area(l) for l in loops]
    outer = int(np.argmax(areas))
    if areas[outer] <= 0 or sum(a > 0 for a in areas) != 1:
        raise BrepError(f"face loops do not form one outer boundary with holes (areas {areas})")
    mk = BRepBuilderAPI_MakeFace(gp_Pln(_pnt(c), gp_Dir(*map(float, normal))), _wire(verts[loops[outer]]), True)
    for i, l in enumerate(loops):
        if i != outer:
            mk.Add(_wire(verts[l]))
    if not mk.IsDone():
        raise BrepError("could not build a planar face")
    return mk.Face()


def _solid_from_faces(faces_occ: list, tol: float = 1e-6):
    sew = BRepBuilderAPI_Sewing(tol)
    for f in faces_occ:
        sew.Add(f)
    sew.Perform()
    sewn = sew.SewedShape()
    shells = []
    ex = TopExp_Explorer(sewn, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if len(shells) != 1:
        raise BrepError(f"sewing gave {len(shells)} shells, expected 1")
    mk = BRepBuilderAPI_MakeSolid(shells[0])
    solid = mk.Solid()
    BRepLib.OrientClosedSolid_s(solid)
    return solid


@dataclass
class Brep:
    shape: object              # TopoDS_Solid (or TopoDS_Shape after a fix)
    face_regions: list         # per OCC face: set of Flat instance ids it came from
    stats: dict = field(default_factory=dict)


def faceted_brep(solid: Solid) -> Brep:
    """Tier 2: one planar face per connected coplanar group of triangles."""
    solid = separate_pinches(solid)
    v = solid.to_print_frame()
    f = solid.faces
    labels = planar_regions(v, f)
    loops = region_loops(f, labels, verts=v)
    pins = touch_points(v, loops)
    n, area = tri_normals_area(v[f])
    occ_faces, prov, max_dev = [], [], 0.0
    for r in range(labels.max() + 1):
        m = labels == r
        normal = (n[m]).sum(0)
        normal /= np.linalg.norm(normal)
        pts = v[np.unique(f[m])]
        max_dev = max(max_dev, float(np.abs((pts - pts.mean(0)) @ normal).max()))
        occ_faces.append(_planar_face(v, loops[r], normal, pins))
        prov.append(set(np.unique(solid.tri_instance[m]).tolist()))
    shape = _solid_from_faces(occ_faces)
    return Brep(shape, prov, {"faces": len(occ_faces), "max_plane_dev_mm": max_dev,
                              "touch_points": len(pins)})


# ----------------------------------------------------------------------------
# Tier 3: true cylinders
# ----------------------------------------------------------------------------
@dataclass
class _Cyl:
    origin: np.ndarray        # mm, the axis point nearest the world origin
    d: np.ndarray             # unit axis, sign normalised
    r: float                  # mm
    hole: bool
    key: tuple                # equal for primitives on the same cylinder


def _frame(d: np.ndarray):
    x = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return x, np.cross(d, x)


def _cylinders_mm(flat: Flat) -> dict:
    """{instance id: _Cyl} for exact cylinder primitives, in the print frame."""
    out = {}
    for iid, s in surfaces(flat).items():
        if s.kind != "cylinder" or not s.exact:
            continue
        o = LDRAW_TO_ZUP @ s.origin * LDU_TO_MM
        d = LDRAW_TO_ZUP @ s.axis
        d = d / np.linalg.norm(d)
        if d[np.flatnonzero(np.abs(d) > 1e-9)[0]] < 0:
            d = -d
        p0 = o - (o @ d) * d
        r = s.radius * LDU_TO_MM
        key = (*np.round(d, 6).tolist(), *np.round(p0, 5).tolist(), round(r, 5), s.inverted)
        out[iid] = _Cyl(p0, d, r, s.inverted, key)
    return out


def _runs(loop: list, nb: list) -> list:
    """Split a loop into maximal runs with one neighbour: [(vertices, neighbour)].

    A loop with a single neighbour gives one run whose first and last vertex
    are the same.
    """
    n = len(loop)
    if all(x == nb[0] for x in nb):
        return [(loop + loop[:1], nb[0])]
    s = next(i for i in range(n) if nb[i] != nb[i - 1])
    loop, nb = loop[s:] + loop[:s], nb[s:] + nb[:s]
    out, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and nb[j + 1] == nb[i]:
            j += 1
        out.append((loop[i:j + 2] if j + 1 < n else loop[i:] + loop[:1], nb[i]))
        i = j + 1
    return out


def _cyl_coords(c: _Cyl, p: np.ndarray):
    """-> (height along axis, angle, radial distance) of points p."""
    x, y = _frame(c.d)
    rel = p - c.origin
    h = rel @ c.d
    rad = rel - np.outer(h, c.d)
    return h, np.arctan2(rad @ y, rad @ x), np.linalg.norm(rad, axis=1)


AXIS_PARALLEL = 1e-4      # |normal . axis| below this: a wall along the cylinder


def _polygon_angles(ang: np.ndarray, r: float, tol: float) -> np.ndarray:
    """Distinct polygon-vertex angles (sorted, in [0, 2pi)) from vertex angles."""
    a = np.sort(np.mod(ang, 2 * np.pi))
    groups = [[a[0]]]
    for x in a[1:]:
        if x - groups[-1][-1] < tol / r:
            groups[-1].append(x)
        else:
            groups.append([x])
    if len(groups) > 1 and groups[0][0] + 2 * np.pi - groups[-1][-1] < tol / r:
        groups[0] = [x - 2 * np.pi for x in groups.pop()] + groups[0]
    return np.mod(np.array([np.mean(g) for g in groups]), 2 * np.pi)


def _facet_of(angles: np.ndarray, a: float) -> int:
    """Index k of the facet [angles[k-1], angles[k]] (cyclic) that contains angle a."""
    return int(np.searchsorted(angles, np.mod(a, 2 * np.pi)) % len(angles))


def _on_chord(c: _Cyl, angles, h, a, rad, tol) -> bool:
    """Is a point (height h, angle a, radius rad) on the chord of its facet?"""
    k = _facet_of(angles, a)
    a0, a1 = angles[k - 1], angles[k]
    half = np.mod(a1 - a0, 2 * np.pi) / 2
    off = np.angle(np.exp(1j * (a - (a0 + half))))           # angle from the facet's middle
    return abs(rad * np.cos(off) - c.r * np.cos(half)) <= tol and abs(off) <= half + 1e-9


def _plan_lift(c: _Cyl, v, tris, loops, nb_of, normals, cyl_regions, tol):
    """Decide whether one cylinder region can become true cylinder faces.

    The region is cut into facet columns (one 16-gon facet each) with their
    height ranges; neighbouring columns with equal ranges form rectangular
    patches in (angle, height), one cylindrical face each.  So a stepped
    cylinder (6377: a tube that runs higher over 3/8 of its circle) lifts too.
    Boundary vertices must be on the circle or on a facet's chord; the latter
    occur where a flat wall continues a facet's plane, and the flat face next to
    that facet then gains the crescent between chord and arc.
    -> (info dict, '') when it can, (None, reason) when it stays faceted.
    """
    if not loops:
        return None, "no boundary"
    pts = np.unique(np.concatenate(loops))
    h, ang, rad = _cyl_coords(c, v[pts])
    on = np.abs(rad - c.r) <= tol
    if on.sum() < 3:
        return None, "too few vertices on the circle"
    angles = _polygon_angles(ang[on], c.r, tol)
    for x in np.flatnonzero(~on):
        if not _on_chord(c, angles, h[x], ang[x], rad[x], tol):
            return None, "vertex off the circle"
    chord_walls, crescent_takers = set(), set()     # (facet, height) pairs
    level = lambda x: round(float(x) / tol)
    for loop in loops:
        for run, nb in _runs(loop, nb_of(loop)):
            if nb in cyl_regions:
                return None, "meets another cylinder"
            along = abs(normals[nb] @ c.d)
            rh = _cyl_coords(c, v[run])[0]
            if along > 1 - 1e-6:
                if np.ptp(rh) > tol:
                    return None, "boundary on a cross plane changes height"
                if run[0] != run[-1]:
                    _, ea, er = _cyl_coords(c, v[[run[0], run[-1]]])
                    for a_, r_ in zip(ea, er):
                        if abs(r_ - c.r) > tol:
                            crescent_takers.add((_facet_of(angles, a_), level(rh[0])))
            elif along > AXIS_PARALLEL:
                return None, "boundary is not a circle or a straight generator"
            else:
                # Along a wall: each step is a straight generator (same angle,
                # on the circle) or runs along a chord at one height.
                rh, ra, rr = _cyl_coords(c, v[run])
                for i in range(len(run) - 1):
                    vertical = (abs(np.angle(np.exp(1j * (ra[i + 1] - ra[i])))) * c.r < tol
                                and abs(rr[i] - c.r) <= tol and abs(rr[i + 1] - c.r) <= tol)
                    flat_step = abs(rh[i + 1] - rh[i]) <= tol
                    if not (vertical or flat_step):
                        return None, "boundary is not a circle or a straight generator"
                    if flat_step and not vertical:
                        mid = np.angle(np.exp(1j * ra[i]) + np.exp(1j * ra[i + 1]))
                        chord_walls.add((_facet_of(angles, mid), level(rh[i])))
    # A wall continuing a facet's plane leaves a crescent between chord and arc
    # once the facet is curved; a cross face must end on that chord to take it.
    if chord_walls - crescent_takers:
        return None, "a wall continues a facet with no face to close the gap"
    # Facet columns and their height ranges.
    cols = defaultdict(list)
    for t in tris:
        th, ta, _ = _cyl_coords(c, v[t])
        cols[_facet_of(angles, np.angle(np.exp(1j * ta).mean()))].append((th.min(), th.max()))
    span_of = {}
    for k, iv in cols.items():
        lo, hi = min(x[0] for x in iv), max(x[1] for x in iv)
        cover = sorted(iv)
        reach = cover[0][1]
        for x0, x1 in cover[1:]:
            if x0 > reach + tol:
                return None, "a facet column has a gap"
            reach = max(reach, x1)
        span_of[k] = (lo, hi)
    n = len(angles)
    width = lambda k: np.mod(angles[k] - angles[k - 1], 2 * np.pi) or 2 * np.pi
    same = lambda p, q: abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol
    if len(span_of) == n and all(same(span_of[k], span_of[0]) for k in span_of):
        patches = [(0.0, 2 * np.pi, *span_of[0])]          # full band, seam at angle 0
    else:
        # Start just after a column that is missing or differs from its successor.
        starts = [k for k in range(n) if k in span_of and
                  ((k - 1) % n not in span_of or not same(span_of[(k - 1) % n], span_of[k]))]
        if not starts:
            return None, "could not split into patches"
        patches = []
        for k0 in starts:
            k, u_span = k0, 0.0
            while k in span_of and same(span_of[k], span_of[k0]) and (k != k0 or u_span == 0):
                u_span += width(k)
                k = (k + 1) % n
            patches.append((float(angles[k0 - 1]), float(u_span), *span_of[k0]))
    delta = sum((hi - lo) * c.r ** 2 / 2 * (width(k) - np.sin(width(k)))
                for k, (lo, hi) in span_of.items()) * (-1 if c.hole else 1)
    return {"cyl": c, "angles": angles, "patches": patches, "volume_delta": delta}, ""


def _cyl_faces(info) -> list:
    """One cylindrical face per (angle, height) patch."""
    c = info["cyl"]
    x, _ = _frame(c.d)
    ax = gp_Ax3(_pnt(c.origin), gp_Dir(*c.d.tolist()), gp_Dir(*x.tolist()))
    surf = Geom_CylindricalSurface(ax, c.r)
    out = []
    for u0, span, v0, v1 in info["patches"]:
        mk = BRepBuilderAPI_MakeFace(surf, u0, u0 + span, v0, v1, 1e-7)
        if not mk.IsDone():
            raise BrepError("could not build a cylindrical face")
        face = mk.Face()
        out.append(TopoDS.Face(face.Reversed()) if c.hole else face)
    return out


def _circle_point(c: _Cyl, a: float, h: float) -> np.ndarray:
    x, y = _frame(c.d)
    return c.origin + h * c.d + c.r * (np.cos(a) * x + np.sin(a) * y)


def _arc_edge(c: _Cyl, pts: np.ndarray, closed: bool, va=None, vb=None):
    """Arc of the cylinder's circle through `pts`, from va to vb when given."""
    h = float((pts[0] - c.origin) @ c.d)
    centre = c.origin + h * c.d
    rel = pts - centre
    ccw = float((np.cross(rel[:-1], rel[1:]) @ c.d).sum()) > 0
    # A full circle starts at the cylinder face's seam, so sewing need not split it.
    x = _frame(c.d)[0] if closed else rel[0] / np.linalg.norm(rel[0])
    circ = gp_Circ(gp_Ax2(_pnt(centre), gp_Dir(*c.d.tolist()), gp_Dir(*x.tolist())), c.r)
    a = va if va is not None else _pnt(pts[0])
    b = vb if vb is not None else _pnt(pts[-1])
    if closed:
        mk = BRepBuilderAPI_MakeEdge(circ)
    elif ccw:
        mk = BRepBuilderAPI_MakeEdge(circ, a, b)
    else:
        mk = BRepBuilderAPI_MakeEdge(circ, b, a)
    if not mk.IsDone():
        raise BrepError(f"could not build an arc edge (error {mk.Error()})")
    e = mk.Edge()
    return e if ccw else TopoDS.Edge(e.Reversed())


def _mixed_wire(v, loop, nb, lifted, tol, pins=np.zeros(0, dtype=np.int64)):
    """Wire for one planar-face loop; runs along a lifted cylinder become arcs or one line.

    Runs also break at pin vertices (see `touch_points`), and pins lying inside
    a straight edge are inserted into it.
    """
    pinned = set(pins.tolist()) | {x for x in loop if loop.count(x) > 1}
    breaks = [i for i, x in enumerate(loop) if x in pinned]
    # Block number changes at each pin; the stretch after the last pin wraps
    # round to the first block, so a loop start that is no pin is no break.
    block = [sum(1 for j in breaks if j <= i) % max(len(breaks), 1) for i in range(len(loop))]
    # Segments in loop order: ('arc', cylinder, points) or ('line', points).
    segs = []
    for run, (r, _) in _runs(loop, list(zip(nb, block))):
        info = lifted.get(r)
        if info is not None and np.ptp((v[run] - info["cyl"].origin) @ info["cyl"].d) < tol:
            segs += _arc_run(info, v, run, tol)
            continue
        if info is not None:                 # wall along the cylinder: keep its corners only
            run = _corners(v, run, tol)
        pts = v[_with_pins(v, run, pins, tol, closed=False)]
        segs += [("line", None, pts[i:i + 2], False) for i in range(len(pts) - 1)]
    # Build the wire from explicit vertices, one per position along the loop.
    # MakeWire would join edges by position and so fuse the two visits of a
    # point the loop passes twice (a tangent hole), which OCC then rejects.
    builder = BRep_Builder()
    wire = TopoDS_Wire()
    builder.MakeWire(wire)
    if len(segs) == 1 and segs[0][3]:
        builder.Add(wire, _arc_edge(segs[0][1], segs[0][2], True))
        return wire
    verts = [BRepBuilderAPI_MakeVertex(_pnt(seg[2][0])).Vertex() for seg in segs]
    for i, (kind, cyl, pts, _) in enumerate(segs):
        va, vb = verts[i], verts[(i + 1) % len(segs)]
        if kind == "arc":
            builder.Add(wire, _arc_edge(cyl, pts, False, va, vb))
        else:
            builder.Add(wire, BRepBuilderAPI_MakeEdge(va, vb).Edge())
    wire.Closed(True)
    return wire


def _arc_run(info, v, run, tol) -> list:
    """Segments for a run along a lifted cylinder at one height.

    A run end on a facet's chord (not on the circle) is where a flat wall
    continues that facet.  The arc then runs on to the facet's corner and a
    straight edge comes back along the chord: this face takes the crescent
    between chord and arc.
    """
    c, angles = info["cyl"], info["angles"]
    pts = v[run]
    if run[0] == run[-1]:
        return [("arc", c, pts, True)]
    h, a, rad = _cyl_coords(c, pts)
    centre = c.origin + h[0] * c.d
    ccw = float((np.cross(pts[:-1] - centre, pts[1:] - centre) @ c.d).sum()) > 0
    arc, before, after = list(pts), [], []
    if abs(rad[0] - c.r) > tol:
        k = _facet_of(angles, a[0])
        q = _circle_point(c, angles[k - 1] if ccw else angles[k], h[0])
        before = [("line", None, np.array([pts[0], q]), False)]
        arc[0] = q
    if abs(rad[-1] - c.r) > tol:
        k = _facet_of(angles, a[-1])
        q = _circle_point(c, angles[k] if ccw else angles[k - 1], h[0])
        after = [("line", None, np.array([q, pts[-1]]), False)]
        arc[-1] = q
    return before + [("arc", c, np.array(arc), False)] + after


def _corners(v, run, tol) -> list:
    """The run without vertices lying on the straight line between their neighbours."""
    out = [run[0]]
    for i in range(1, len(run) - 1):
        p, q, w = v[out[-1]], v[run[i]], v[run[i + 1]]
        d = w - p
        if np.linalg.norm(np.cross(q - p, d)) / (np.linalg.norm(d) or 1) > tol:
            out.append(run[i])
    return out + [run[-1]]


def _mixed_planar_face(v, loops, nbs, normal, lifted, tol, pins=np.zeros(0, dtype=np.int64)):
    pts = np.concatenate([v[l] for l in loops])
    c = pts.mean(0)
    e1, e2 = _frame(normal)

    def area(loop):
        p = v[loop] - c
        x, y = p @ e1, p @ e2
        return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)

    areas = [area(l) for l in loops]
    outer = int(np.argmax(areas))
    if areas[outer] <= 0 or sum(a > 0 for a in areas) != 1:
        raise BrepError(f"face loops do not form one outer boundary with holes (areas {areas})")
    pln = gp_Pln(_pnt(c), gp_Dir(*map(float, normal)))
    mk = BRepBuilderAPI_MakeFace(pln, _mixed_wire(v, loops[outer], nbs[outer], lifted, tol, pins), True)
    for i, l in enumerate(loops):
        if i != outer:
            mk.Add(_mixed_wire(v, l, nbs[i], lifted, tol, pins))
    if not mk.IsDone():
        raise BrepError("could not build a planar face")
    return mk.Face()


def analytic_brep(solid: Solid, flat: Flat, tol: float = PLANE_TOL) -> Brep:
    """Tier 3: exact cylinders become cylindrical faces, everything else as in Tier 2.

    A cylinder is lifted only when it is a simple band whose neighbours are
    planes across its axis (circle edges) or along it (straight edges);
    otherwise it stays faceted.  Boundary vertices of lifted cylinders are
    moved onto the true circle (by at most `tol`) before any face is built,
    so every face sees the same coordinates.
    """
    solid = separate_pinches(solid)
    v = solid.to_print_frame()
    f = solid.faces
    cyl_of = _cylinders_mm(flat)
    em = _edge_map(f)
    n, _ = tri_normals_area(v[f])

    def regions(keys):
        labels = planar_regions(v, f, tol, keys)
        loops = region_loops(f, labels, em, v)
        normals = {}
        for r in range(labels.max() + 1):
            s = n[labels == r].sum(0)
            normals[r] = s / (np.linalg.norm(s) or 1)
        return labels, loops, normals

    def nb_fn(labels):
        return lambda loop: [int(labels[em[(b, a)]]) for a, b in zip(loop, loop[1:] + loop[:1])]

    # 1. Every exact cylinder is a candidate; decide which ones can be lifted.
    keys = [cyl_of[i].key if i in cyl_of else None for i in solid.tri_instance.tolist()]
    labels, loops, normals = regions(keys)
    cyl_regions = {int(labels[t]): cyl_of[i] for t, i in enumerate(solid.tri_instance.tolist()) if i in cyl_of}
    lift1, fallback = {}, {}          # keyed by region in this first labelling
    for r, c in cyl_regions.items():
        info, why = _plan_lift(c, v, f[labels == r], loops.get(r, []), nb_fn(labels),
                               normals, cyl_regions, tol)
        if info is None:
            fallback[r] = why
        else:
            lift1[r] = info

    # 2. Move vertices of lifted cylinders onto the true circle, at the exact
    #    polygon angle, so a step's straight edge runs exactly along the axis.
    #    Vertices on a chord are left alone.
    for t, r in enumerate(labels.tolist()):
        if r in lift1:
            info = lift1[r]
            c = info["cyl"]
            for k in f[t]:
                h, a, rad = _cyl_coords(c, v[k][None])
                if abs(rad[0] - c.r) <= tol:
                    ang = info["angles"]
                    near = ang[np.argmin(np.abs(np.angle(np.exp(1j * (ang - a[0])))))]
                    v[k] = _circle_point(c, near, h[0])

    # 3. Regions again, with only the lifted cylinders kept whole.  Keying them
    #    by first-pass region keeps two separate pieces of one cylinder apart.
    keys = [r if r in lift1 else None for r in labels.tolist()]
    labels, loops, normals = regions(keys)
    lifted = {}
    for t, k in enumerate(keys):
        if k is not None:
            lifted[int(labels[t])] = lift1[k]
    nb_of = nb_fn(labels)
    pins = touch_points(v, loops, tol)
    occ_faces, prov = [], []
    for r in range(labels.max() + 1):
        if r in lifted:
            faces_r = _cyl_faces(lifted[r])
            occ_faces += faces_r
            prov += [set(np.unique(solid.tri_instance[labels == r]).tolist())] * (len(faces_r) - 1)
        else:
            occ_faces.append(_mixed_planar_face(v, loops[r], [nb_of(l) for l in loops[r]],
                                                normals[r], lifted, tol, pins))
        prov.append(set(np.unique(solid.tri_instance[labels == r]).tolist()))
    shape = _solid_from_faces(occ_faces, 1e-6)
    # Expected volume: the mesh after snapping, plus the slivers between each
    # lifted cylinder and its polygon.
    delta = float(sum(i["volume_delta"] for i in lifted.values()))
    return Brep(shape, prov, {
        "faces": len(occ_faces), "cylinders_lifted": len(lifted), "touch_points": len(pins),
        "cylinders_faceted": sorted(fallback.values()),
        "volume_delta_mm3": delta, "expected_volume_mm3": signed_volume(v, f) + delta})


# ----------------------------------------------------------------------------
# Checks and STEP I/O
# ----------------------------------------------------------------------------
def _count(shape, kind) -> int:
    ex, k = TopExp_Explorer(shape, kind), 0
    while ex.More():
        k += 1
        ex.Next()
    return k


def face_types(shape) -> dict:
    names = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
             GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus"}
    out = defaultdict(int)
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out[names.get(BRepAdaptor_Surface(TopoDS.Face(ex.Current())).GetType(), "other")] += 1
        ex.Next()
    return dict(out)


def check(shape, mesh_volume_mm3: float | None = None, bounds_mm=None) -> dict:
    """Validity, closedness, volume and bounds of an OCC shape."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    lo = np.array(box.CornerMin().Coord())
    hi = np.array(box.CornerMax().Coord())
    fb = ShapeAnalysis_FreeBounds(shape)
    r = {"valid": BRepCheck_Analyzer(shape).IsValid(),
         "solids": _count(shape, TopAbs_SOLID), "shells": _count(shape, TopAbs_SHELL),
         "faces": _count(shape, TopAbs_FACE),
         "free_edges": _count(fb.GetOpenWires(), TopAbs_EDGE) + _count(fb.GetClosedWires(), TopAbs_EDGE),
         "volume_mm3": props.Mass(), "face_types": face_types(shape)}
    if mesh_volume_mm3 is not None:
        r["volume_err_mm3"] = abs(props.Mass() - mesh_volume_mm3)
    if bounds_mm is not None:
        r["bounds_err_mm"] = float(max(np.abs(lo - bounds_mm[0]).max(), np.abs(hi - bounds_mm[1]).max()))
    return r


def tessellation_check(shape, deflection: float = 0.01) -> dict:
    """The solid as a slicer sees a STEP file: meshed, then welded by float32 position.

    -> {'tris', 'open_edges', 'edges_3plus'}.  A B-rep can be valid while its
    faces mesh with slits between them (vertices a few nanometres apart).
    """
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.2, True)
    tris = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        fc = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(fc, loc)
        if t is not None:
            trsf = loc.Transformation()
            p = np.array([[q.X(), q.Y(), q.Z()] for q in
                          (t.Node(i).Transformed(trsf) for i in range(1, t.NbNodes() + 1))])
            idx = np.array([t.Triangle(i).Get() for i in range(1, t.NbTriangles() + 1)]) - 1
            tris.append(p[idx])
        ex.Next()
    tris = np.concatenate(tris).astype(np.float32)
    _, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"tris": len(f), "open_edges": int((cnt == 1).sum()), "edges_3plus": int((cnt > 2).sum())}


def fix(shape):
    sf = ShapeFix_Shape(shape)
    sf.Perform()
    return sf.Shape()


def write_step(path: str, shape, name: str = "part"):
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.product.name", name)
    w = STEPControl_Writer()
    if w.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or w.Write(path) != IFSelect_RetDone:
        raise BrepError(f"STEP export failed for {path}")


def read_step(path: str):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise BrepError(f"could not read {path}")
    r.TransferRoots()
    return r.OneShape()


def mesh_reference(solid: Solid):
    """(volume mm^3, (lo, hi) mm) of the Tier 1 mesh in the print frame."""
    solid = separate_pinches(solid)
    v = solid.to_print_frame()
    return signed_volume(v, solid.faces), (v.min(0), v.max(0))
