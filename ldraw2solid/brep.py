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

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
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
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

from .mesh import Solid, tri_normals_area, signed_volume

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


def planar_regions(verts: np.ndarray, faces: np.ndarray, tol: float = PLANE_TOL) -> np.ndarray:
    """Label connected groups of coplanar triangles.  -> (T,) region index."""
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
        far = [v for v in faces[u] if v not in (a, b)][0]
        if n[t] @ n[u] > 0.999 and abs((verts[far] - verts[a]) @ n[t]) < tol:
            parent[find(u)] = find(t)
    roots = np.array([find(t) for t in range(len(faces))])
    return np.unique(roots, return_inverse=True)[1].reshape(-1)


def region_loops(faces: np.ndarray, labels: np.ndarray, em: dict | None = None) -> dict:
    """{region: [loop, ...]}: boundary vertex loops, in the triangles' winding."""
    em = _edge_map(faces) if em is None else em
    nxt = defaultdict(dict)
    for (a, b), t in em.items():
        if labels[em[(b, a)]] != labels[t]:
            if a in nxt[labels[t]]:
                raise BrepError(f"region {labels[t]} touches itself at vertex {a}")
            nxt[labels[t]][a] = b
    out = {}
    for r, step in nxt.items():
        loops, seen = [], set()
        for start in step:
            if start in seen:
                continue
            loop, v = [], start
            while v not in seen:
                seen.add(v)
                loop.append(v)
                v = step[v]
            loops.append(loop)
        out[int(r)] = loops
    return out


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


def _planar_face(verts: np.ndarray, loops: list, normal: np.ndarray):
    """Face on the best-fit plane; the loop with positive area is the outer one."""
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
    v = solid.to_print_frame()
    f = solid.faces
    labels = planar_regions(v, f)
    loops = region_loops(f, labels)
    n, area = tri_normals_area(v[f])
    occ_faces, prov, max_dev = [], [], 0.0
    for r in range(labels.max() + 1):
        m = labels == r
        normal = (n[m]).sum(0)
        normal /= np.linalg.norm(normal)
        pts = v[np.unique(f[m])]
        max_dev = max(max_dev, float(np.abs((pts - pts.mean(0)) @ normal).max()))
        occ_faces.append(_planar_face(v, loops[r], normal))
        prov.append(set(np.unique(solid.tri_instance[m]).tolist()))
    shape = _solid_from_faces(occ_faces)
    return Brep(shape, prov, {"faces": len(occ_faces), "max_plane_dev_mm": max_dev})


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
    v = solid.to_print_frame()
    return signed_volume(v, solid.faces), (v.min(0), v.max(0))
