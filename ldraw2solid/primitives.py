"""
Map LDraw primitive instances to analytic surfaces.

`classify(name)` recognises a primitive by file name and returns its surface in
the primitive's own unit coordinates.  `surface_of(flat, instance_id)` walks up
the reference chain of a triangle, finds the nearest recognised primitive and
returns the surface placed in model coordinates, with a flag saying whether the
placement keeps it exact (a scaled cylinder is still a cylinder; a sheared one
is not).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .parser import Flat, Instance

_FRAC = r"(\d+)-(\d+)"
_RULES = [
    # (regex on bare name, kind)
    (re.compile(rf"^{_FRAC}cyl[io]\d*$"), "cylinder"),
    (re.compile(rf"^{_FRAC}cyls\d*$"), "cylinder"),          # obliquely cut cylinder
    (re.compile(rf"^{_FRAC}con(\d+)$"), "cone"),
    (re.compile(rf"^{_FRAC}(?:disc|ndis|tndis|chrd|tang)$"), "plane"),
    (re.compile(rf"^{_FRAC}(?:ring|rin|ri)(\d+)$"), "plane"),
    (re.compile(rf"^{_FRAC}sphe$"), "sphere"),
    (re.compile(r"^[tr](\d\d)([ioq])(\d{4})$"), "torus"),
    (re.compile(rf"^{_FRAC}edge$"), "circle"),               # lines only
]
_FEATURES = [
    (re.compile(r"^stud(4|3|16|18)"), "tube"),        # underside tubes and pins
    (re.compile(r"^stud"), "stud"),
    (re.compile(r"^(peghole|npeghol)"), "peghole"),
    (re.compile(r"^(axlehol|axl[0-9e])"), "axlehole"),
    (re.compile(r"^connhol"), "connhole"),
]


@dataclass
class Primitive:
    kind: str                 # cylinder | cone | plane | sphere | torus | circle
    sweep: float = 1.0        # fraction of a full turn
    r0: float = 1.0           # radius at y=0 (cone/cylinder); inner radius (ring)
    r1: float = 1.0           # radius at y=1 (cone/cylinder); outer radius (ring)
    minor: float = 0.0        # torus tube radius
    shape: str = ""           # disc | ndis | ring | chrd | ... for planar ones


def bare(name: str) -> str:
    n = name.lower().replace("\\", "/").split("/")[-1]
    return n[:-4] if n.endswith(".dat") else n


def classify(name: str, description: str = "") -> Optional[Primitive]:
    b = bare(name)
    for rx, kind in _RULES:
        m = rx.match(b)
        if not m:
            continue
        if kind == "torus":
            # 't04o3333': 1/4 turn, outer half of tube, minor radius .3333.
            # The header description is authoritative when it has the numbers.
            sweep = 1.0 / int(m.group(1)) if int(m.group(1)) else 1.0
            minor = int(m.group(3)) / 10000.0
            nums = re.findall(r"(\d*\.?\d+)\s*x\s*(\d*\.?\d+)\s*x\s*(\d*\.?\d+)", description)
            major = 1.0
            if nums:
                major, minor, sweep = (float(x) for x in nums[0])
            elif b[0] == "r":            # 'r' files: tube radius 1, major = 1/ratio
                major, minor = 1.0 / minor if minor else 1.0, 1.0
            return Primitive("torus", sweep, major, major, minor, m.group(2))
        sweep = int(m.group(1)) / int(m.group(2))
        if kind == "cone":
            n = int(m.group(3))
            return Primitive("cone", sweep, n + 1.0, float(n))
        if kind == "plane":
            shape = re.sub(r"^\d+-\d+", "", b)
            if m.lastindex and m.lastindex >= 3:
                n = int(m.group(3))
                return Primitive("plane", sweep, float(n), n + 1.0, shape="ring")
            return Primitive("plane", sweep, 0.0, 1.0, shape=shape)
        return Primitive(kind, sweep)
    return None


def feature_of(name: str) -> Optional[str]:
    b = bare(name)
    for rx, feat in _FEATURES:
        if rx.match(b):
            return feat
    return None


@dataclass
class Surface:
    kind: str                 # cylinder | cone | plane | sphere | torus
    instance: int             # id of the primitive instance that defines it
    primitive: Primitive
    origin: np.ndarray        # model coords of the primitive's (0,0,0)
    axis: np.ndarray          # model image of the primitive's +Y (not normalised:
                              #   its length is the height for cylinder/cone)
    u: np.ndarray             # model image of +X  (length = radius scale)
    w: np.ndarray             # model image of +Z
    exact: bool               # still a true cylinder/cone/sphere/torus/circle-disc
    why_not: str              # '' | 'elliptic' | 'sheared' | 'stretched'
    inverted: bool            # faces point towards the axis: a hole, not a boss
    feature: Optional[str]    # 'stud', 'peghole', ... if inside such a composite

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.u)) * self.primitive.r0


def _place(inst: Instance, prim: Primitive, tol: float = 1e-4):
    a = inst.matrix[:3, :3]
    u, axis, w = a[:, 0], a[:, 1], a[:, 2]
    lu, la, lw = (np.linalg.norm(v) for v in (u, axis, w))
    scale = max(lu, la, lw, 1e-12)
    round_ = abs(lu - lw) < tol * scale and abs(u @ w) < tol * lu * lw + 1e-12
    upright = (abs(axis @ u) < tol * la * lu + 1e-12) and (abs(axis @ w) < tol * la * lw + 1e-12)
    why = ""
    if prim.kind == "plane":
        if prim.shape and not round_:
            why = "elliptic"             # still an exact plane; outline is an ellipse
        return u, axis, w, True if not why else True, why
    if not round_:
        why = "elliptic"
    elif not upright and la > 1e-9:
        why = "sheared"
    elif prim.kind in ("sphere", "torus") and abs(la - lu) > tol * scale:
        why = "stretched"
    return u, axis, w, why == "", why


def surface_of(flat: Flat, instance_id: int) -> Optional[Surface]:
    """Analytic surface for geometry owned by `instance_id`, or None (= facet)."""
    chain = flat.chain(instance_id)
    feature = next((f for f in (feature_of(i.name) for i in chain) if f), None)
    for inst in chain:
        if inst.category != "primitive":
            break                        # left primitive land: hand-made facets
        prim = classify(inst.name, inst.description)
        if prim is None:
            continue
        if prim.kind == "circle":
            return None
        u, axis, w, exact, why = _place(inst, prim)
        return Surface(prim.kind, inst.id, prim, inst.matrix[:3, 3].copy(), axis.copy(),
                       u.copy(), w.copy(), exact, why, inst.inverted, feature)
    return None


def surfaces(flat: Flat) -> dict:
    """{instance_id: Surface} for every instance that owns triangles."""
    out = {}
    for iid in np.unique(flat.tri_instance):
        s = surface_of(flat, int(iid))
        if s is not None:
            out[int(iid)] = s
    return out


def hole_boss_mismatches(flat: Flat, surf: Optional[dict] = None) -> int:
    """Exact cylinders whose `inverted` flag disagrees with their triangle normals."""
    surf = surfaces(flat) if surf is None else surf
    bad = 0
    for iid, s in surf.items():
        if s.kind != "cylinder" or not s.exact:
            continue
        t = flat.tris[flat.tri_instance == iid]
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ax = s.axis / (np.linalg.norm(s.axis) or 1)
        rel = t.mean(axis=1) - s.origin
        radial = rel - np.outer(rel @ ax, ax)
        outward = np.einsum("ij,ij->i", nn, radial) > 0
        if outward.mean() > 0.5 and s.inverted or outward.mean() < 0.5 and not s.inverted:
            bad += 1
    return bad
