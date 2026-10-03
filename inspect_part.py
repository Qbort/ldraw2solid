#!/usr/bin/env python3
"""Flatten LDraw parts and report what the parser recovered.

usage: inspect_part.py LIBRARY_ROOT PART [PART ...] [--hi] [--stl DIR]
"""
import argparse
import os
import struct
from collections import Counter

import numpy as np

from ldraw2solid import Library, Flattener, surfaces
from ldraw2solid.primitives import bare


def tri_normals_area(t):
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1) / 2
    return n, a


def topology(tris, tol=1e-3):
    """Weld vertices on a grid and count how many faces meet at each edge."""
    v = np.round(tris.reshape(-1, 3) / tol).astype(np.int64)
    _, idx = np.unique(v, axis=0, return_inverse=True)
    f = idx.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts == 2).sum()), int((counts > 2).sum())


def write_stl(path, tris):
    n, _ = tri_normals_area(tris)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    with open(path, "wb") as fh:
        fh.write(b"ldraw2solid raw flatten (not yet watertight)".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        fh.write(rec.tobytes())


def report(fl, name, flat, stl_dir=None):
    print(f"\n=== {name}: {flat.instances[0].description}")
    lo, hi = flat.bounds()
    size_mm = (hi - lo) * 0.4
    print(f"  triangles {len(flat.tris)}   hard edges {len(flat.lines)}   "
          f"smooth edges {len(flat.clines)}   subfile instances {len(flat.instances)}")
    print(f"  bounds LDU x[{lo[0]:g},{hi[0]:g}] y[{lo[1]:g},{hi[1]:g}] z[{lo[2]:g},{hi[2]:g}]"
          f"   = {size_mm[0]:.1f} x {size_mm[2]:.1f} x {size_mm[1]:.1f} mm (w x d x h)")
    print(f"  BFC-certified triangles: {flat.tri_certified.mean() * 100:.0f}%"
          + (f"   MISSING FILES: {sorted(flat.missing)}" if flat.missing else ""))

    n, area = tri_normals_area(flat.tris)
    vol = np.einsum("ij,ij->i", flat.tris[:, 0], np.cross(flat.tris[:, 1], flat.tris[:, 2])).sum() / 6
    print(f"  signed volume (rough, mesh is open): {vol * 0.4 ** 3:.0f} mm^3")
    b, m, nm = topology(flat.tris)
    print(f"  edges after welding: {m} shared by 2 faces, {b} open, {nm} shared by 3+"
          f"   -> {'watertight' if b == 0 and nm == 0 else 'NOT watertight yet'}")

    surf = surfaces(flat)
    kind_area, kinds = Counter(), Counter()
    for iid, s in surf.items():
        k = s.kind if s.exact else f"{s.kind} ({s.why_not})"
        kinds[k] += 1
        kind_area[k] += area[flat.tri_instance == iid].sum()
    covered = np.isin(flat.tri_instance, list(surf))
    kind_area["hand-made facets"] = area[~covered].sum()
    tot = area.sum()
    print("  surface area by origin:")
    for k, a in kind_area.most_common():
        cnt = f"{kinds[k]:4d} instances" if k in kinds else " " * 14
        print(f"    {k:22s} {cnt}  {a / tot * 100:5.1f}% of area")

    print("  analytic surfaces found (grouped):")
    groups = Counter()
    for s in surf.values():
        if s.kind == "plane":
            continue
        if s.kind in ("cylinder", "cone"):
            d = f"r={s.radius * 0.4:.2f}mm h={np.linalg.norm(s.axis) * 0.4:.2f}mm"
        elif s.kind == "torus":
            d = f"R={s.radius * 0.4:.2f}mm tube={s.primitive.minor * np.linalg.norm(s.u) * 0.4:.2f}mm"
        else:
            d = f"r={s.radius * 0.4:.2f}mm"
        tag = ("hole" if s.inverted else "boss") + (f", in {s.feature}" if s.feature else "")
        ex = "" if s.exact else f" [{s.why_not}]"
        groups[f"{s.kind:8s} {d}  sweep {s.primitive.sweep:.3g}  ({tag}){ex}"] += 1
    for g, c in sorted(groups.items(), key=lambda x: -x[1])[:14]:
        print(f"    {c:3d} x {g}")

    # Does 'hole vs boss' from INVERTNEXT bookkeeping agree with real geometry?
    bad = 0
    for iid, s in surf.items():
        if s.kind != "cylinder" or not s.exact:
            continue
        t = flat.tris[flat.tri_instance == iid]
        nn, _ = tri_normals_area(t)
        ax = s.axis / (np.linalg.norm(s.axis) or 1)
        rel = t.mean(axis=1) - s.origin
        radial = rel - np.outer(rel @ ax, ax)
        outward = np.einsum("ij,ij->i", nn, radial) > 0
        if outward.mean() > 0.5 and s.inverted or outward.mean() < 0.5 and not s.inverted:
            bad += 1
    print(f"  hole/boss flag vs. actual triangle normals: {bad} mismatches")

    if stl_dir:
        os.makedirs(stl_dir, exist_ok=True)
        p = os.path.join(stl_dir, bare(name) + "_raw.stl")
        write_stl(p, flat.to_print_frame())
        print(f"  wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--hi", action="store_true", help="use 48-segment primitives")
    ap.add_argument("--stl", help="folder for raw STL dumps (mm, Z up)")
    a = ap.parse_args()
    fl = Flattener(Library(a.library, "hi" if a.hi else "std"))
    for part in a.parts:
        report(fl, part, fl.flatten(part), a.stl)
