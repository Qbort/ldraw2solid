#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Flatten LDraw parts and report what the parser recovered.

usage: inspect_part.py LIBRARY_ROOT PART [PART ...] [--hi] [--stl DIR]
"""
import argparse
import os
from collections import Counter

import numpy as np

from ldraw2solid import Library, Flattener, surfaces
from ldraw2solid.mesh import topology, tri_normals_area, write_stl
from ldraw2solid.primitives import bare, hole_boss_mismatches


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
    print(f"  hole/boss flag vs. actual triangle normals: {hole_boss_mismatches(flat, surf)} mismatches")

    if stl_dir:
        os.makedirs(stl_dir, exist_ok=True)
        p = os.path.join(stl_dir, bare(name) + "_raw.stl")
        write_stl(p, flat.to_print_frame(), b"ldraw2solid raw flatten (not yet watertight)")
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
