#!/usr/bin/env python3
"""Turn LDraw parts into closed solids and report the validation numbers.

usage: make_solid.py LIBRARY_ROOT PART [PART ...] [--out DIR] [--stl] [--3mf]
"""
import argparse
import os
import sys

from ldraw2solid import Library, Flattener
from ldraw2solid.mesh import SolidifyError, solidify, validate, write_3mf, write_stl
from ldraw2solid.primitives import bare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--hi", action="store_true", help="use 48-segment primitives")
    ap.add_argument("--out", default="out", help="output folder (default: out)")
    ap.add_argument("--stl", action="store_true", help="write the Tier 1 mesh as STL")
    ap.add_argument("--3mf", dest="tmf", action="store_true", help="write the Tier 1 mesh as 3MF")
    a = ap.parse_args()

    fl = Flattener(Library(a.library, "hi" if a.hi else "std"))
    failed = 0
    for part in a.parts:
        flat = fl.flatten(part)
        print(f"\n=== {part}: {flat.instances[0].description}")
        try:
            solid = solidify(flat)
        except SolidifyError as e:
            print(f"  tier 1 FAILED: {e}")
            failed += 1
            continue
        st, r = solid.stats, validate(solid, flat)
        print(f"  repairs: {st['tjunction_verts']} T-junction vertices, "
              f"{st['loops_planar']} planar + {st['loops_nonplanar']} non-planar loops capped, "
              f"{st['pieces']} closed pieces unioned")
        print(f"  tier 1: {r['tris']} triangles, {r['open_edges']} open edges, "
              f"{r['edges_3plus']} shared by 3+, {r['flipped_edges']} flipped, "
              f"{r['self_intersections']} self-intersections")
        print(f"          volume {r['volume_ldu3'] * 0.4 ** 3:.1f} mm^3, bounds error {r['bounds_err_ldu']:.2g} LDU, "
              f"caps left {r['cap_tris_left']}   -> {'OK' if r['ok'] else 'NOT VALID'}")
        failed += not r["ok"]
        if a.stl or a.tmf:
            os.makedirs(a.out, exist_ok=True)
            v = solid.to_print_frame()
            if a.stl:
                p = os.path.join(a.out, bare(part) + ".stl")
                write_stl(p, v[solid.faces], b"ldraw2solid tier 1 watertight mesh")
                print(f"  wrote {p}")
            if a.tmf:
                p = os.path.join(a.out, bare(part) + ".3mf")
                write_3mf(p, v, solid.faces, bare(part))
                print(f"  wrote {p}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
