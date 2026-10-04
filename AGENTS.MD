# AGENTS.md

Guidance for coding agents working in this repository. Read `README.MD` for what the project does and `docs/TECHNICAL.md` for how it works, what has been measured and what is open.

## Goal

Convert LDraw parts into solids for 3D printing. The target is a STEP file with analytic surfaces where the part allows it, with a single watertight mesh as the fallback. A result that only looks right is not acceptable: the output must be a valid closed solid that matches the LDraw geometry.

## Layout

```
ldraw2solid/
  parser.py       Library lookup, per-file parsing, recursive flattening with provenance
  primitives.py   Primitive name -> analytic surface, placed in model coordinates; feature tags
  mesh.py         Tier 1: weld, T-junctions, edge pairing, caps, manifold3d union, validation, STL/3MF
  brep.py         Tiers 2 and 3: OpenCASCADE faces and solid, checks, STEP read/write
inspect_part.py   CLI report for individual parts, optional raw STL dump
make_solid.py     CLI: Tier 1 mesh and Tier 2/3 STEP, with validation report
sweep.py          Runs the parser over every part in the library and prints statistics
tests/            pytest: parser reference values and Tiers 1-3 on the reference parts
docs/TECHNICAL.md How it works, measurements, known gaps, roadmap
lib/              LDraw parts library (not in git, see below)
out/              Generated files (not in git)
```

## Setup and commands

- Python 3.10+. The parser needs numpy only. `mesh.solidify` needs `manifold3d`; `brep.py` needs `cadquery-ocp` (OCP 8). Install into `.venv` with `.venv/bin/pip install -r requirements.txt`; system pip is externally managed.
- The LDraw library must be at `./lib`, unpacked from `complete.zip` so that `lib/parts/` and `lib/p/` exist. It is about 500 MB. Never commit it or `out/`.
- Tests: `.venv/bin/python -m pytest` (about 30 s; they skip if the library is missing; `LDRAW_LIB` points elsewhere).
- Make and check solids: `.venv/bin/python make_solid.py lib 3001.dat 3062b.dat 3700.dat 6377.dat --stl --3mf --step` (exit code 1 if any check fails).
- Parser report: `python3 inspect_part.py lib 3001.dat 3062b.dat 3700.dat` (runs on the system Python, no manifold3d needed).
- Library sweep: `python3 sweep.py`. It takes about 9 minutes on the official `complete.zip`, so run it in the background. It must finish with an empty error list.

## Reference values

Parser (any parser change must keep these; pinned in `tests/test_reference.py`):

| Part | Bounds (LDU) | Triangles | Open edges after weld | Hole/boss mismatches |
|---|---|---|---|---|
| 3001.dat | x ±40, y −4..24, z ±20 | 700 | 224 | 0 |
| 3062b.dat | x ±10, y −4..24, z ±10 | 384 | 0 | 0 |
| 3700.dat | x ±20, y −4..24, z ±10 | 550 | 126 | 0 |

Stud cylinders must come out as radius 2.4 mm, height 1.6 mm.

Solids (pinned in `tests/test_mesh.py` and `tests/test_brep.py`):

| Part | Tier 1 triangles | Tier 1 volume (LDU³) | Tier 2 faces | Tier 3 faces | Tier 3 cylinders faceted |
|---|---|---|---|---|---|
| 3001.dat | 964 | 39390.2532 | 249 planes | 25 planes, 14 cylinders | 0 |
| 3062b.dat | 384 | 4641.2796 | 85 planes | 5 planes, 5 cylinders | 0 |
| 3700.dat | 666 | 12101.5283 | 159 planes | 47 planes, 7 cylinders | 2 (pin joint) |
| 6377.dat | 7320 | 315239.0716 | — | 16 cylinder faces, 12 cylinders lifted | 2 (female connectors) |

6377 (Duplo track 4×8) is the hard case: it needed most of the Tier 1 repairs. Use it as a regression part for anything that touches `mesh.py` or `brep.py`.

## Checks every output must pass

- **Tier 1 mesh:**
  - 0 open edges, 0 edges with 3+ faces, 0 flipped edges, 0 self-intersections, no leftover caps;
  - bounds unchanged;
  - `off_surface == 0`: every triangle lies within 0.001 LDU of an LDraw triangle;
  - `export_check` clean: the float32 STL, merged by position, as a slicer reads it.
- **STEP, read back from the file:**
  - 1 valid solid, 0 free edges;
  - `tessellation_check` reports 0 open edges: the file meshed and merged like a slicer does.

Edges with 4 faces from surfaces touching along a line are reported, not failed.
- **Pinch vertices** (the surface touching itself at a point) are reported by `validate`, not failed. Slicers accept them.

All of this is in `validate()` and `make_solid.py`. A check passing by vertex index is not enough: twice this session a mesh that was valid by index had defects the user's slicer showed.

## Invariants

Do not break these. Later stages depend on them.

- **Units and axes.** Everything inside the package is in LDU with LDraw axes (−Y up). Conversion to millimetres and Z up happens only at export, through `to_print_frame()` or `LDU_TO_MM` and `LDRAW_TO_ZUP`. The axis change is a proper rotation; do not replace it with a mirror.
- **Winding.** Every triangle in a `Flat` is counter-clockwise seen from outside. A reference reverses winding when its matrix has a negative determinant XOR it is preceded by `INVERTNEXT`.
- **`Instance.inverted`** is the parity of `INVERTNEXT` only, not of mirroring. It means the surface faces its own axis (a hole). Keep `hole_boss_mismatches` at 0.
- **Provenance.**
  - Every triangle keeps the instance it came from. `Solid.tri_source` indexes `Flat.tris` (−1 for a cap), and `Solid.tri_instance` holds the instance id (for a cap, the instance whose loop it closes).
  - manifold3d face IDs carry this through the union. Any stage that creates, splits or merges faces must keep it.
- **Caching.** `Flattener` caches flattened files in their own coordinates. Never modify a returned `Flat` in place, and call `Flattener.forget()` between parts in batch runs.

## Tooling lessons (measured this session)

**manifold3d 3.x**
- `Mesh64` needs writeable, C-contiguous arrays of the exact dtype (float64 vertices, uint64 indices). Read-only views are rejected. Use `np.array(..., copy)`.
- It assumes a piece never overlaps itself. With a coplanar self-overlap or zero-area triangles in its input, it silently invents faces that lie on no input surface (the 6377 fin wedges). The mesh stays closed, so only `off_surface` catches it.
- `mesh._cancel_self_contact` and `_clean_triangulate` exist to keep its input clean. Keep them.
- `set_tolerance(0.001 LDU)` on each piece is needed: library coordinates have 4 decimals, and without it caps leave slivers.

**OpenCASCADE (OCP 8, cadquery-ocp)**
- Static methods lost the `_s` suffix in places: use `TopoDS.Face` and `TopoDS.Shell`. `BRepGProp.VolumeProperties_s` and `BRepBndLib.AddOptimal_s` keep it.
- Shape lists are `OCP.OCP.collections.List_TopoDS_Shape`. There is no `TopTools_IndexedMapOfShape`.
- `BRepCheck_Result.Status()` cannot be converted to Python. Call `BRepCheck_Face(f).IntersectWires()`, `BRepCheck_Wire(w).Closed()` and similar directly.
- `BRepBuilderAPI_MakeWire` merges vertices by position. A loop that passes one point twice (a tangent hole) must be built with `BRep_Builder` and explicit vertices.
- A wire may pass the same vertex twice. It may not touch the middle of another edge: insert such points (`brep.touch_points`).
- OCC's STEP import splits pinch vertices and leaves wires open. `separate_pinches` moves the copies 1e-3 LDU apart before building STEP. At 1e-5 LDU the mesher merged some copies and not others, leaving slits.
- Sewing tolerance is 1e-6 mm. Any rounding that places edges (for example cylinder angle ranges) must stay far below that: angles are rounded to 1e-9 rad.
- OCC booleans (fusing circle-segment slivers onto a Tier 2 solid) were tried for Tier 3. They depended on operation order and fuzzy value, and one test run on 6377 took more than 30 minutes. Avoid them for whole-part work.

**LDraw data**
- Coordinates have 4 decimals. A primitive placed at 45° gets a radius like 12.00003 instead of 12. Vertices of neighbouring pieces can be 0.0005 LDU apart; `weld` uses eight half-cell-shifted grids for that.
- Overlapping coplanar faces, faces of one piece resting on another part of the same piece, and open primitives are common. Examples: `tri4` is a prism with no base; `box2-7`, `box3-7a` and `box4-1` are boxes missing faces, sometimes sheared; `box3-7a` is used inverted as a cavity lining.
- `cylj4x8` (a cylinder joint with a saddle-shaped end) is not classified as a cylinder.
- The official `complete.zip` differs from the `pybricks/ldraw` mirror the older statistics came from. Its sweep found 4 parts with missing subfiles; which ones has not been checked.

## LDraw facts

- Line types: 0 meta, 1 subfile reference, 2 line (hard edge), 3 triangle, 4 quad, 5 conditional line.
- Colour 16 inherits from the referencing line; colour 24 is the edge colour.
- File names in references use backslashes and are case-insensitive (`s\3001s01.dat`).
- Primitive names encode the sweep: `1-4cyli` is a quarter cylinder, `4-4` a full turn. Unit primitives have radius 1 and height 1 along +Y.
- `conN` is a cone from radius N+1 at y=0 to N at y=1. `ringN` is a flat annulus from N to N+1.
- Stud dimensions are nominal: radius 6 LDU, height 4 LDU (2.4 mm × 1.6 mm). There is no clutch-fit clearance in the library.
- Flat faces next to a curved primitive are authored to meet the 16-gon, not the true circle.

## Working rules

- Lift per face and fall back per face. The solid must be valid either way.
- Validate every output with the checks above, and read STEP files back before judging them.
- Prefer measuring over reasoning: write a short diagnostic script (the scratchpad is fine) before changing code. Most fixes this session came from locating the exact triangles involved.
- Report numbers you measured, and say when something was not checked.
- Keep the parser dependency-light (numpy only). Heavier dependencies go in `mesh.py`, `brep.py` or new modules.
- Do not download the parts library inside tests or scripts. Take its path as an argument.

## Working with the user

- The user checks STL and STEP output in a 3D-printing slicer. That is the acceptance test. When a fix is meant to change what the slicer shows, regenerate `out/`, let the user test, and wait for their go-ahead before committing.
- Commit each finished stage or fix separately before starting the next one. The user asks for this and prefers small, separate commits: a licence change, a docs rewrite and a code fix are three commits.
- The code is GPL-3.0-or-later. Every Python file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`; add it to new files.

## Open issues

- 6377 Tier 3: the volume is 0.065 mm³ above the prediction, all from the two 270° connector pegs. The faces match theory, and the cause has not been found, so `make_solid.py` reports NOT VALID for it.
- 6377: the female connectors' outer half-cylinders stay faceted. They touch a cavity wall along a line, which meshes as 2 edges with 4 faces.
- `sweep.py` has not been rerun since `weld` changed, and Tiers 1-3 have not been run over the library.
- Tori flagged `stretched` outnumber exact ones; `_place` may be too strict.
- Quads are split along the 0–2 diagonal with no check for concave, bow-tie or non-planar cases.
- `resolution="lo"` is accepted by `Library` but not implemented.

## Next task: make parts fit each other

LDraw dimensions are nominal, so printed studs and tubes fit too tightly. The roadmap item is to offset the mating surfaces by a printer-specific clearance. Starting points:

- **Feature tags.** `primitives.feature_of` tags surfaces by the subfile they sit in: `stud`, `tube` (`stud4`, `stud3`, `stud16`, `stud18`), `peghole`, `axlehole` and `connhole`. `Surface.feature` and `Surface.inverted` (hole or boss) say what kind of surface a cylinder is. In 3001: 8 stud bosses r = 2.4 mm, 3 tube holes r = 2.4 mm, and 3 tube bosses r = 3.2 mm. The tags match by name prefix, so check them on other parts: 6377 uses `stud8a`, which is tagged `stud`.
- **Not only cylinders.** A stud is held by the tubes and by the inner walls of the brick below. Those walls are plain hand-made faces with no tag, so fit may need flat offsets too. Measure which surfaces actually touch before designing.
- **Where to apply an offset.**
  - *Tier 1 mesh:* moving a cylinder region's vertices radially keeps flat faces across the axis in their plane, but bends faces along the axis that share those vertices.
  - *Tier 3:* the radius `_Cyl.r` drives the patches, the arcs on neighbouring flat faces, snapping and the volume prediction together. Snapping and `_plan_lift` currently require boundary vertices within 0.001 LDU of `c.r`, so a changed radius has to move the vertices too, or keep separate "polygon radius" and "output radius" values.
- **Checks.** Keep every output check above passing. A clearance changes the volume, so update the Tier 3 volume prediction, and mesh-volume comparisons, by the offset.
