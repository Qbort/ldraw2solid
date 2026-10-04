# AGENTS.md

Guidance for coding agents working in this repository. Read `README.MD` for what the project does and `docs/TECHNICAL.md` for how it works and where it stands.

## Goal

Convert LDraw parts into solids for 3D printing. The target is a STEP file with analytic surfaces where the part allows it, with a single watertight mesh as the fallback. A result that only looks right is not acceptable: the output must be a valid closed solid.

## Layout

```
ldraw2solid/
  parser.py       Library lookup, per-file parsing, recursive flattening with provenance
  primitives.py   Primitive name -> analytic surface, placed in model coordinates
  mesh.py         Tier 1: weld, T-junctions, caps, manifold3d union, validation, STL/3MF
  brep.py         Tiers 2 and 3: OpenCASCADE faces and solid, checks, STEP read/write
inspect_part.py   CLI report for individual parts, optional raw STL dump
make_solid.py     CLI: Tier 1 mesh and Tier 2/3 STEP, with validation report
sweep.py          Runs every part in the library and prints summary statistics
tests/            pytest: parser reference values and Tiers 1–3 on the reference parts
lib/              LDraw parts library (not in git, see below)
out/              Generated files (not in git)
```

## Setup and commands

- Python 3.10+. The parser needs numpy only. `mesh.solidify` needs `manifold3d`, and `brep.py` needs `cadquery-ocp`. Install everything into `.venv` with `.venv/bin/pip install -r requirements.txt` (system pip is externally managed).
- The LDraw library must be at `./lib` (unpacked `complete.zip` from library.ldraw.org, containing `parts/` and `p/`). It is about 500 MB. Never commit it, and never commit `out/`.
- Run the tests: `.venv/bin/python -m pytest` (they skip if the library is missing; set `LDRAW_LIB` to point elsewhere)
- Make and check solids: `.venv/bin/python make_solid.py lib 3001.dat 3062b.dat 3700.dat --step` (exit code 1 if any check fails)
- Check three reference parts: `python3 inspect_part.py lib 3001.dat 3062b.dat 3700.dat`
- Check the whole library: `python3 sweep.py` (about 5 minutes, must finish with `errors {}`)

The tests and the sweep are the regression check.

## Reference values

Any change to the parser must keep these:

| Part | Bounds (LDU) | Triangles | Open edges after weld | Hole/boss mismatches |
|---|---|---|---|---|
| 3001.dat | x ±40, y −4..24, z ±20 | 700 | 224 | 0 |
| 3062b.dat | x ±10, y −4..24, z ±10 | 384 | 0 | 0 |
| 3700.dat | x ±20, y −4..24, z ±10 | 550 | 126 | 0 |

Stud cylinders must come out as radius 2.4 mm, height 1.6 mm. The library sweep must flatten all parts with no exceptions and no missing subfiles.

Solids (pinned in `tests/test_mesh.py` and `tests/test_brep.py`):

| Part | Tier 1 triangles | Tier 1 volume (LDU³) | Tier 2 faces | Tier 3 faces | Tier 3 cylinders faceted |
|---|---|---|---|---|---|
| 3001.dat | 964 | 39390.2532 | 249 planes | 25 planes, 14 cylinders | 0 |
| 3062b.dat | 384 | 4641.2796 | 85 planes | 5 planes, 5 cylinders | 0 |
| 3700.dat | 682 | 12101.5283 | 159 planes | 47 planes, 7 cylinders | 2 (pin joint) |

Every tier must keep:
- bounds unchanged;
- 1 valid solid with 0 free edges, after reading the STEP back;
- a Tier 1 mesh with 0 open edges, 0 flipped edges, 0 self-intersections and no leftover caps.

## Invariants

Do not break these. Later stages depend on them.

- **Units and axes.** Everything inside the package is in LDU with LDraw axes (−Y up). Conversion to millimetres and Z up happens only at export, through `Flat.to_print_frame()` or the constants `LDU_TO_MM` and `LDRAW_TO_ZUP`. The axis change is a proper rotation; do not replace it with a mirror.
- **Winding.** Every triangle in a `Flat` is counter-clockwise seen from outside. A reference reverses winding when its matrix has a negative determinant XOR it is preceded by `INVERTNEXT`.
- **`Instance.inverted`** is the parity of `INVERTNEXT` only, not of mirroring. It means the surface faces its own axis (a hole). `inspect_part.py` checks this against real triangle normals; keep that check at 0 mismatches.
- **Provenance.** Every triangle and edge keeps the id of the instance it came from, and `Instance.matrix` is the cumulative file-local to root transform. Any stage that creates, splits, or merges faces must carry the instance id through. Dropping it makes tier 3 impossible.
- **`tri_quad`.** The two triangles of a quad share an id. Keep it when filtering or reordering triangles.
- **Certification.** `tri_certified` is false when the winding cannot be trusted. Do not assume orientation for those faces.
- **Caching.** `Flattener` caches each file's flattened result in that file's own coordinates and composes it into the parent. Cached arrays are shared, so never modify a returned `Flat` in place. Batch runs must call `Flattener.forget()` between parts or memory grows until the process is killed.

## LDraw facts worth knowing

- Line types: 0 meta, 1 subfile reference, 2 line (hard edge), 3 triangle, 4 quad, 5 conditional line (smooth edge with two control points).
- Colour 16 means "inherit from the referencing line"; colour 24 is the edge colour.
- File names in references use backslashes and are case-insensitive (`s\3001s01.dat`).
- Primitive names encode the sweep as a fraction: `1-4cyli` is a quarter cylinder, `4-4` is a full turn. Unit primitives have radius 1 and height 1 along +Y and are sized by the reference matrix.
- `conN` is a cone from radius N+1 at y=0 to N at y=1. `ringN` is a flat annulus from N to N+1.
- Stud dimensions are nominal: radius 6 LDU, height 4 LDU. There is no clutch-fit clearance in the library.
- Flat faces next to a curved primitive are authored to meet the 16-gon, not the true circle.

## Working rules

- Lift per face and fall back per face. A face becomes analytic when its primitive placement is `exact`; otherwise it stays as planar facets. The solid must be valid either way.
- Validate every output: each edge shared by exactly two faces, no self-intersections, positive volume, and bounds equal to the input bounds.
- Prefer measuring on the library over reasoning about it. `sweep.py` is the place to add a statistic before designing around an assumption.
- Report numbers you measured, and say so when something was not checked.
- Keep the parser dependency-light (numpy only). Heavier dependencies (`trimesh`, `manifold3d`, `libigl`, OpenCASCADE through `build123d` or `pythonocc`) belong in the later stages, in their own modules.
- Do not download the parts library inside tests or scripts. Take its path as an argument.

## Open questions

- Tori flagged `stretched` outnumber exact ones in the sweep. Check whether `_place` in `primitives.py` is too strict for tori or the library really scales them unevenly.
- How much of the "hand-made facets" area is flat and how much is hand-drawn curvature. This decides how much of the library tier 3 can cover.
- Quads are split along the 0–2 diagonal with no check for concave, bow-tie, or non-planar cases.
- `resolution="lo"` is accepted by `Library` but not implemented.

## Next task

Run Tiers 1–3 over the library and add the results to `sweep.py`: how many parts solidify, why the others fail, and how many cylinders Tier 3 lifts. Design the next Tier 3 surfaces (cones, cylinder–cylinder joints like the 3700 pin) around what that measures. See the roadmap in `docs/TECHNICAL.md`.