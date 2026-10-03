# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` first for what the project does and where it stands.

## Goal

Convert LDraw parts into solids for 3D printing. The target is a STEP file with analytic surfaces where the part allows it, with a single watertight mesh as the fallback. A result that only looks right is not acceptable: the output must be a valid closed solid.

## Layout

```
ldraw2solid/
  parser.py       Library lookup, per-file parsing, recursive flattening with provenance
  primitives.py   Primitive name -> analytic surface, placed in model coordinates
inspect_part.py   CLI report for individual parts, optional raw STL dump
sweep.py          Runs every part in the library and prints summary statistics
lib/              LDraw parts library (not in git, see below)
out/              Generated files (not in git)
```

## Setup and commands

- Python 3.10+, numpy. Nothing else is required for the current code.
- The LDraw library must be at `./lib` (unpacked `complete.zip` from library.ldraw.org, containing `parts/` and `p/`). It is about 500 MB. Never commit it, and never commit `out/`.
- Check three reference parts: `python3 inspect_part.py lib 3001.dat 3062b.dat 3700.dat`
- Check the whole library: `python3 sweep.py` (about 5 minutes, must finish with `errors {}`)

There is no test suite yet. Until there is one, the two commands above are the regression check. Adding pytest tests that pin the reference values below is welcome.

## Reference values

Any change to the parser must keep these:

| Part | Bounds (LDU) | Triangles | Open edges after weld | Hole/boss mismatches |
|---|---|---|---|---|
| 3001.dat | x ±40, y −4..24, z ±20 | 700 | 224 | 0 |
| 3062b.dat | x ±10, y −4..24, z ±10 | 384 | 0 | 0 |
| 3700.dat | x ±20, y −4..24, z ±10 | 550 | 126 | 0 |

Stud cylinders must come out as radius 2.4 mm, height 1.6 mm. The library sweep must flatten all parts with no exceptions and no missing subfiles.

Tier 1 work is expected to bring open edges to 0. It must not change bounds.

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

Tier 1 on 3001 and 3700: weld vertices, split T-junctions, resolve the stud-on-surface overlaps, remove internal faces, and reach 0 open edges with provenance intact. See the roadmap in `README.md` for what follows.