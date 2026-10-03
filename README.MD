# ldraw2solid

Turn LDraw parts into solid geometry for 3D printing: a watertight mesh at minimum, a STEP file with true cylinders and planes where the part allows it.

LDraw parts are triangle and quad soups built for rendering, so a plain export looks right but is not a solid. They do carry more CAD information than a mesh, though: curved surfaces are almost always references to named primitives (`4-4cyli.dat` is a cylinder with a known axis and radius). This project keeps that information through flattening and uses it to rebuild analytic surfaces.

## Status

| Stage | State |
|---|---|
| Parser with provenance tagging | Working, run on all 18,615 library parts without errors |
| Primitive to analytic surface mapping | Working for cylinder, cone, plane, sphere, torus |
| Tier 1: watertight mesh (STL/3MF) | Working on the three reference parts |
| Tier 2: faceted STEP | Working on the three reference parts |
| Tier 3: analytic STEP | Working on the three reference parts, cylinders only |
| Print tolerance offsets | Not started |

## Setup

Requires Python 3.10+. The parser needs only numpy. Tier 1 needs `manifold3d`, Tiers 2 and 3 need OpenCASCADE through `cadquery-ocp`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

You also need the LDraw parts library, which is not included in this repository. Download `complete.zip` from https://library.ldraw.org/library/updates/complete.zip and unpack it. The result is a folder containing:

- `parts/` for parts, with subparts in `parts/s/`
- `p/` for the standard 16-segment primitives, `p/48/` for high resolution, `p/8/` for low resolution

The commands below assume the library is at `./lib`. That folder is about 500 MB and should stay out of git.

## Usage

Inspect one or more parts and dump raw STL files:

```
python3 inspect_part.py lib 3001.dat 3062b.dat 3700.dat --stl out
```

Add `--hi` to substitute the 48-segment primitives. The STL files are in millimetres with Z up. They are the raw flattened mesh and are not watertight, so they are for looking at and not for printing.

Make solids, check them, and write STL, 3MF and STEP files to `out/`:

```
.venv/bin/python make_solid.py lib 3001.dat 3062b.dat 3700.dat --stl --3mf --step
```

STEP is Tier 3 by default; `--tier 2` writes planar facets only. Every STEP file is read back and checked before the command reports OK. All output is in millimetres with Z up.

Run the tests (they skip if the library is not at `./lib` or `$LDRAW_LIB`):

```
.venv/bin/python -m pytest
```

Run the whole library and print summary statistics (about 5 minutes):

```
python3 sweep.py
```

From Python:

```python
from ldraw2solid import load, surfaces

flat = load("3001.dat", "lib")          # resolution="hi" for 48-segment primitives
flat.tris                               # (T, 3, 3) triangles in LDU, CCW from outside
flat.tri_instance                       # (T,) which subfile instance each triangle came from
flat.chain(flat.tri_instance[0])        # that instance and its parents, up to the root
surfaces(flat)                          # {instance_id: Surface} for analytic primitives
flat.to_print_frame()                   # triangles in mm, Z up

from ldraw2solid.mesh import solidify, validate
from ldraw2solid import brep

solid = solidify(flat)                  # Tier 1: closed mesh, solid.tri_source -> flat.tris
validate(solid, flat)                   # edge, orientation, volume, bounds and cap checks
b = brep.analytic_brep(solid, flat)     # Tier 3 (brep.faceted_brep(solid) for Tier 2)
brep.write_step("out/3001.step", b.shape)
```

## How it works

### Parser (`ldraw2solid/parser.py`)

- `Library` indexes the library folder and resolves file names case-insensitively, with optional high-resolution primitive and stud-logo substitution.
- `parse_text` reads one file into raw commands, tracking BFC state (`CERTIFY`, `CW`/`CCW`, `CLIP`, `INVERTNEXT`). MPD files with embedded `0 FILE` sections are supported.
- `Flattener` resolves subfile references recursively and caches each file once. It returns a `Flat`:
  - triangles, hard edges (line type 2), and smooth edges (line type 5)
  - per triangle: colour, owning instance, whether its winding is BFC-certified, and a quad id shared by the two halves of a quad
  - a list of `Instance` records, one per subfile placement, each with its name, parent, cumulative 4×4 transform, and an `inverted` flag

Every triangle is emitted counter-clockwise seen from outside. `inverted` is the parity of `INVERTNEXT` above an instance, which tells a hole from a boss.

### Primitives (`ldraw2solid/primitives.py`)

- `classify(name)` recognises a primitive by file name and returns its surface in unit coordinates (kind, sweep fraction, radii).
- `surface_of(flat, instance_id)` walks up a triangle's reference chain to the nearest recognised primitive and places the surface in model coordinates.
- A surface is `exact` if the transform keeps it a true cylinder, cone, sphere, or torus. Otherwise `why_not` says `elliptic`, `sheared`, or `stretched`.
- Surfaces inside studs, underside tubes, and peg holes carry a `feature` tag, intended for print-tolerance offsets.

### Tier 1: watertight mesh (`ldraw2solid/mesh.py`)

Pieces of an LDraw part touch without sharing edges: a stud stands on the top face, a tube ends against the underside of the top plate. `solidify`:

1. welds vertices at 0.001 LDU;
2. splits T-junctions on open edges;
3. caps every open loop: loops on one plane are triangulated together, so nested loops give a ring; a non-planar loop is fanned to its centroid;
4. unions the resulting closed pieces with manifold3d, with each triangle's source index as its face ID.

A cap lies on the surface it rests against, so the union removes it and cuts the matching hole into that surface. A cap that survives the union fails validation. The union runs with a precision of 0.001 LDU because library coordinates have 4 decimals: without it, 3700's pin cap leaves slivers about 1e-4 LDU wide.

### Tiers 2 and 3: STEP (`ldraw2solid/brep.py`)

Tier 2 merges connected coplanar triangles into planar faces. Their boundary loops come from the closed mesh, so neighbouring faces share their vertices exactly. The faces are sewn into one solid with OpenCASCADE.

Tier 3 replaces exact cylinder primitives with cylindrical faces. A cylinder is lifted only if it is a simple band whose neighbours are planes across its axis (the shared edge becomes a circle arc) or along it (a straight edge). Otherwise it stays faceted. Before any face is built, the boundary vertices of lifted cylinders are moved onto the true circle, by at most 0.001 LDU. The expected volume is the mesh volume plus the circle segments that the lifted cylinders add (bosses) or remove (holes), and the check compares against it.

### Coordinates and units

LDraw uses LDU with −Y up. 1 LDU = 0.4 mm. Everything inside the package stays in LDU and LDraw axes. `Flat.to_print_frame()` converts to millimetres with Z up, using a proper rotation so winding is preserved.

## Findings so far

Test parts:

| Part | Size (mm) | Triangles | Analytic cylinders | Watertight after welding |
|---|---|---|---|---|
| 3001 (2×4 brick) | 32 × 16 × 11.2 | 700 | 14 | No, 224 open edges |
| 3062b (round 1×1) | 8 × 8 × 11.2 | 384 | 6 | Yes |
| 3700 (Technic 1×2) | 16 × 8 × 11.2 | 550 | 9 | No, 126 open edges |

Library sweep (18,615 parts, 16-segment primitives, vertices welded at 0.001 LDU):

- 2,157 parts (12%) are already watertight after welding alone.
- 18,289 parts (98%) are fully BFC-certified.
- Recognised primitives account for a median of 16% of a part's surface area (mean 27%), mostly cylinders. The rest is hand-made facets. How much of that is flat and how much is hand-drawn curves has not been measured.
- Elliptic and sheared cylinders together are under 1% of area.

These numbers came from a GitHub mirror of the library (`pybricks/ldraw`), not from the official `complete.zip`, and may differ slightly on the current release.

Solids (measured; volumes in mm³, Tier 3 includes the circle segments):

| Part | Tier 1 triangles | Tier 1 volume | Tier 2 faces | Tier 3 faces | Cylinders lifted | Tier 3 volume |
|---|---|---|---|---|---|---|
| 3001 | 964 | 2521.0 | 249 | 39 | 14 of 14 | 2535.5 |
| 3062b | 384 | 297.0 | 85 | 10 | 5 of 5 | 304.8 |
| 3700 | 682 | 774.5 | 159 | 54 | 7 of 9 | 771.1 |

Cylinders are counted per face: in 3062b two stacked primitives on the same cylinder merge into one face, so its 6 primitives give 5. In 3700 the underside pin and the half cylinder it cuts into stay faceted: they meet along a saddle curve, which Tier 3 does not build yet.

## Known gaps

- More tori are flagged `stretched` than exact. Whether that is real or a mistake in the torus check has not been verified.
- Quads are always split along the 0–2 diagonal. Concave, bow-tie, and non-planar quads are not detected.
- `resolution="lo"` is accepted but does nothing yet.
- Colours stay as LDraw colour codes. `LDConfig.ldr` is not read.
- `!TEXMAP` is ignored, which gives the untextured fallback geometry.
- Tiers 1–3 have only been run on the three reference parts, not on the library.
- Tier 3 lifts cylinders only. Cones, spheres and tori stay faceted, as do cylinders that meet another cylinder or an oblique plane.
- On 3001 the Tier 3 volume differs from the expected value by 2e-4 mm³ (3062b: 8e-9, 3700: 5e-5). The cause is not confirmed. A likely candidate is vertices the union places on polygon edges with its 0.001 LDU precision.

## Roadmap

1. **Tier 1, watertight mesh.** Done for the reference parts. Next: run it over the library and add failure statistics to `sweep.py`.
2. **Tier 2, faceted STEP.** Done for the reference parts.
3. **Tier 3, analytic STEP.** Cylinders done for the reference parts. Next: cones, cylinder–cylinder intersections such as the 3700 pin, and hand-drawn curvature.
4. **Print tolerances.** Offset stud and tube radii by a printer-specific amount using the feature tags.

## Licence

The code licence has not been chosen yet.

The LDraw parts library is distributed separately under its own terms (CC BY 4.0 for current files; see `CAreadme.txt` in the library). Geometry produced by this tool is derived from those part files.

LDraw is a trademark of the Estate of James Jessiman. LEGO is a trademark of the LEGO Group, which does not sponsor or endorse this project.