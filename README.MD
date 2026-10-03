# ldraw2solid

Turn LDraw parts into solid geometry for 3D printing: a watertight mesh at minimum, a STEP file with true cylinders and planes where the part allows it.

LDraw parts are triangle and quad soups built for rendering, so a plain export looks right but is not a solid. They do carry more CAD information than a mesh, though: curved surfaces are almost always references to named primitives (`4-4cyli.dat` is a cylinder with a known axis and radius). This project keeps that information through flattening and uses it to rebuild analytic surfaces.

## Status

| Stage | State |
|---|---|
| Parser with provenance tagging | Working, run on all 18,615 library parts without errors |
| Primitive to analytic surface mapping | Working for cylinder, cone, plane, sphere, torus |
| Tier 1: watertight mesh (STL/3MF) | Not started |
| Tier 2: faceted STEP | Not started |
| Tier 3: analytic STEP | Not started |
| Print tolerance offsets | Not started |

## Setup

Requires Python 3.10+ and numpy.

```
pip install numpy
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

## Known gaps

- More tori are flagged `stretched` than exact. Whether that is real or a mistake in the torus check has not been verified.
- Quads are always split along the 0–2 diagonal. Concave, bow-tie, and non-planar quads are not detected.
- `resolution="lo"` is accepted but does nothing yet.
- Colours stay as LDraw colour codes. `LDConfig.ldr` is not read.
- `!TEXMAP` is ignored, which gives the untextured fallback geometry.
- There are no automated tests yet.

## Roadmap

1. **Tier 1, watertight mesh.** Weld vertices, split T-junctions, resolve the overlaps where studs sit on a surface, remove internal faces, and validate (every edge shared by exactly two faces, no self-intersections, positive volume). First targets are 3001 and 3700. Candidate tools: `trimesh`, `manifold3d`, `libigl`.
2. **Tier 2, faceted STEP.** Merge coplanar triangles into planar faces and sew them into a solid with OpenCASCADE (`build123d` or `pythonocc`).
3. **Tier 3, analytic STEP.** Replace facet groups with true surfaces using the provenance tags, and replace polygonal boundary segments on neighbouring planar faces with arcs. Lift per face and fall back per face, so the solid is valid either way.
4. **Print tolerances.** Offset stud and tube radii by a printer-specific amount using the feature tags.

## Licence

The code licence has not been chosen yet.

The LDraw parts library is distributed separately under its own terms (CC BY 4.0 for current files; see `CAreadme.txt` in the library). Geometry produced by this tool is derived from those part files.

LDraw is a trademark of the Estate of James Jessiman. LEGO is a trademark of the LEGO Group, which does not sponsor or endorse this project.