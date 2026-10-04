# ldraw2solid

> [!WARNING]
> **This code is entirely AI-generated and has not been reviewed or checked for correctness.**
> It may produce wrong geometry without telling you. Inspect every model before you print it or use it in CAD, and do not rely on it for anything that matters.

ldraw2solid turns a LEGO part from the [LDraw](https://www.ldraw.org) library into a solid you can 3D print or open in CAD:

- **STL and 3MF**: a closed, watertight mesh for your slicer.
- **STEP**: a solid with true round cylinders where the part allows it, for CAD programs.

All files are in millimetres with Z up.

## What you need

- Python 3.10 or newer.
- The LDraw parts library. Download [complete.zip](https://library.ldraw.org/library/updates/complete.zip) and unpack it into a folder named `lib` next to this README, so that you have `lib/parts/` and `lib/p/`. It is about 500 MB.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Convert a part

Give the part's file name from the library:

```
.venv/bin/python make_solid.py lib 3001.dat --stl --3mf --step
```

The files are written to `out/` (`out/3001.stl`, `out/3001.3mf`, `out/3001.step`). Use `--out DIR` for another folder. You can list several parts in one command.

For each part the tool checks its own output and prints a report that ends in `OK` or `NOT VALID`. If a part says `NOT VALID`, look at the model carefully before using it.

Options:

| Option | Effect |
|---|---|
| `--stl` | Write a mesh for slicers. |
| `--3mf` | Write the same mesh as 3MF. |
| `--step` | Write a STEP solid. |
| `--tier 2` | STEP with flat facets only, no round surfaces. |
| `--hi` | Use the library's smoother 48-sided primitives. |

## Limitations

- **Tested parts.** It has only been tried on a few parts: 3001 (brick 2×4), 3062b (round brick 1×1), 3700 (Technic brick 1×2 with hole) and 6377 (Duplo track). Other parts may fail or come out wrong.
- **Size.** Studs and holes have the nominal LDraw size, with no clearance for printing. Parts may fit too tightly.
- **Round surfaces in STEP.** Only cylinders become round. Cones, spheres, curved hand-made surfaces, and cylinders that meet another curved surface stay faceted.
- **Colours and decorations.** Colours are not exported, and printed patterns are ignored.

How the conversion works, what has been measured and what is still open: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Licence

Copyright (C) 2026 Roberto Fernandez Bautista

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. The full text is in [LICENSE](LICENSE).

The LDraw parts library is not part of this project. It is distributed under its own terms (CC BY 4.0 for current files; see `CAreadme.txt` in the library), and models made with this tool are derived from those part files.

LDraw is a trademark of the Estate of James Jessiman. LEGO is a trademark of the LEGO Group, which does not sponsor or endorse this project.
