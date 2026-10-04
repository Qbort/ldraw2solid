# SPDX-License-Identifier: GPL-3.0-or-later
"""
LDraw parser with provenance tagging.

Flattens an LDraw file (.dat / .ldr / .mpd) into triangles and edge lines while
keeping, for every triangle, a pointer to the subfile *instance* it came from.
Each instance records its name, parent, cumulative transform and whether it was
turned inside-out by BFC INVERTNEXT, so later stages can replace e.g. all faces
of one `4-4cyli.dat` instance by a true analytic cylinder.

Coordinates stay in LDraw units (LDU, -Y up).  Use `to_print_frame()` to get
millimetres, Z up.

Winding: every triangle is emitted counter-clockwise seen from outside (when
the source is BFC certified; `tri_certified` says whether that can be trusted).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LDU_TO_MM = 0.4
# LDraw (x, y, z), -Y up  ->  print frame (x, z, -y), +Z up.  Proper rotation.
LDRAW_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])

MAIN_COLOUR = 16
EDGE_COLOUR = 24


# ----------------------------------------------------------------------------
# Library lookup
# ----------------------------------------------------------------------------
class Library:
    """Finds files in an LDraw library folder (case-insensitive, like LDraw).

    resolution: 'std' (16-gon primitives), 'hi' (48-gon, p/48) or 'lo' (p/8).
    stud_logo:  False, or a logo variant number 1-5 to swap stud.dat for
                stud-logoN.dat when the library has it.
    """

    def __init__(self, root: str, resolution: str = "std", stud_logo=False,
                 extra_dirs: tuple = ()):
        self.root = os.path.abspath(root)
        self.resolution = resolution
        self.stud_logo = stud_logo
        self._index: dict[str, tuple[str, str]] = {}  # key -> (path, category)
        # Later entries do not override earlier ones: official beats unofficial.
        for sub, cat in (("parts", "part"), ("p", "primitive"), ("models", "model"),
                         ("unofficial/parts", "part"), ("unofficial/p", "primitive")):
            self._scan(os.path.join(self.root, sub), cat)
        for d in extra_dirs:
            self._scan(d, "model")
        if not self._index:
            raise FileNotFoundError(f"no LDraw library found under {self.root}")

    def _scan(self, base: str, cat: str):
        if not os.path.isdir(base):
            return
        for dirpath, _dirs, files in os.walk(base):
            rel = os.path.relpath(dirpath, base).replace(os.sep, "/").lower()
            rel = "" if rel == "." else rel + "/"
            for fn in files:
                if not fn.lower().endswith((".dat", ".ldr", ".mpd")):
                    continue
                c = cat
                if cat == "part" and rel.startswith("s/"):
                    c = "subpart"
                # parts and p share one namespace for lookup: "s/3001s01.dat",
                # "48/4-4cyli.dat", "stud.dat".
                self._index.setdefault(rel + fn.lower(), (os.path.join(dirpath, fn), c))

    @staticmethod
    def normalise(name: str) -> str:
        return name.strip().replace("\\", "/").lower()

    def resolve(self, name: str) -> Optional[tuple[str, str, str]]:
        """-> (key, path, category) or None."""
        key = self.normalise(name)
        if self.stud_logo and key == "stud.dat":
            alt = f"stud-logo{int(self.stud_logo)}.dat"
            if alt in self._index:
                key = alt
        base = key.split("/")[-1]
        if self.resolution == "hi" and "/" not in key and ("48/" + base) in self._index:
            key = "48/" + base
        elif self.resolution == "std" and key.startswith("8/") and base in self._index:
            key = base
        hit = self._index.get(key)
        if hit is None and os.path.isfile(name):
            return (key, name, "model")
        return (key, hit[0], hit[1]) if hit else None


# ----------------------------------------------------------------------------
# Per-file parse (no recursion): raw commands in the file's own coordinates
# ----------------------------------------------------------------------------
@dataclass
class Ref:
    colour: int
    matrix: np.ndarray        # 4x4
    name: str
    invert: bool              # preceded by BFC INVERTNEXT


@dataclass
class RawFile:
    key: str
    category: str             # part | subpart | primitive | model
    description: str = ""
    certified: Optional[bool] = None   # None = no BFC statement at all
    refs: list = field(default_factory=list)
    tris: list = field(default_factory=list)       # (colour, 9 floats, clip_ok, quad_id)
    lines: list = field(default_factory=list)      # (colour, 6 floats)
    clines: list = field(default_factory=list)     # (colour, 12 floats)
    order: list = field(default_factory=list)      # ('ref', i) markers are implicit


def _colour(tok: str) -> int:
    return int(tok, 16) if tok.lower().startswith("0x") else int(tok)


def parse_text(text: str, key: str, category: str) -> dict[str, RawFile]:
    """Parse one physical file. Returns {key: RawFile}; MPD files yield several."""
    files: dict[str, RawFile] = {}
    cur = RawFile(key, category)
    files[key] = cur
    first_file_seen = False
    winding_ccw, clip, invert_next = True, True, False
    got_desc = False
    in_data = False
    quad_counter = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tok = line.split()
        lt = tok[0]

        if lt == "0":
            up = [t.upper() for t in tok[1:]]
            if up[:1] == ["FILE"]:
                name = Library.normalise(line.split(None, 2)[2])
                if not first_file_seen:
                    # first embedded file *is* the main model
                    first_file_seen = True
                    cur.embedded_name = name  # type: ignore[attr-defined]
                    files[name] = cur
                else:
                    cur = RawFile(name, "model")
                    files[name] = cur
                winding_ccw, clip, invert_next, got_desc, in_data = True, True, False, False, False
                continue
            if up[:1] == ["NOFILE"]:
                cur = RawFile("__nofile__", "model")
                continue
            if up[:1] == ["!DATA"]:
                in_data = True
                continue
            if up[:1] == ["BFC"]:
                for flag in up[1:]:
                    if flag == "CERTIFY":
                        if cur.certified is None:
                            cur.certified = True
                    elif flag == "NOCERTIFY":
                        cur.certified = False
                    elif flag == "CCW":
                        winding_ccw = True
                    elif flag == "CW":
                        winding_ccw = False
                    elif flag == "CLIP":
                        clip = True
                    elif flag == "NOCLIP":
                        clip = False
                    elif flag == "INVERTNEXT":
                        invert_next = True
                continue
            if not got_desc and len(tok) > 1 and not tok[1].startswith("!") \
                    and up[0] not in ("NAME:", "AUTHOR:", "//"):
                cur.description = line[1:].strip()
                got_desc = True
            # every other meta (TEXMAP, HISTORY, STEP, ...) is ignored
            continue

        if in_data:
            continue
        try:
            if lt == "1" and len(tok) >= 15:
                v = [float(x) for x in tok[2:14]]
                m = np.eye(4)
                m[:3, 3] = v[0:3]
                m[:3, :3] = np.array(v[3:12]).reshape(3, 3)
                name = line.split(None, 14)[14]
                cur.refs.append(Ref(_colour(tok[1]), m, name, invert_next))
            elif lt == "2" and len(tok) >= 8:
                cur.lines.append((_colour(tok[1]), [float(x) for x in tok[2:8]]))
            elif lt == "3" and len(tok) >= 11:
                p = [float(x) for x in tok[2:11]]
                if not winding_ccw:
                    p = p[0:3] + p[6:9] + p[3:6]
                cur.tris.append((_colour(tok[1]), p, clip, -1))
            elif lt == "4" and len(tok) >= 14:
                p = [float(x) for x in tok[2:14]]
                a, b, c, d = p[0:3], p[3:6], p[6:9], p[9:12]
                if not winding_ccw:
                    b, d = d, b
                col = _colour(tok[1])
                cur.tris.append((col, a + b + c, clip, quad_counter))
                cur.tris.append((col, a + c + d, clip, quad_counter))
                quad_counter += 1
            elif lt == "5" and len(tok) >= 14:
                cur.clines.append((_colour(tok[1]), [float(x) for x in tok[2:14]]))
        except ValueError:
            pass  # malformed line: skip, like every LDraw viewer does
        invert_next = False
    files.pop("__nofile__", None)
    return files


# ----------------------------------------------------------------------------
# Flattened result
# ----------------------------------------------------------------------------
@dataclass
class Instance:
    """One placement of a subfile inside the flattened model."""
    id: int
    parent: int               # -1 for the root file
    name: str                 # normalised library key, e.g. '4-4cyli.dat'
    category: str             # part | subpart | primitive | model
    matrix: np.ndarray        # cumulative 4x4, file-local -> root coordinates
    inverted: bool            # odd number of INVERTNEXT above it (hole vs. boss)
    description: str = ""


@dataclass
class Flat:
    tris: np.ndarray              # (T, 3, 3) float64
    tri_colour: np.ndarray        # (T,) int   (16 = inherit from whoever uses this)
    tri_instance: np.ndarray      # (T,) int   index into instances
    tri_certified: np.ndarray     # (T,) bool  winding can be trusted
    tri_quad: np.ndarray          # (T,) int   same id on the 2 halves of a quad, else -1
    lines: np.ndarray             # (L, 2, 3)  type 2: hard edges
    line_instance: np.ndarray     # (L,)
    clines: np.ndarray            # (C, 4, 3)  type 5: smooth edge + 2 control points
    cline_instance: np.ndarray    # (C,)
    instances: list               # list[Instance]
    missing: set = field(default_factory=set)

    # -- convenience ---------------------------------------------------------
    def chain(self, inst_id: int) -> list:
        """Instances from the given one up to the root."""
        out = []
        while inst_id >= 0:
            out.append(self.instances[inst_id])
            inst_id = self.instances[inst_id].parent
        return out

    def bounds(self):
        v = self.tris.reshape(-1, 3)
        return v.min(0), v.max(0)

    def to_print_frame(self) -> np.ndarray:
        """Triangles in millimetres, Z up, winding preserved."""
        return (self.tris @ LDRAW_TO_ZUP.T) * LDU_TO_MM


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:3, :3].T + m[:3, 3]


class Flattener:
    """Recursively resolves subfile references, caching each file once."""

    def __init__(self, library: Library):
        self.lib = library
        self._raw: dict[str, RawFile] = {}
        self._flat: dict[str, Flat] = {}
        self._stack: list[str] = []

    # -- raw file access -----------------------------------------------------
    def _load(self, name: str, local: dict) -> Optional[RawFile]:
        key = Library.normalise(name)
        if key in local:
            return local[key]
        hit = self.lib.resolve(name)
        if hit is None:
            return None
        key, path, cat = hit
        if key not in self._raw:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                parsed = parse_text(fh.read(), key, cat)
            self._raw[key] = parsed[key]
            if len(parsed) > 1:                       # MPD: keep its embedded files
                self._raw[key].embedded = parsed      # type: ignore[attr-defined]
        return self._raw[key]

    # -- public --------------------------------------------------------------
    def flatten(self, name: str) -> Flat:
        raw = self._load(name, {})
        if raw is None:
            raise FileNotFoundError(name)
        return self._flatten(raw, getattr(raw, "embedded", {}))

    def forget(self, keep=("primitive",)):
        """Drop cached results except the given categories (for batch runs)."""
        for cache in (self._flat, self._raw):
            for k in [k for k, v in cache.items()
                      if (v.instances[0].category if isinstance(v, Flat) else v.category) not in keep]:
                del cache[k]

    # -- recursion -----------------------------------------------------------
    def _flatten(self, raw: RawFile, local: dict) -> Flat:
        cache_key = raw.key
        if cache_key in self._flat:
            return self._flat[cache_key]
        if cache_key in self._stack:
            raise RecursionError("circular reference: " + " -> ".join(self._stack + [cache_key]))
        self._stack.append(cache_key)

        # Geometry lives in a library file whose winding nobody vouched for.
        own_cert = bool(raw.certified)
        # An uncertified part/primitive also spoils what it references, because
        # its INVERTNEXTs are missing.  A plain model file does not.
        taints = (not own_cert) and raw.category != "model"

        instances = [Instance(0, -1, raw.key, raw.category, np.eye(4), False, raw.description)]
        T = [np.array([t[1] for t in raw.tris], dtype=float).reshape(-1, 3, 3)]
        Tc = [np.array([t[0] for t in raw.tris], dtype=np.int64)]
        Ti = [np.zeros(len(raw.tris), dtype=np.int64)]
        Tq = [np.array([t[3] for t in raw.tris], dtype=np.int64)]
        Tcert = [np.array([own_cert and t[2] for t in raw.tris], dtype=bool)]
        L = [np.array([l[1] for l in raw.lines], dtype=float).reshape(-1, 2, 3)]
        Li = [np.zeros(len(raw.lines), dtype=np.int64)]
        C = [np.array([c[1] for c in raw.clines], dtype=float).reshape(-1, 4, 3)]
        Ci = [np.zeros(len(raw.clines), dtype=np.int64)]
        missing: set = set()
        quad_base = (int(Tq[0].max()) + 1) if len(raw.tris) else 0

        for ref in raw.refs:
            child_raw = self._load(ref.name, local)
            if child_raw is None:
                missing.add(Library.normalise(ref.name))
                continue
            child = self._flatten(child_raw, {**local, **getattr(child_raw, "embedded", {})})
            missing |= child.missing
            off = len(instances)
            for ci in child.instances:
                instances.append(Instance(
                    ci.id + off, 0 if ci.parent < 0 else ci.parent + off, ci.name, ci.category,
                    ref.matrix @ ci.matrix, ci.inverted ^ ref.invert, ci.description))

            tris = _apply(ref.matrix, child.tris)
            # A mirroring matrix reverses winding; INVERTNEXT asks for reversal.
            if (np.linalg.det(ref.matrix[:3, :3]) < 0) ^ ref.invert:
                tris = tris[:, ::-1, :]
            T.append(tris)
            col = child.tri_colour.copy()
            col[col == MAIN_COLOUR] = ref.colour
            Tc.append(col)
            Ti.append(child.tri_instance + off)
            Tcert.append(child.tri_certified & (not taints))
            q = child.tri_quad.copy()
            q[q >= 0] += quad_base
            quad_base = max(quad_base, int(q.max()) + 1 if len(q) else quad_base)
            Tq.append(q)
            L.append(_apply(ref.matrix, child.lines))
            Li.append(child.line_instance + off)
            C.append(_apply(ref.matrix, child.clines))
            Ci.append(child.cline_instance + off)

        flat = Flat(np.concatenate(T), np.concatenate(Tc), np.concatenate(Ti),
                    np.concatenate(Tcert), np.concatenate(Tq),
                    np.concatenate(L), np.concatenate(Li),
                    np.concatenate(C), np.concatenate(Ci), instances, missing)
        self._stack.pop()
        self._flat[cache_key] = flat
        return flat


def load(name: str, library_root: str, resolution: str = "std", stud_logo=False) -> Flat:
    """One-call helper: flatten `name` (e.g. '3001.dat') from the library."""
    return Flattener(Library(library_root, resolution, stud_logo)).flatten(name)
