# SPDX-License-Identifier: GPL-3.0-or-later
"""Tiers 2 and 3: every reference part becomes one valid STEP solid."""
import pytest

pytest.importorskip("manifold3d")
pytest.importorskip("OCP")

from ldraw2solid import brep  # noqa: E402
from ldraw2solid.mesh import solidify  # noqa: E402

from conftest import REFERENCE_PARTS  # noqa: E402

# Planar faces after merging coplanar triangles, measured when Tier 2 first passed.
TIER2_FACES = {"3001.dat": 249, "3062b.dat": 85, "3700.dat": 159}


@pytest.fixture(scope="module")
def solid(flat):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = solidify(flat(name))
        return cache[name]
    return get


def assert_valid(r):
    assert r["valid"]
    assert r["solids"] == 1 and r["shells"] == 1
    assert r["free_edges"] == 0
    assert r["volume_err_mm3"] < 1e-6
    assert r["bounds_err_mm"] < 1e-6


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier2_step_roundtrip(solid, tmp_path, part):
    s = solid(part)
    vol, bounds = brep.mesh_reference(s)
    b = brep.faceted_brep(s)
    assert_valid(brep.check(b.shape, vol, bounds))
    path = str(tmp_path / "part.step")
    brep.write_step(path, b.shape)
    r = brep.check(brep.read_step(path), vol, bounds)
    assert_valid(r)
    assert r["face_types"] == {"plane": TIER2_FACES[part]}


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier2_provenance(solid, part):
    b = brep.faceted_brep(solid(part))
    assert len(b.face_regions) == TIER2_FACES[part]
    assert all(ids and min(ids) >= 0 for ids in b.face_regions)


# Tier 3 face types, measured when it first passed. In 3700 the underside pin
# and the half cylinder it cuts into stay faceted: they meet along a saddle.
TIER3_FACES = {
    "3001.dat": {"plane": 25, "cylinder": 14},
    "3062b.dat": {"plane": 5, "cylinder": 5},
    "3700.dat": {"plane": 47, "cylinder": 7},
}
TIER3_FACETED = {"3001.dat": 0, "3062b.dat": 0, "3700.dat": 2}


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier3_step_roundtrip(flat, solid, tmp_path, part):
    s = solid(part)
    _, bounds = brep.mesh_reference(s)
    b = brep.analytic_brep(s, flat(part))
    assert len(b.stats["cylinders_faceted"]) == TIER3_FACETED[part]
    path = str(tmp_path / "part.step")
    brep.write_step(path, b.shape)
    r = brep.check(brep.read_step(path), b.stats["expected_volume_mm3"], bounds)
    assert r["valid"]
    assert r["solids"] == 1 and r["shells"] == 1
    assert r["free_edges"] == 0
    # Measured worst case 2e-4 mm^3 (3001), against a volume of 2535 mm^3.
    assert r["volume_err_mm3"] < 1e-3
    assert r["bounds_err_mm"] < 1e-6
    assert r["face_types"] == TIER3_FACES[part]


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier3_lifting_adds_circle_segments(flat, solid, part):
    """True cylinders differ from the 16-gon by the circle segments, in the right direction."""
    s = solid(part)
    vol, _ = brep.mesh_reference(s)
    b = brep.analytic_brep(s, flat(part))
    assert b.stats["expected_volume_mm3"] == pytest.approx(vol + b.stats["volume_delta_mm3"], abs=0.05)
    assert all(ids and min(ids) >= 0 for ids in b.face_regions)


def test_tier3_6377(flat, solid, tmp_path):
    """Duplo track: stepped tubes underneath lift; the STEP meshes without slits."""
    s = solid("6377.dat")
    b = brep.analytic_brep(s, flat("6377.dat"))
    assert b.stats["cylinders_lifted"] == 12
    path = str(tmp_path / "part.step")
    brep.write_step(path, b.shape)
    shape = brep.read_step(path)
    r = brep.check(shape)
    assert r["valid"] and r["solids"] == 1 and r["free_edges"] == 0
    assert r["face_types"]["cylinder"] == 16
    m = brep.tessellation_check(shape)
    assert m["open_edges"] == 0
    # The female connectors' outer wall touches a cavity wall along a line.
    assert m["edges_3plus"] == 2
