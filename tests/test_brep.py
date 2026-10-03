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
