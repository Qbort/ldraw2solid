# SPDX-License-Identifier: GPL-3.0-or-later
"""Tier 1: every reference part becomes one valid closed mesh."""
import numpy as np
import pytest

pytest.importorskip("manifold3d")

from ldraw2solid.mesh import solidify, validate  # noqa: E402

from conftest import REFERENCE_PARTS  # noqa: E402

# Volumes of the 16-segment meshes, measured when Tier 1 first passed (LDU^3).
VOLUME = {"3001.dat": 39390.2532, "3062b.dat": 4641.2796, "3700.dat": 12101.5283}


@pytest.fixture(scope="module")
def solid(flat):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = solidify(flat(name))
        return cache[name]
    return get


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier1_valid(flat, solid, part):
    r = validate(solid(part), flat(part))
    assert r["open_edges"] == 0
    assert r["edges_3plus"] == 0
    assert r["flipped_edges"] == 0
    assert r["self_intersections"] == 0
    assert r["cap_tris_left"] == 0
    assert r["off_surface"] == 0
    assert r["bounds_err_ldu"] < 1e-3
    assert r["volume_ldu3"] == pytest.approx(VOLUME[part], abs=1e-3)
    assert r["ok"]


@pytest.mark.parametrize("part", REFERENCE_PARTS)
def test_tier1_provenance(flat, solid, part):
    f, s = flat(part), solid(part)
    assert (s.tri_source >= 0).all()
    np.testing.assert_array_equal(s.tri_instance, f.tri_instance[s.tri_source])
    # Studs survive the union with their own provenance.
    names = {inst.name for iid in np.unique(s.tri_instance) for inst in f.chain(int(iid))}
    assert any(n.startswith("stud") for n in names)


def test_tier1_6377(flat):
    """Duplo track 4x8: needs edge sorting, tolerant welding, loop splitting and
    cancelling the places where a piece lies flat against itself."""
    f = flat("6377.dat")
    s = solidify(f)
    r = validate(s, f)
    assert r["ok"], r
    assert r["off_surface"] == 0
    assert r["volume_ldu3"] == pytest.approx(315239.0716, abs=1e-2)
    assert s.stats["sorted_edges"] == 2
    assert s.stats["loops_split"] == 38
    assert s.stats["self_contact_planes"] == 6
    assert s.stats["double_wall_cuts"] == 0
