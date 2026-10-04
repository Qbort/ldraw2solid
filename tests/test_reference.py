# SPDX-License-Identifier: GPL-3.0-or-later
"""Parser reference values from AGENTS.MD. Any parser change must keep these."""
import numpy as np
import pytest

from ldraw2solid import surfaces
from ldraw2solid.mesh import topology
from ldraw2solid.primitives import hole_boss_mismatches

REFERENCE = {
    # part: (bounds lo, bounds hi, triangles, open edges after weld)
    "3001.dat": ((-40, -4, -20), (40, 24, 20), 700, 224),
    "3062b.dat": ((-10, -4, -10), (10, 24, 10), 384, 0),
    "3700.dat": ((-20, -4, -10), (20, 24, 10), 550, 126),
}


@pytest.mark.parametrize("part", sorted(REFERENCE))
def test_flatten_reference(flat, part):
    lo, hi, n_tris, n_open = REFERENCE[part]
    f = flat(part)
    assert not f.missing
    got_lo, got_hi = f.bounds()
    np.testing.assert_allclose(got_lo, lo, atol=1e-9)
    np.testing.assert_allclose(got_hi, hi, atol=1e-9)
    assert len(f.tris) == n_tris
    assert topology(f.tris)[0] == n_open
    assert hole_boss_mismatches(f) == 0
    assert f.tri_certified.all()


@pytest.mark.parametrize("part", sorted(REFERENCE))
def test_stud_cylinders(flat, part):
    studs = [s for s in surfaces(flat(part)).values()
             if s.kind == "cylinder" and s.feature == "stud" and not s.inverted]
    assert studs
    for s in studs:
        if abs(s.radius * 0.4 - 2.4) < 1e-9:
            assert np.linalg.norm(s.axis) * 0.4 == pytest.approx(1.6)
    assert any(abs(s.radius * 0.4 - 2.4) < 1e-9 for s in studs)
