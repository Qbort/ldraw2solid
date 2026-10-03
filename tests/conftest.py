import os

import pytest

from ldraw2solid import Flattener, Library

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("LDRAW_LIB", os.path.join(ROOT, "lib"))

REFERENCE_PARTS = ("3001.dat", "3062b.dat", "3700.dat")


@pytest.fixture(scope="session")
def flattener():
    if not os.path.isdir(os.path.join(LIB, "parts")):
        pytest.skip(f"LDraw library not found at {LIB} (set LDRAW_LIB)")
    return Flattener(Library(LIB))


@pytest.fixture(scope="session")
def flat(flattener):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = flattener.flatten(name)
        return cache[name]
    return get
