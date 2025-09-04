"""Tests for :mod:`cubie.systemmodels.symbolic.odefile`."""

import uuid
from pathlib import Path

import pytest

from cubie.systemmodels.symbolic.odefile import HEADER, ODEFile


@pytest.fixture()
def unique_odefile():
    name = f"test_{uuid.uuid4().hex}"
    odefile = ODEFile(name, "hash1")
    yield odefile, name
    generated = Path("generated") / f"{name}.py"
    if generated.exists():
        generated.unlink()


def _simple_code(func_name: str) -> str:
    base = func_name.replace("_factory", "")
    return (
        f"def {func_name}():\n"
        f"    def {base}():\n"
        "        return 1\n"
        f"    return {base}\n"
    )


def test_header_contains_notice():
    assert "# This file was generated automatically" in HEADER


def test_import_generates_and_returns(unique_odefile):
    odefile, _ = unique_odefile
    code = _simple_code("foo_factory")
    factory = odefile.import_function("foo_factory", code)
    assert factory()() == 1


def test_import_uses_cache(unique_odefile):
    odefile, _ = unique_odefile
    code = _simple_code("foo_factory")
    odefile.import_function("foo_factory", code)
    factory = odefile.import_function("foo_factory")
    assert factory()() == 1


def test_import_missing_without_code_raises(unique_odefile):
    odefile, _ = unique_odefile
    with pytest.raises(ValueError):
        odefile.import_function("missing_factory")


def test_import_reinitialises_on_hash_change(unique_odefile):
    _, name = unique_odefile
    odefile1 = ODEFile(name, "hash1")
    code = _simple_code("foo_factory")
    odefile1.import_function("foo_factory", code)
    odefile2 = ODEFile(name, "hash2")
    with pytest.raises(ValueError):
        odefile2.import_function("foo_factory")
    factory = odefile2.import_function("foo_factory", code)
    assert factory()() == 1
