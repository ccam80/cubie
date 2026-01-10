"""Tests for :mod:`cubie.odesystems.symbolic.odefile`."""

import uuid
from pathlib import Path

import pytest

from cubie.odesystems.symbolic.odefile import HEADER, ODEFile


@pytest.fixture()
def unique_odefile():
    name = f"test_{uuid.uuid4().hex}"
    odefile = ODEFile(name, "hash1")
    yield odefile, name
    generated_dir = Path("generated") / name
    if generated_dir.exists():
        try:
            import shutil

            shutil.rmtree(generated_dir)
        except OSError:
            # Race condition guard; another thread already removed it or is
            # using it
            pass


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


# New tests covering additional caching and generation behavior


def test_creates_generated_dir_when_missing(tmp_path, monkeypatch):
    # Point GENERATED_DIR to a temp path and ensure it doesn't exist
    from cubie.odesystems import symbolic as symbolic_pkg

    odefile_mod = symbolic_pkg.odefile
    temp_gen = tmp_path / "generated_temp"
    if temp_gen.exists():
        # Ensure clean slate
        for p in temp_gen.glob("*"):
            p.unlink()
        temp_gen.rmdir()
    monkeypatch.setattr(odefile_mod, "GENERATED_DIR", temp_gen, raising=True)

    name = f"test_{uuid.uuid4().hex}"
    odef = ODEFile(name, "hashX")
    # Directory should have been created
    assert temp_gen.exists() and temp_gen.is_dir()
    # File should be within the system-specific subdirectory
    assert odef.file_path.parent.parent == temp_gen
    assert odef.file_path.parent.name == name


def test_cache_persists_across_instances_same_hash(unique_odefile):
    _, name = unique_odefile
    code = _simple_code("foo_factory")
    o1 = ODEFile(name, "hash1")
    o1.import_function("foo_factory", code)
    # New instance with same name and same hash should reuse cache without code
    o2 = ODEFile(name, "hash1")
    factory = o2.import_function("foo_factory")
    assert factory()() == 1


def test_multiple_factories_appended_and_importable(unique_odefile):
    odefile, _ = unique_odefile
    code1 = _simple_code("foo_factory")
    code2 = _simple_code("bar_factory")
    f1 = odefile.import_function("foo_factory", code1)
    assert f1()() == 1
    # Append second factory and ensure both coexist
    f2 = odefile.import_function("bar_factory", code2)
    assert f2()() == 1
    # Re-import first factory without code; should still work
    f1_cached = odefile.import_function("foo_factory")
    assert f1_cached()() == 1


def test_malformed_cached_function_triggers_regeneration(unique_odefile):
    odefile, _ = unique_odefile
    # Manually append a malformed function (missing `return base_name`)
    with open(odefile.file_path, "a", encoding="utf-8") as f:
        f.write("def foo_factory():\n")
        f.write("    pass\n")
    # Without code_lines this should raise
    with pytest.raises(ValueError):
        odefile.import_function("foo_factory")
    # Providing code should append a correct definition that is importable
    factory = odefile.import_function(
        "foo_factory", _simple_code("foo_factory")
    )
    assert factory()() == 1


def test_header_contains_required_imports():
    assert "from numba import cuda" in HEADER
    assert "import math" in HEADER
