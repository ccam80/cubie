"""Tests for the package source hash used to salt both cache layers."""

import re

from cubie._utils import package_source_hash


def _make_package(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_default_hash_is_hex_digest():
    result = package_source_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", result)


def test_default_hash_is_memoised():
    assert package_source_hash() == package_source_hash()


def test_hash_is_deterministic_for_directory(tmp_path):
    package_a = tmp_path / "a"
    package_b = tmp_path / "b"
    files = {"a.py": "x = 1\n", "sub/b.py": "y = 2\n"}
    _make_package(package_a, files)
    _make_package(package_b, files)
    first = package_source_hash(package_a)
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    # Identical content at a different resolved path hashes the same,
    # so the digest is content-driven rather than path- or run-local.
    assert package_source_hash(package_b) == first


def test_hash_changes_when_content_changes(tmp_path):
    package_a = tmp_path / "a"
    package_b = tmp_path / "b"
    _make_package(package_a, {"mod.py": "x = 1\n"})
    _make_package(package_b, {"mod.py": "x = 2\n"})
    assert package_source_hash(package_a) != package_source_hash(package_b)


def test_hash_changes_when_file_added(tmp_path):
    package_a = tmp_path / "a"
    package_b = tmp_path / "b"
    _make_package(package_a, {"mod.py": "x = 1\n"})
    _make_package(
        package_b, {"mod.py": "x = 1\n", "extra.py": "y = 2\n"}
    )
    assert package_source_hash(package_a) != package_source_hash(package_b)


def test_hash_changes_when_file_renamed(tmp_path):
    package_a = tmp_path / "a"
    package_b = tmp_path / "b"
    _make_package(package_a, {"mod.py": "x = 1\n"})
    _make_package(package_b, {"dom.py": "x = 1\n"})
    assert package_source_hash(package_a) != package_source_hash(package_b)


def test_hash_ignores_non_python_files(tmp_path):
    package_a = tmp_path / "a"
    package_b = tmp_path / "b"
    _make_package(package_a, {"mod.py": "x = 1\n"})
    _make_package(package_b, {"mod.py": "x = 1\n"})
    (package_b / "notes.txt").write_text("irrelevant", encoding="utf-8")
    assert package_source_hash(package_a) == package_source_hash(package_b)
