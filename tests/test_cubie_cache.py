"""Tests for cubie_cache module."""

import pytest
from numpy import array, float32

from attrs import define, field

from cubie.cubie_cache import (
    CUBIECacheLocator,
    CUBIECacheImpl,
    CUBIECache,
)


# --- Test fixtures (attrs classes for testing) ---


@define
class MockCompileSettings:
    """Simple attrs class for testing hash_compile_settings."""

    precision: type = float32
    value: int = 42


@define
class MockSettingsWithCallable:
    """Attrs class with eq=False field for testing."""

    precision: type = float32
    callback: object = field(default=None, eq=False)


@define
class MockNestedSettings:
    """Attrs class with nested attrs for testing."""

    precision: type = float32
    nested: MockCompileSettings = field(factory=MockCompileSettings)


@define
class MockSettingsWithArray:
    """Attrs class with numpy array for testing."""

    precision: type = float32
    data: object = field(factory=lambda: array([1.0, 2.0, 3.0]))


# --- CUBIECacheLocator tests ---


def test_cache_locator_get_cache_path():
    """Verify cache path is in generated/<system_name>/cache/."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "cache" in path


def test_cache_locator_get_source_stamp():
    """Verify source stamp returns system_hash."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert locator.get_source_stamp() == "abc123"


def test_cache_locator_get_disambiguator():
    """Verify disambiguator returns truncated settings hash."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456789012345678",
    )
    disambiguator = locator.get_disambiguator()
    assert len(disambiguator) == 16
    assert disambiguator == "def4567890123456"


def test_cache_locator_from_function_raises():
    """Verify from_function raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        CUBIECacheLocator.from_function(None, None)


# --- CUBIECacheImpl tests ---


def test_cache_impl_locator_property():
    """Verify locator property returns CUBIECacheLocator."""
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert isinstance(impl.locator, CUBIECacheLocator)


def test_cache_impl_filename_base():
    """Verify filename_base format."""
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456789012345678",
    )
    assert impl.filename_base.startswith("test_system-")
    assert "def4567890123456" in impl.filename_base


def test_cache_impl_check_cachable():
    """Verify check_cachable returns True."""
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert impl.check_cachable(None) is True


# --- CUBIECache tests ---


@pytest.mark.nocudasim
def test_cubie_cache_init():
    """Verify CUBIECache initializes with system info."""
    settings = MockCompileSettings()
    cache = CUBIECache(system_name="test_system", system_hash="abc123")
    assert cache._system_name == "test_system"
    assert cache._system_hash == "abc123"
    assert cache._name == "CUBIECache(test_system)"


@pytest.mark.nocudasim
def test_cubie_cache_index_key():
    """Verify _index_key includes system and settings hashes."""
    settings = MockCompileSettings()
    cache = CUBIECache(system_name="test_system", system_hash="abc123")

    # Create a mock codegen object
    class MockCodegen:
        def magic_tuple(self):
            return ("magic", "tuple")

    sig = ("float32", "float32")
    codegen = MockCodegen()

    key = cache._index_key(sig, codegen)

    # Key should be tuple of (sig, magic_tuple, system_hash, settings_hash)
    assert len(key) == 4
    assert key[0] == sig
    assert key[1] == ("magic", "tuple")
    assert key[2] == "abc123"

@pytest.mark.nocudasim
def test_cubie_cache_path():
    """Verify cache_path property returns expected path."""
    settings = MockCompileSettings()
    cache = CUBIECache(system_name="test_system", system_hash="abc123")
    assert "test_system" in cache.cache_path
    assert "cache" in cache.cache_path


# --- BatchSolverKernel integration tests ---


def test_batch_solver_kernel_no_cache_in_cudasim(solverkernel):
    """Verify no cache attached in CUDASIM mode.

    In CUDA simulation mode, file-based caching is disabled because the
    numba caching infrastructure is not available. The kernel should not
    have a _cache attribute when running under the simulator.
    """
    from cubie.cuda_simsafe import is_cudasim_enabled

    # Build the kernel to trigger cache attachment logic
    kernel = solverkernel.kernel

    if is_cudasim_enabled():
        # In CUDASIM mode, cache should not be attached
        assert not hasattr(kernel, "_cache") or kernel._cache is None
    # When not in CUDASIM, cache may or may not be attached depending on
    # caching_enabled setting - that's tested separately
