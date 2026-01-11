"""Tests for cubie_cache module."""

import pytest
from numpy import array, float32

from attrs import define, field

from cubie.cubie_cache import (
    CUBIECacheLocator,
    CUBIECacheImpl,
    CUBIECache,
    CubieCacheHandler,
    CacheConfig,
    ALL_CACHE_PARAMETERS,
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
    """Verify cache path includes system_hash subdirectory."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "abc123" in path  # system_hash in path
    assert path.endswith("CUDA_cache")


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


def test_cache_locator_path_includes_system_hash():
    """Verify cache path follows generated/<system_name>/<hash>/CUDA_cache."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "abc123" in path
    assert path.endswith("CUDA_cache")


def test_cache_locator_empty_hash_uses_default():
    """Verify empty system_hash uses 'default' subdirectory."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "default" in path
    assert path.endswith("CUDA_cache")


def test_cache_locator_custom_dir_not_affected_by_hash_change(tmp_path):
    """Verify custom cache_dir is preserved when set_system_hash is called."""
    custom_dir = tmp_path / "my_custom_cache"
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="original_hash",
        compile_settings_hash="def456",
        custom_cache_dir=custom_dir,
    )
    original_path = locator.get_cache_path()
    assert str(custom_dir) in original_path

    # Update the system hash
    locator.set_system_hash("new_hash")

    # Path should remain the same (custom dir is unchanged)
    new_path = locator.get_cache_path()
    assert new_path == original_path
    assert str(custom_dir) in new_path


# --- ALL_CACHE_PARAMETERS and CacheConfig tests ---


def test_all_cache_parameters_contains_expected_keys():
    """Verify ALL_CACHE_PARAMETERS contains expected cache parameter names."""
    expected_keys = {"cache_enabled", "cache_mode", "max_cache_entries",
                     "cache_dir"}
    assert ALL_CACHE_PARAMETERS == expected_keys


def test_cache_config_has_cache_enabled_field():
    """Verify CacheConfig has cache_enabled field (not enabled)."""
    config = CacheConfig()
    assert hasattr(config, "cache_enabled")
    assert not hasattr(config, "enabled")
    assert config.cache_enabled is False

    # Verify it can be set to True
    config_enabled = CacheConfig(cache_enabled=True)
    assert config_enabled.cache_enabled is True


def test_cache_config_has_system_name_field():
    """Verify CacheConfig has system_name field."""
    config = CacheConfig()
    assert hasattr(config, "system_name")
    assert config.system_name == ""

    # Verify it can be set
    config_with_name = CacheConfig(system_name="my_system")
    assert config_with_name.system_name == "my_system"


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


def test_cubie_cache_init():
    """Verify CUBIECache initializes with system info."""
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
    )
    assert cache._system_name == "test_system"
    assert cache._system_hash == "abc123"
    assert cache._name == "CUBIECache(test_system)"


def test_cubie_cache_index_key():
    """Verify _index_key includes system and settings hashes."""
    config_hash = (
        "def456789012345678901234567890123456789012345678901234567890abcd"
    )
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash=config_hash,
    )

    # Create a mock codegen object
    class MockCodegen:
        def magic_tuple(self):
            return ("magic", "tuple")

    sig = ("float32", "float32")
    codegen = MockCodegen()

    key = cache._index_key(sig, codegen)

    # Key should be tuple of (sig, magic_tuple, system_hash, config_hash)
    assert len(key) == 4
    assert key[0] == sig
    assert key[1] == ("magic", "tuple")
    assert key[2] == "abc123"
    assert key[3] == config_hash


def test_cubie_cache_path():
    """Verify cache_path includes system_hash subdirectory."""
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456"
        "789012345678901234567890abcd",
    )
    path_str = str(cache.cache_path)
    assert "test_system" in path_str
    assert "abc123" in path_str  # system_hash in path
    assert "CUDA_cache" in path_str


# --- BatchSolverKernel integration tests ---


def test_batch_solver_kernel_builds_without_error(solverkernel):
    """Verify BatchSolverKernel builds successfully in all modes.

    This smoke test confirms the kernel compilation path executes
    without raising exceptions, regardless of CUDA/CUDASIM mode.
    """
    # Build the kernel - this exercises the full compilation path
    kernel = solverkernel.kernel

    # Verify kernel was created
    assert kernel is not None


# --- CUDASIM Mode Compatibility Tests ---


def test_cache_locator_instantiation_works():
    """Verify CUBIECacheLocator can be instantiated regardless of mode."""
    locator = CUBIECacheLocator(
        system_name="cudasim_test",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    # Path operations should work
    assert locator.get_cache_path() is not None
    assert locator.get_source_stamp() == "abc123"
    assert locator.get_disambiguator() == "def456"


def test_cache_impl_instantiation_works():
    """Verify CUBIECacheImpl can be instantiated regardless of mode."""
    impl = CUBIECacheImpl(
        system_name="cudasim_test",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    # Properties should be accessible
    assert impl.locator is not None
    assert impl.filename_base is not None


# --- Module-level function tests ---


def test_create_cache_returns_none_when_disabled():
    """Verify create_cache returns None when caching disabled."""
    from cubie.cubie_cache import create_cache

    result = create_cache(
        cache_arg=False,
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )
    assert result is None


def test_create_cache_returns_cache_when_enabled():
    """Verify create_cache returns CUBIECache when enabled."""
    from cubie.cubie_cache import create_cache, CUBIECache

    result = create_cache(
        cache_arg=True,
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456"
        "789012345678901234567890abcd",
    )
    assert isinstance(result, CUBIECache)


def test_invalidate_cache_no_op_when_hash_mode(tmp_path):
    """Verify invalidate_cache does nothing in hash mode."""
    from cubie.cubie_cache import invalidate_cache

    # Create a marker file
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    marker = cache_dir / "marker.txt"
    marker.write_text("test")

    # invalidate_cache with hash mode should not touch files
    invalidate_cache(
        cache_arg=True,  # hash mode by default
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )

    # Marker file should still exist (no flush happened)
    assert marker.exists()


def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
    """Verify invalidate_cache flushes cache in flush_on_change mode."""
    from cubie.cubie_cache import invalidate_cache

    # Create cache directory with marker file
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker_file = cache_dir / "marker.txt"
    marker_file.write_text("test")
    assert marker_file.exists()

    # invalidate_cache should remove the cache contents
    invalidate_cache(
        cache_arg="flush_on_change",
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456"
        "789012345678901234567890abcd",
        custom_cache_dir=cache_dir,
    )

    # Verify cache was flushed (directory removed)
    assert not cache_dir.exists()


# --- CubieCacheHandler tests ---


def test_cache_handler_init_with_disabled_cache():
    """Verify CubieCacheHandler initializes with None cache when disabled."""
    handler = CubieCacheHandler(
        cache_arg=False,
        system_name="test_system",
        system_hash="abc123",
    )
    assert handler.cache is None
    assert handler.config.cache_enabled is False


def test_cache_handler_init_with_enabled_cache():
    """Verify CubieCacheHandler creates CUBIECache when cache_arg=True."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="test_system",
        system_hash="abc123",
    )
    assert handler.cache is not None
    assert isinstance(handler.cache, CUBIECache)
    assert handler.config.cache_enabled is True


def test_cache_handler_update_returns_recognized_params():
    """Verify update() returns set of recognized parameter names."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="test_system",
        system_hash="abc123",
    )
    recognized = handler.update({"cache_mode": "flush_on_change"})
    assert "cache_mode" in recognized
    assert handler.config.cache_mode == "flush_on_change"


def test_cache_handler_configured_cache_returns_none_when_disabled():
    """Verify configured_cache() returns None when cache disabled."""
    handler = CubieCacheHandler(
        cache_arg=False,
        system_name="test_system",
        system_hash="abc123",
    )
    result = handler.configured_cache("compile_settings_hash_123")
    assert result is None


def test_cache_handler_configured_cache_sets_hashes():
    """Verify configured_cache() updates system and compile hashes."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="test_system",
        system_hash="abc123",
    )
    compile_hash = "def456789012345678901234567890123456" \
                   "789012345678901234567890abcd"
    result = handler.configured_cache(compile_hash)
    assert result is not None
    assert result._compile_settings_hash == compile_hash
    assert result._system_hash == "abc123"


def test_cache_handler_flush_handles_none_cache():
    """Verify flush() does not error when cache is None."""
    handler = CubieCacheHandler(
        cache_arg=False,
        system_name="test_system",
        system_hash="abc123",
    )
    # This should not raise
    handler.flush()


def test_cache_handler_enable_cache_via_update():
    """Verify cache can be enabled via update(cache_enabled=True)."""
    handler = CubieCacheHandler(
        cache_arg=False,
        system_name="test_system",
        system_hash="abc123",
    )
    assert handler.cache is None

    recognized = handler.update({"cache_enabled": True})
    assert "cache_enabled" in recognized
    assert handler.cache is not None
    assert isinstance(handler.cache, CUBIECache)


# --- BatchSolverKernel cache integration tests ---


def test_batch_solver_kernel_extracts_system_hash_from_symbolic_ode(
    solverkernel, system
):
    """Verify BatchSolverKernel extracts fn_hash from SymbolicODE."""
    # SymbolicODE should have fn_hash attribute
    assert hasattr(system, 'fn_hash')

    # Verify cache_handler config has the system_hash from system.fn_hash
    handler_config = solverkernel.cache_handler.config
    assert handler_config.system_hash == system.fn_hash


def test_batch_solver_kernel_uses_name_from_system(solverkernel, system):
    """Verify BatchSolverKernel uses system.name for cache directory."""
    # System should have a name
    assert hasattr(system, 'name')

    # Verify cache_handler config has the system_name from system.name
    handler_config = solverkernel.cache_handler.config
    # If system.name is set, it should be used; otherwise hash prefix
    if system.name:
        assert handler_config.system_name == system.name
    else:
        # Fallback to first 12 chars of hash
        assert handler_config.system_name == system.fn_hash[:12]


def test_batch_solver_kernel_handles_none_cache_settings(
    system,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Verify BatchSolverKernel works when cache_settings=None."""
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

    # This should not raise - cache_settings defaults to None
    kernel = BatchSolverKernel(
        system,
        step_control_settings=step_controller_settings,
        algorithm_settings=algorithm_settings,
        output_settings=output_settings,
        memory_settings=memory_settings,
        loop_settings=loop_settings,
        cache=False,
        cache_settings=None,
    )
    assert kernel.cache_handler is not None
    assert kernel.cache_handler.cache is None


def test_batch_solver_kernel_update_forwards_cache_params(
    solverkernel_mutable,
):
    """Verify update(cache_mode='flush_on_change') is recognized."""
    # Initial mode should be 'hash' (default)
    initial_mode = solverkernel_mutable.cache_handler.config.cache_mode
    assert initial_mode == "hash"

    # Update cache mode
    recognized = solverkernel_mutable.update(cache_mode="flush_on_change")

    # Verify cache_mode is recognized and updated
    assert "cache_mode" in recognized
    assert (
        solverkernel_mutable.cache_handler.config.cache_mode
        == "flush_on_change"
    )


# --- Integration tests for complete cache flow ---


def test_cache_handler_uses_symbolic_ode_fn_hash(system):
    """Verify CubieCacheHandler uses fn_hash from SymbolicODE."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name=system.name,
        system_hash=system.fn_hash,
    )

    assert handler.config.system_hash == system.fn_hash
    assert handler.config.system_name == system.name

    # Verify cache path includes the hash
    if handler.cache is not None:
        path_str = str(handler.cache.cache_path)
        assert system.fn_hash in path_str or "default" in path_str


def test_solver_cache_configuration_flow(system):
    """Verify Solver correctly configures cache handler."""
    from cubie import Solver

    solver = Solver(
        system,
        algorithm="euler",
        cache=True,
        cache_mode="hash",
        max_cache_entries=5,
    )

    # Verify cache handler is configured
    assert solver.kernel.cache_handler is not None
    assert solver.kernel.cache_handler.config.cache_mode == "hash"
    assert solver.kernel.cache_handler.config.max_cache_entries == 5


def test_solver_kernel_update_cache_mode(solverkernel_mutable):
    """Verify BatchSolverKernel.update forwards cache parameters."""
    # Initial state
    initial_mode = solverkernel_mutable.cache_handler.config.cache_mode

    # Update cache mode
    recognized = solverkernel_mutable.update(cache_mode="flush_on_change")

    assert "cache_mode" in recognized
    assert (
        solverkernel_mutable.cache_handler.config.cache_mode
        == "flush_on_change"
    )
