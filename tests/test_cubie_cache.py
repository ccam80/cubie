"""Tests for cubie_cache module."""

from pathlib import Path

import pytest
from numpy import array, float32

from attrs import define, field

from cubie.cuda_backend import IS_MLIR
from cubie.cubie_cache import (
    CUBIECacheLocator,
    CUBIECacheImpl,
    CUBIECache,
    CubieCacheHandler,
    CacheConfig,
    ALL_CACHE_PARAMETERS,
    _compiler_environment_entries,
    environment_hash,
)
from cubie._utils import package_source_hash


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
    assert path.endswith("CUDA_cache_abc123")


def test_cache_locator_get_source_stamp():
    """Verify source stamp combines system_hash and environment hash."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert locator.get_source_stamp() == f"abc123-{environment_hash()}"


def test_environment_hash_is_stable_hex_digest():
    """Verify the environment hash is a deterministic sha256 digest."""
    digest = environment_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert environment_hash() == digest


def test_environment_hash_uses_compiler_packages_only():
    """The cache stamp ignores unrelated runner packages."""
    entries = _compiler_environment_entries()
    assert any(entry.startswith("numpy==") for entry in entries)
    assert any(entry.startswith("python==") for entry in entries)
    assert not any(entry.startswith("pytest==") for entry in entries)


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
    """Verify cache path follows generated/<system_name>/CUDA_cache_<hash[
    :8]>."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "abc123" in path
    assert path.endswith("CUDA_cache_abc123")


# --- ALL_CACHE_PARAMETERS and CacheConfig tests ---


def test_all_cache_parameters_contains_expected_keys():
    """Verify ALL_CACHE_PARAMETERS contains expected cache parameter names."""
    expected_keys = {
        "cache_enabled",
        "cache_mode",
        "max_cache_entries",
        "cache_dir",
    }
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
    """Verify check_cachable accepts what the backend can serialize.

    numba-cuda kernels are always cachable; the MLIR compile-result
    scheme inspects targetoptions and refuses results that link
    external files.
    """
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    if IS_MLIR:

        class LinkFreeResult:
            metadata = {"targetoptions": {"link": []}}

        class LinkedResult:
            metadata = {"targetoptions": {"link": ["kernels.cu"]}}

        assert impl.check_cachable(LinkFreeResult()) is True
        with pytest.raises(RuntimeError):
            impl.check_cachable(LinkedResult())
    else:
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
    """The index key includes CuBIE and launch settings."""
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

    # Key is (sig, magic_tuple, system_hash, config_hash, package_hash)
    assert len(key) == 5
    assert key[0] == sig
    assert key[1] == ("magic", "tuple")
    assert key[2] == "abc123"
    assert key[3] == config_hash
    assert key[4] == package_source_hash()

    cache._launch_config_key = (1, 2, 3)
    launch_key = cache._index_key(sig, codegen)
    assert launch_key[:-1] == key
    assert launch_key[-1] == ("launch_config", (1, 2, 3))


def test_cubie_cache_path(monkeypatch):
    """Verify cache_path includes system_hash subdirectory."""
    monkeypatch.delenv("CUBIE_KERNEL_CACHE_DIR", raising=False)
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


def test_kernel_cache_dir_env_relocates(tmp_path, monkeypatch):
    """An explicit cache directory overrides the environment."""
    env_dir = tmp_path / "artifact"
    monkeypatch.setenv("CUBIE_KERNEL_CACHE_DIR", str(env_dir))
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )
    assert str(cache.cache_path).startswith(str(env_dir))
    explicit_dir = tmp_path / "explicit"
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
        custom_cache_dir=explicit_dir,
    )
    assert str(cache.cache_path).startswith(str(explicit_dir))


def test_max_cache_entries_env_default(monkeypatch):
    """An explicit cache limit overrides the environment."""
    monkeypatch.setenv("CUBIE_MAX_CACHE_ENTRIES", "0")
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )
    assert cache._max_entries == 0
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
        max_entries=3,
    )
    assert cache._max_entries == 3


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
    assert locator.get_source_stamp() == f"abc123-{environment_hash()}"
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
    result = handler.configured_cache("abc123", "compile_settings_hash_123")
    assert result is None


def test_cache_handler_configured_cache_sets_hashes():
    """Verify configured_cache() updates system and compile hashes."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="test_system",
        system_hash="abc123",
    )
    compile_hash = (
        "def456789012345678901234567890123456789012345678901234567890abcd"
    )
    result = handler.configured_cache("abc123", compile_hash)
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
    assert hasattr(system, "fn_hash")

    # Verify cache_handler config has the system_hash from system.fn_hash
    handler_config = solverkernel.cache_handler.config
    assert handler_config.system_hash == system.fn_hash


def test_batch_solver_kernel_uses_name_from_system(solverkernel, system):
    """Verify BatchSolverKernel uses system.name for cache directory."""
    # System should have a name
    assert hasattr(system, "name")

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
    path_str = str(handler.cache.cache_path)
    assert system.fn_hash[:8] in path_str


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
    # Update cache mode
    recognized = solverkernel_mutable.update(cache_mode="flush_on_change")

    assert "cache_mode" in recognized
    assert (
        solverkernel_mutable.cache_handler.config.cache_mode
        == "flush_on_change"
    )


# --- CUBIECache.enforce_cache_limit eviction-failure tests ---


def test_enforce_cache_limit_warns_and_continues_on_unlink_failure(
    tmp_path,
):
    """Verify enforce_cache_limit warns when an .nbi entry cannot be
    removed and silently continues past a matching .nbc removal
    failure, while still evicting entries that can be removed.

    A directory cannot be removed via Path.unlink(), so entries are
    represented as directories to force real OSError conditions
    without mocking.
    """
    cache = CUBIECache(
        system_name="evict_test",
        system_hash="evicthash1",
        config_hash="a" * 64,
        max_entries=1,
        custom_cache_dir=tmp_path,
    )
    cache_path = Path(cache.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    # "blocked" entries are directories: unlink() raises OSError for both
    # the .nbi and its matching .nbc file.
    blocked_nbi = cache_path / "blocked.nbi"
    blocked_nbi.mkdir()
    blocked_nbc = cache_path / "blocked.0.nbc"
    blocked_nbc.mkdir()

    # "removable" entries are real files: unlink() succeeds normally.
    removable_nbi = cache_path / "removable.nbi"
    removable_nbi.write_bytes(b"index")
    removable_nbc = cache_path / "removable.0.nbc"
    removable_nbc.write_bytes(b"data")

    with pytest.warns(RuntimeWarning, match="Failed to remove cache file"):
        cache.enforce_cache_limit()

    # The removable entry was evicted; the blocked one remains because
    # its unlink() failed (and the failure was reported via warn()).
    assert not removable_nbi.exists()
    assert not removable_nbc.exists()
    assert blocked_nbi.exists()
    assert blocked_nbc.exists()


def test_flush_cache_handles_rmtree_failure_silently(tmp_path):
    """Verify flush_cache does not raise when rmtree fails to remove
    the cache directory, e.g. because a file inside it is still open.

    Uses a real open file handle to force a genuine OSError from
    shutil.rmtree rather than mocking.
    """
    cache = CUBIECache(
        system_name="flush_test",
        system_hash="flushhash1",
        config_hash="b" * 64,
        custom_cache_dir=tmp_path,
    )
    cache_path = Path(cache.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    locked_file = cache_path / "locked.nbi"
    locked_file.write_bytes(b"data")

    handle = open(locked_file, "rb")
    try:
        # Should not raise even though rmtree cannot remove the open file.
        cache.flush_cache()
    finally:
        handle.close()

    # flush_cache recreates the directory regardless of rmtree's outcome.
    assert cache_path.exists()


# --- CubieCacheHandler additional coverage ---


def test_cache_handler_update_empty_returns_empty_set():
    """Verify update() with no arguments returns an empty set."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="empty_update",
        system_hash="h1",
    )
    recognized = handler.update()
    assert recognized == set()


def test_cache_handler_disable_cache_via_update():
    """Verify cache is torn down when cache_enabled is set to False."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="disable_test",
        system_hash="h2",
    )
    assert handler.cache is not None

    recognized = handler.update({"cache_enabled": False})
    assert "cache_enabled" in recognized
    assert handler.cache is None


def test_cache_handler_update_unrecognized_raises_keyerror():
    """Verify update() raises KeyError for unrecognized parameters."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="keyerror_test",
        system_hash="h3",
    )
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        handler.update({"not_a_real_param": 1})


def test_cache_handler_update_unrecognized_silent_no_raise():
    """Verify update() with silent=True suppresses the KeyError."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="silent_test",
        system_hash="h4",
    )
    recognized = handler.update({"not_a_real_param": 1}, silent=True)
    assert "not_a_real_param" not in recognized


def test_cache_handler_configured_cache_updates_system_hash_in_config():
    """Verify configured_cache() writes a changed system_hash back into
    the handler's config."""
    handler = CubieCacheHandler(
        cache_arg=True,
        system_name="hash_update_test",
        system_hash="old_hash",
    )
    assert handler.config.system_hash == "old_hash"

    handler.configured_cache("new_hash", "c" * 64)
    assert handler.config.system_hash == "new_hash"


def test_cache_handler_cache_enabled_property():
    """Verify the cache_enabled property reflects the config."""
    handler_on = CubieCacheHandler(
        cache_arg=True,
        system_name="enabled_prop",
        system_hash="h5",
    )
    assert handler_on.cache_enabled is True

    handler_off = CubieCacheHandler(
        cache_arg=False,
        system_name="disabled_prop",
        system_hash="h6",
    )
    assert handler_off.cache_enabled is False
