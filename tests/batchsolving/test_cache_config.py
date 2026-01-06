"""Tests for CacheConfig and related caching configuration."""

import os
from pathlib import Path

import pytest

from cubie.batchsolving.BatchSolverConfig import (
    BatchSolverConfig,
    CacheConfig,
)
from cubie.cubie_cache import CUBIECache


class TestCacheConfigDefaults:
    """Tests for CacheConfig default values."""

    def test_cache_config_defaults(self):
        """Verify CacheConfig has correct default values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.mode == 'hash'
        assert config.max_entries == 10
        assert config.cache_dir is None


class TestCacheConfigModeValidation:
    """Tests for CacheConfig mode validation."""

    def test_cache_config_mode_hash_valid(self):
        """Verify mode 'hash' is accepted."""
        config = CacheConfig(mode='hash')
        assert config.mode == 'hash'

    def test_cache_config_mode_flush_on_change_valid(self):
        """Verify mode 'flush_on_change' is accepted."""
        config = CacheConfig(mode='flush_on_change')
        assert config.mode == 'flush_on_change'

    def test_cache_config_mode_validation(self):
        """Verify mode only accepts 'hash' or 'flush_on_change'."""
        with pytest.raises(ValueError):
            CacheConfig(mode='invalid_mode')


class TestCacheConfigMaxEntriesValidation:
    """Tests for CacheConfig max_entries validation."""

    def test_cache_config_max_entries_zero_valid(self):
        """Verify max_entries=0 is accepted (disables eviction)."""
        config = CacheConfig(max_entries=0)
        assert config.max_entries == 0

    def test_cache_config_max_entries_positive_valid(self):
        """Verify positive max_entries is accepted."""
        config = CacheConfig(max_entries=100)
        assert config.max_entries == 100

    def test_cache_config_max_entries_validation(self):
        """Verify max_entries rejects negative values."""
        with pytest.raises(ValueError):
            CacheConfig(max_entries=-1)


class TestCacheConfigCacheDirConversion:
    """Tests for CacheConfig cache_dir conversion."""

    def test_cache_config_cache_dir_none(self):
        """Verify cache_dir None is accepted."""
        config = CacheConfig(cache_dir=None)
        assert config.cache_dir is None

    def test_cache_config_cache_dir_path(self):
        """Verify cache_dir Path is accepted."""
        path = Path('/tmp/cache')
        config = CacheConfig(cache_dir=path)
        assert config.cache_dir == path
        assert isinstance(config.cache_dir, Path)

    def test_cache_config_cache_dir_conversion(self):
        """Verify str cache_dir converts to Path."""
        config = CacheConfig(cache_dir='/tmp/cache')
        assert config.cache_dir == Path('/tmp/cache')
        assert isinstance(config.cache_dir, Path)


class TestBatchSolverConfigCacheConfig:
    """Tests for BatchSolverConfig cache_config integration."""

    def test_batch_solver_config_cache_config_field(self):
        """Verify BatchSolverConfig has cache_config field."""
        config = BatchSolverConfig()

        assert hasattr(config, 'cache_config')
        assert isinstance(config.cache_config, CacheConfig)

    def test_batch_solver_config_cache_config_default(self):
        """Verify cache_config defaults are applied."""
        config = BatchSolverConfig()

        assert config.cache_config.enabled is True
        assert config.cache_config.mode == 'hash'
        assert config.cache_config.max_entries == 10
        assert config.cache_config.cache_dir is None

    def test_batch_solver_config_with_custom_cache_config(self):
        """Verify custom CacheConfig can be provided."""
        cache_cfg = CacheConfig(
            enabled=False,
            mode='flush_on_change',
            max_entries=5,
            cache_dir='/tmp/custom',
        )
        config = BatchSolverConfig(cache_config=cache_cfg)

        assert config.cache_config.enabled is False
        assert config.cache_config.mode == 'flush_on_change'
        assert config.cache_config.max_entries == 5
        assert config.cache_config.cache_dir == Path('/tmp/custom')


class TestCachingEnabledBackwardsCompat:
    """Tests for backwards-compatible caching_enabled property."""

    def test_caching_enabled_backwards_compat(self):
        """Verify caching_enabled property returns cache_config.enabled."""
        config = BatchSolverConfig()
        assert config.caching_enabled is True

        # When cache_config.enabled is False
        config_disabled = BatchSolverConfig(
            cache_config=CacheConfig(enabled=False)
        )
        assert config_disabled.caching_enabled is False

    def test_caching_enabled_is_readonly(self):
        """Verify caching_enabled is a read-only property."""
        config = BatchSolverConfig()

        with pytest.raises(AttributeError):
            config.caching_enabled = False


class TestCUBIECacheMaxEntries:
    """Tests for CUBIECache max_entries parameter."""

    def test_cubie_cache_max_entries_stored(self, tmp_path):
        """Verify max_entries is stored on CUBIECache instance."""
        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            max_entries=5,
        )
        assert cache._max_entries == 5

    def test_cubie_cache_max_entries_default(self, tmp_path):
        """Verify max_entries defaults to 10."""
        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
        )
        assert cache._max_entries == 10


class TestEnforceCacheLimitNoEviction:
    """Tests for enforce_cache_limit when under limit."""

    def test_enforce_cache_limit_no_eviction_under_limit(self, tmp_path):
        """Verify no eviction when file count < max_entries."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 3 .nbi files (under limit of 10)
        for i in range(3):
            (cache_dir / f"cache_{i}.nbi").write_text("test")
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            max_entries=10,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # All files should remain
        assert len(list(cache_dir.glob("*.nbi"))) == 3
        assert len(list(cache_dir.glob("*.nbc"))) == 3


class TestEnforceCacheLimitEviction:
    """Tests for enforce_cache_limit eviction behavior."""

    def test_enforce_cache_limit_evicts_oldest(self, tmp_path):
        """Verify oldest files evicted when limit exceeded."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 5 .nbi files with different mtimes
        for i in range(5):
            nbi_file = cache_dir / f"cache_{i}.nbi"
            nbc_file = cache_dir / f"cache_{i}.0.nbc"
            nbi_file.write_text("test")
            nbc_file.write_text("test")
            # Set mtime in order (oldest first)
            mtime = 1000000 + i * 100
            os.utime(nbi_file, (mtime, mtime))
            os.utime(nbc_file, (mtime, mtime))

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            max_entries=3,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # Should have evicted 3 files (5 - 3 + 1 = 3) to make room
        remaining_nbi = list(cache_dir.glob("*.nbi"))
        assert len(remaining_nbi) == 2

        # The oldest files (cache_0, cache_1, cache_2) should be gone
        remaining_names = [f.stem for f in remaining_nbi]
        assert "cache_0" not in remaining_names
        assert "cache_1" not in remaining_names
        assert "cache_2" not in remaining_names
        # Newest files should remain
        assert "cache_3" in remaining_names
        assert "cache_4" in remaining_names


class TestEnforceCacheLimitDisabled:
    """Tests for enforce_cache_limit when disabled."""

    def test_enforce_cache_limit_zero_disables(self, tmp_path):
        """Verify max_entries=0 disables eviction."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 10 .nbi files
        for i in range(10):
            (cache_dir / f"cache_{i}.nbi").write_text("test")
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            max_entries=0,  # Disable eviction
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # All files should remain (eviction disabled)
        assert len(list(cache_dir.glob("*.nbi"))) == 10
        assert len(list(cache_dir.glob("*.nbc"))) == 10


class TestEnforceCacheLimitPairs:
    """Tests for enforce_cache_limit .nbi/.nbc pairing."""

    def test_enforce_cache_limit_pairs_nbi_nbc(self, tmp_path):
        """Verify .nbi and .nbc files evicted together."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 3 .nbi files with multiple .nbc files each
        for i in range(3):
            nbi_file = cache_dir / f"cache_{i}.nbi"
            nbi_file.write_text("test")
            # Each .nbi has multiple .nbc files
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")
            (cache_dir / f"cache_{i}.1.nbc").write_text("test")
            # Set mtime in order
            mtime = 1000000 + i * 100
            os.utime(nbi_file, (mtime, mtime))

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            max_entries=2,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # Should have evicted 2 entries (3 - 2 + 1 = 2)
        remaining_nbi = list(cache_dir.glob("*.nbi"))
        assert len(remaining_nbi) == 1

        # The .nbc files for evicted entries should also be gone
        remaining_nbc = list(cache_dir.glob("*.nbc"))
        assert len(remaining_nbc) == 2  # Only cache_2's .nbc files remain

        # Verify cache_0 and cache_1 .nbc files are gone
        remaining_nbc_names = [f.name for f in remaining_nbc]
        assert "cache_0.0.nbc" not in remaining_nbc_names
        assert "cache_0.1.nbc" not in remaining_nbc_names
        assert "cache_1.0.nbc" not in remaining_nbc_names
        assert "cache_1.1.nbc" not in remaining_nbc_names


class TestCUBIECacheModeStored:
    """Tests for CUBIECache mode parameter."""

    def test_cubie_cache_mode_stored(self, tmp_path):
        """Verify mode is stored on CUBIECache instance."""
        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            mode='flush_on_change',
        )
        assert cache._mode == 'flush_on_change'

    def test_cubie_cache_mode_default(self, tmp_path):
        """Verify mode defaults to 'hash'."""
        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
        )
        assert cache._mode == 'hash'


class TestFlushCacheRemovesFiles:
    """Tests for flush_cache file removal."""

    def test_flush_cache_removes_files(self, tmp_path):
        """Verify flush_cache removes all cache files."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create some cache files
        (cache_dir / "cache_0.nbi").write_text("test")
        (cache_dir / "cache_0.0.nbc").write_text("test")
        (cache_dir / "cache_1.nbi").write_text("test")
        (cache_dir / "cache_1.0.nbc").write_text("test")

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.flush_cache()

        # Directory should exist but be empty
        assert cache_dir.exists()
        assert len(list(cache_dir.glob("*"))) == 0


class TestFlushCacheRecreatesDirectory:
    """Tests for flush_cache directory recreation."""

    def test_flush_cache_recreates_directory(self, tmp_path):
        """Verify flush_cache creates empty directory after removal."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Add a file
        (cache_dir / "test.nbi").write_text("test")

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
        )
        cache._cache_path = str(cache_dir)

        cache.flush_cache()

        # Directory should be recreated and empty
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert len(list(cache_dir.iterdir())) == 0

    def test_flush_cache_handles_missing_directory(self, tmp_path):
        """Verify flush_cache handles non-existent directory."""
        cache_dir = tmp_path / "nonexistent" / "cache"

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
        )
        cache._cache_path = str(cache_dir)

        # Should not raise an error
        cache.flush_cache()

        # Directory should now exist
        assert cache_dir.exists()


class TestCustomCacheDir:
    """Tests for custom_cache_dir parameter."""

    def test_custom_cache_dir_used(self, tmp_path):
        """Verify custom_cache_dir overrides default path."""
        custom_dir = tmp_path / "my_custom_cache"

        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            custom_cache_dir=custom_dir,
        )

        assert cache._cache_path == str(custom_dir)

    def test_custom_cache_dir_none_uses_default(self, tmp_path):
        """Verify None cache_dir uses GENERATED_DIR path."""
        compile_settings = BatchSolverConfig()
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            compile_settings=compile_settings,
            custom_cache_dir=None,
        )

        # Should use default path based on GENERATED_DIR
        assert "test_system" in cache._cache_path
        assert cache._cache_path.endswith("cache")


class TestParseCacheParam:
    """Tests for BatchSolverKernel._parse_cache_param."""

    def test_parse_cache_param_true(self, simple_system):
        """Verify cache=True creates enabled CacheConfig in hash mode."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )

        assert kernel.cache_config.enabled is True
        assert kernel.cache_config.mode == 'hash'
        assert kernel.cache_config.cache_dir is None

    def test_parse_cache_param_false(self, simple_system):
        """Verify cache=False creates disabled CacheConfig."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=False
        )

        assert kernel.cache_config.enabled is False

    def test_parse_cache_param_flush_on_change(self, simple_system):
        """Verify cache='flush_on_change' sets flush mode."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache='flush_on_change',
        )

        assert kernel.cache_config.enabled is True
        assert kernel.cache_config.mode == 'flush_on_change'

    def test_parse_cache_param_path(self, simple_system, tmp_path):
        """Verify cache=Path sets custom cache_dir."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        custom_path = tmp_path / "custom_cache"
        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=custom_path,
        )

        assert kernel.cache_config.enabled is True
        assert kernel.cache_config.mode == 'hash'
        assert kernel.cache_config.cache_dir == custom_path

    def test_parse_cache_param_string_path(self, simple_system, tmp_path):
        """Verify cache=string path sets custom cache_dir."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        custom_path = str(tmp_path / "custom_cache")
        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=custom_path,
        )

        assert kernel.cache_config.enabled is True
        assert kernel.cache_config.cache_dir == Path(custom_path)


class TestKernelCacheConfigProperty:
    """Tests for BatchSolverKernel.cache_config property."""

    def test_kernel_cache_config_property(self, simple_system):
        """Verify cache_config property returns correct object."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )

        cache_config = kernel.cache_config
        assert isinstance(cache_config, CacheConfig)
        assert cache_config.enabled is True

    def test_kernel_cache_config_matches_compile_settings(self, simple_system):
        """Verify cache_config property equals compile_settings.cache_config."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache='flush_on_change',
        )

        assert kernel.cache_config is kernel.compile_settings.cache_config
        assert kernel.cache_config.mode == 'flush_on_change'


class TestSetCacheDir:
    """Tests for BatchSolverKernel.set_cache_dir method."""

    def test_set_cache_dir_updates_config(self, simple_system, tmp_path):
        """Verify set_cache_dir updates cache_config.cache_dir."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )
        new_path = tmp_path / "new_cache_dir"

        kernel.set_cache_dir(new_path)

        assert kernel.cache_config.cache_dir == new_path

    def test_set_cache_dir_invalidates_cache(self, simple_system, tmp_path):
        """Verify set_cache_dir calls _invalidate_cache."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )

        # Manually set cache_valid to True to verify it gets invalidated
        kernel._cache_valid = True

        new_path = tmp_path / "new_cache_dir"
        kernel.set_cache_dir(new_path)

        # Cache should be invalidated (cleared) by set_cache_dir
        assert kernel._cache_valid is False

    def test_set_cache_dir_accepts_string(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts string path."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )
        new_path_str = str(tmp_path / "string_cache_dir")

        kernel.set_cache_dir(new_path_str)

        assert kernel.cache_config.cache_dir == Path(new_path_str)
        assert isinstance(kernel.cache_config.cache_dir, Path)

    def test_set_cache_dir_accepts_path(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts Path object."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system, algorithm_settings={"algorithm": "euler"}, cache=True
        )
        new_path = tmp_path / "path_cache_dir"

        kernel.set_cache_dir(new_path)

        assert kernel.cache_config.cache_dir == new_path
        assert isinstance(kernel.cache_config.cache_dir, Path)


@pytest.fixture(scope="function")
def simple_system():
    """Create a minimal ODE system for cache parameter testing."""
    from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

    equations = ["dx0 = -x0"]
    states = {"x0": 1.0}

    return create_ODE_system(
        dxdt=equations,
        states=states,
        name="simple_cache_test_system",
    )


class TestSolverCacheParam:
    """Tests for Solver cache parameter pass-through."""

    def test_solver_cache_param_passed_to_kernel(self, simple_system, tmp_path):
        """Verify Solver passes cache param to kernel."""
        from cubie.batchsolving.solver import Solver

        custom_path = tmp_path / "solver_cache"
        solver = Solver(simple_system, cache=custom_path)

        assert solver.kernel.cache_config.enabled is True
        assert solver.kernel.cache_config.cache_dir == custom_path

    def test_solver_cache_true_default(self, simple_system):
        """Verify cache=True is the default."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system)

        assert solver.kernel.cache_config.enabled is True
        assert solver.kernel.cache_config.mode == 'hash'

    def test_solver_cache_false(self, simple_system):
        """Verify cache=False disables caching."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=False)

        assert solver.kernel.cache_config.enabled is False

    def test_solver_cache_flush_on_change(self, simple_system):
        """Verify cache='flush_on_change' sets mode."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache='flush_on_change')

        assert solver.kernel.cache_config.enabled is True
        assert solver.kernel.cache_config.mode == 'flush_on_change'


class TestSolverCacheProperties:
    """Tests for Solver cache-related properties."""

    def test_solver_cache_enabled_property(self, simple_system):
        """Verify cache_enabled returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        assert solver.cache_enabled is True

        solver_disabled = Solver(simple_system, cache=False)
        assert solver_disabled.cache_enabled is False

    def test_solver_cache_mode_property(self, simple_system):
        """Verify cache_mode returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        assert solver.cache_mode == 'hash'

        solver_flush = Solver(simple_system, cache='flush_on_change')
        assert solver_flush.cache_mode == 'flush_on_change'

    def test_solver_cache_dir_property(self, simple_system, tmp_path):
        """Verify cache_dir returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        assert solver.cache_dir is None

        custom_path = tmp_path / "cache_dir_test"
        solver_custom = Solver(simple_system, cache=custom_path)
        assert solver_custom.cache_dir == custom_path


class TestSolverSetCacheDir:
    """Tests for Solver.set_cache_dir method."""

    def test_solver_set_cache_dir_delegates(self, simple_system, tmp_path):
        """Verify set_cache_dir calls kernel method."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        new_path = tmp_path / "new_cache"

        solver.set_cache_dir(new_path)

        assert solver.cache_dir == new_path
        assert solver.kernel.cache_config.cache_dir == new_path

    def test_solver_set_cache_dir_string(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts string."""
        from cubie.batchsolving.solver import Solver
        from pathlib import Path

        solver = Solver(simple_system, cache=True)
        new_path_str = str(tmp_path / "string_cache")

        solver.set_cache_dir(new_path_str)

        assert solver.cache_dir == Path(new_path_str)

    def test_solver_set_cache_dir_path(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts Path object."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        new_path = tmp_path / "path_cache"

        solver.set_cache_dir(new_path)

        assert solver.cache_dir == new_path
