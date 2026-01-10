"""Tests for CacheConfig class and cache configuration."""

from pathlib import Path

import pytest

from cubie.batchsolving.BatchSolverConfig import CacheConfig
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel


class TestCacheConfigFromCacheParam:
    """Tests for CacheConfig.from_cache_param factory method."""

    def test_cache_config_from_false(self):
        """Verify from_cache_param(False) returns disabled config."""
        config = CacheConfig.from_cache_param(False)
        assert config.enabled is False
        assert config.cache_path is None

    def test_cache_config_from_none(self):
        """Verify from_cache_param(None) returns disabled config."""
        config = CacheConfig.from_cache_param(None)
        assert config.enabled is False
        assert config.cache_path is None

    def test_cache_config_from_true(self):
        """Verify from_cache_param(True) returns enabled with None path."""
        config = CacheConfig.from_cache_param(True)
        assert config.enabled is True
        assert config.cache_path is None

    def test_cache_config_from_string_path(self, tmp_path):
        """Verify from_cache_param with string returns enabled with Path."""
        path_str = str(tmp_path / "cache")
        config = CacheConfig.from_cache_param(path_str)
        assert config.enabled is True
        assert config.cache_path == Path(path_str)

    def test_cache_config_from_path_object(self, tmp_path):
        """Verify from_cache_param with Path returns enabled config."""
        cache_path = tmp_path / "cache_dir"
        config = CacheConfig.from_cache_param(cache_path)
        assert config.enabled is True
        assert config.cache_path == cache_path


class TestCacheConfigProperties:
    """Tests for CacheConfig property accessors."""

    def test_cache_directory_returns_none_when_disabled(self):
        """Verify cache_directory is None when caching is disabled."""
        config = CacheConfig(enabled=False, cache_path=Path("/some/path"))
        assert config.cache_directory is None

    def test_cache_directory_returns_path_when_enabled(self, tmp_path):
        """Verify cache_directory returns path when enabled."""
        config = CacheConfig(enabled=True, cache_path=tmp_path)
        assert config.cache_directory == tmp_path

    def test_cache_directory_returns_none_when_enabled_no_path(self):
        """Verify cache_directory is None when enabled but no path set."""
        config = CacheConfig(enabled=True, cache_path=None)
        assert config.cache_directory is None

    def test_cache_config_works_without_cuda(self):
        """Verify CacheConfig can be used without CUDA intrinsics.

        This test inherently validates CUDASIM compatibility when run
        with NUMBA_ENABLE_CUDASIM=1.
        """
        config = CacheConfig.from_cache_param(True)
        # All these operations should work without CUDA
        _ = config.enabled
        _ = config.cache_path
        _ = config.cache_directory
        _ = config.values_hash


class TestCacheConfigHashing:
    """Tests for CacheConfig hashing and update behavior."""

    def test_values_hash_stable_for_equivalent_configs(self, tmp_path):
        """Verify equivalent configs produce same hash."""
        config1 = CacheConfig(enabled=True, cache_path=tmp_path)
        config2 = CacheConfig(enabled=True, cache_path=tmp_path)
        assert config1.values_hash == config2.values_hash

    def test_values_hash_differs_for_different_enabled(self):
        """Verify different enabled values produce different hashes."""
        config1 = CacheConfig(enabled=True)
        config2 = CacheConfig(enabled=False)
        assert config1.values_hash != config2.values_hash

    def test_update_recognizes_enabled_field(self):
        """Verify update() recognizes the enabled field."""
        config = CacheConfig(enabled=False)
        recognized, changed = config.update({"enabled": True})
        assert "enabled" in recognized
        assert "enabled" in changed
        assert config.enabled is True

    def test_update_recognizes_cache_path_field(self, tmp_path):
        """Verify update() recognizes the cache_path field."""
        config = CacheConfig(enabled=True, cache_path=None)
        recognized, changed = config.update({"cache_path": tmp_path})
        assert "cache_path" in recognized
        assert "cache_path" in changed
        assert config.cache_path == tmp_path


class TestBatchSolverKernelCacheConfig:
    """Tests for BatchSolverKernel cache configuration integration."""

    def test_batchsolverkernel_cache_disabled_by_default(self, system):
        """Verify cache is disabled when cache param not provided."""
        kernel = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
        )
        assert kernel.cache_enabled is False
        assert kernel.cache_config.enabled is False

    def test_batchsolverkernel_cache_enabled_with_true(self, system):
        """Verify cache is enabled when cache=True."""
        kernel = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=True,
        )
        assert kernel.cache_enabled is True
        assert kernel.cache_config.cache_path is None

    def test_batchsolverkernel_cache_enabled_with_path(
        self, system, tmp_path
    ):
        """Verify cache is enabled with custom path."""
        cache_dir = tmp_path / "kernel_cache"
        kernel = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=cache_dir,
        )
        assert kernel.cache_enabled is True
        assert kernel.cache_config.cache_path == cache_dir

    def test_cache_config_property_returns_cacheconfig(self, system):
        """Verify cache_config property returns CacheConfig instance."""
        kernel = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=True,
        )
        assert isinstance(kernel.cache_config, CacheConfig)

    def test_cache_settings_not_in_compile_hash(self, system, tmp_path):
        """Verify cache settings don't affect compile_settings hash."""
        # Create two kernels with different cache settings
        kernel1 = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=False,
        )
        kernel2 = BatchSolverKernel(
            system,
            algorithm_settings={"algorithm": "euler"},
            cache=tmp_path,
        )
        # Compile settings hash should be the same
        # (cache settings are separate from compile settings)
        assert (
            kernel1.compile_settings.values_hash
            == kernel2.compile_settings.values_hash
        )
