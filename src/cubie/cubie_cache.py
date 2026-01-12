"""File-based caching infrastructure for CuBIE compiled kernels.

Provides cache classes that persist compiled CUDA kernels to disk,
enabling faster startup on subsequent runs with identical settings.
Cache files are stored in
``generated/<system_name>/CUDA_cache_{system_hash[:8]}`` within the
configured GENERATED_DIR or in  ``custom_dir/CUDA_cache_{system_hash[:8]}``
if a custom directory is provided as an argument to "cache".

Notes
-----
This module depends on numba-cuda internal classes and may require
updates when numba-cuda versions change. Where cache modules are not
imported by Numba in CUDASIM mode, this module relies on vendored versions
in the `src/vendored` directory. This allows caching to function in CUDASIM
mode for testing purposes, even though no compiled kernels are produced.
"""

from shutil import rmtree
from warnings import warn
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from attrs import field, validators as val, define, converters

from cubie.CUDAFactory import _CubieConfigBase
from cubie._utils import getype_validator, build_config
from numba.cuda.core.caching import (  # noqa: F401
    _CacheLocator,  # noqa: F401
    CacheImpl,  # noqa: F401
    IndexDataCacheFile,  # noqa: F401
)

from cubie.cuda_simsafe import is_cudasim_enabled
from cubie.vendored.numba_cuda_cache import CUDACache
from cubie.odesystems.symbolic.odefile import GENERATED_DIR


ALL_CACHE_PARAMETERS: Set[str] = {
    "cache_enabled",
    "cache_mode",
    "max_cache_entries",
    "cache_dir",
}


class CUBIECacheLocator(_CacheLocator):
    """Locate cache files in CuBIE's generated directory structure.

    Directs cache files to ``generated/<system_name>/CUDA_cache/`` instead
    of the default ``__pycache__`` location used by numba.

    Parameters
    ----------
    system_name
        Name of the ODE system for directory organization.
    system_hash
        Hash representing the ODE system definition for freshness.
    compile_settings_hash
        Hash of compile settings for disambiguation.
    custom_cache_dir
        Optional custom cache directory. Overrides default location.
    """

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        self._system_name = system_name
        self._system_hash = system_hash
        self._compile_settings_hash = compile_settings_hash

        if custom_cache_dir is None:
            cache_root = GENERATED_DIR / system_name
        else:
            cache_root = Path(custom_cache_dir)

        self._cache_root_dir = cache_root
        hash_dir = f"CUDA_cache_{system_hash[:8]}"
        self._cache_path = cache_root / hash_dir

    def get_cache_path(self) -> str:
        """Return the directory where cache files are stored.

        Returns
        -------
        str
            Absolute path to the cache directory.
        """
        return str(self._cache_path)

    def get_source_stamp(self) -> str:
        """Return a stamp representing source freshness.

        Returns
        -------
        str
            The system hash acts as the freshness indicator.
        """
        return self._system_hash

    def get_disambiguator(self) -> str:
        """Return a string to disambiguate similar functions.

        Returns
        -------
        str
            First 16 characters of compile_settings_hash.
        """
        return self._compile_settings_hash[:16]

    def set_compile_settings_hash(self, compile_settings_hash: str) -> None:
        """Update the compile settings hash.

        Parameters
        ----------
        compile_settings_hash
            New compile settings hash to set.
        """
        self._compile_settings_hash = compile_settings_hash

    def set_system_hash(self, system_hash: str) -> None:
        """Update the system hash and refresh cache path.

        Parameters
        ----------
        system_hash
            New system hash to set.
        """
        self._system_hash = system_hash
        hash_dir = f"CUDA_cache_{system_hash[:8]}"
        self._cache_path = self._cache_root_dir / hash_dir

    @classmethod
    def from_function(cls, py_func, py_file):
        """Not used - CuBIE creates locators directly.

        Raises
        ------
        NotImplementedError
            This locator does not use the from_function pattern.
        """
        raise NotImplementedError(
            "CUBIECacheLocator requires explicit system info"
        )


class CUBIECacheImpl(CacheImpl):
    """Serialization logic for CuBIE compiled kernels.

    Delegates actual serialization to numba's built-in _Kernel methods
    while using CuBIE-specific cache locator for file paths.

    Parameters
    ----------
    system_name
        Name of the ODE system.
    system_hash
        Hash representing the ODE system definition.
    compile_settings_hash
        Hash of compile settings for cache key.
    custom_cache_dir
        Optional custom cache directory.
    """

    # Override locator classes to use only CuBIE locator
    _locator_classes = []

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        # Create CUBIECacheLocator directly
        self._locator = CUBIECacheLocator(
            system_name,
            system_hash,
            compile_settings_hash,
            custom_cache_dir=custom_cache_dir,
        )
        disambiguator = self._locator.get_disambiguator()
        self._filename_base = f"{system_name}-{disambiguator}"

    @property
    def locator(self) -> CUBIECacheLocator:
        """Return the cache locator instance."""
        return self._locator

    @property
    def filename_base(self) -> str:
        """Return base filename for cache files."""
        return self._filename_base

    def set_hashes(
        self, system_hash: Optional[str], compile_settings_hash: Optional[str]
    ) -> None:
        """Update system and compile settings hashes in locator.

        Parameters
        ----------
        system_hash
            New system hash to set or None to leave unchanged.
        compile_settings_hash
            New compile settings hash to set or None to leave unchanged.
        """
        if system_hash is not None:
            self._locator.set_system_hash(system_hash)
        if compile_settings_hash is not None:
            self._locator.set_compile_settings_hash(compile_settings_hash)

    def reduce(self, kernel) -> dict:
        """Reduce kernel to serializable form.

        Parameters
        ----------
        kernel
            Compiled CUDA kernel with _reduce_states method.

        Returns
        -------
        dict
            Serializable state dictionary.
        """
        if not is_cudasim_enabled():
            return kernel._reduce_states()
        else:
            raise RuntimeError(
                "CUBIECacheImpl.reduce() was called inside "
                "CUDASIM mode, indicating a cache miss when "
                "there are no compiled kernels available. This "
                "indicates a config error; it should not be reachable if "
                "CUDASIM mode was properly enabled."
            )

    def rebuild(self, target_context, payload: dict):
        """Rebuild kernel from cached payload.

        Parameters
        ----------
        target_context
            CUDA target context for kernel reconstruction.
        payload
            Serialized kernel state from reduce().

        Returns
        -------
        _Kernel
            Reconstructed CUDA kernel.
        """
        if not is_cudasim_enabled():
            from numba.cuda.dispatcher import _Kernel

            return _Kernel._rebuild(**payload)
        else:
            raise RuntimeError(
                "CUBIECacheImpl.rebuild() was called inside "
                "CUDASIM mode, indicating a cache hit when "
                "there are no compiled kernels available. This "
                "indicates a config error; it should not be reachable if "
                "CUDASIM mode was properly enabled."
            )

    def check_cachable(self, data) -> bool:
        """Check if the data is cachable.

        CUDA kernels are always cachable.

        Returns
        -------
        bool
            Always True for CUDA kernels.
        """
        return True


class CUBIECache(CUDACache):
    """File-based cache for CuBIE compiled kernels.

    Coordinates loading and saving of cached kernels, incorporating
    ODE system hash and compile settings hash into cache keys.

    Parameters
    ----------
    system_name
        Name of the ODE system.
    system_hash
        Hash representing the ODE system definition.
    config_hash
        Pre-computed hash of the compile settings.
    max_entries
        Maximum number of cache entries before LRU eviction.
        Set to 0 to disable eviction.
    mode
        Caching mode: 'hash' for content-addressed caching,
        'flush_on_change' to clear cache when settings change.
    custom_cache_dir
        Optional custom cache directory. Overrides default location.

    Notes
    -----
    Unlike the base Cache class, this does not use py_func for
    initialization. Instead, system info is passed directly.
    """

    _impl_class = CUBIECacheImpl

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        config_hash: str,
        max_entries: int = 10,
        mode: str = "hash",
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize CUBIECache with system and compile info.

        Note: Does not call inherited init using __super__(), absorbs the
        responsibilities directly due to  a different set of config parameters.
        """

        self._system_name = system_name
        self._system_hash = system_hash

        self._compile_settings_hash = config_hash

        self._name = f"CUBIECache({system_name})"
        self._max_entries = max_entries
        self._mode = mode

        self._impl = CUBIECacheImpl(
            system_name,
            system_hash,
            self._compile_settings_hash,
            custom_cache_dir=custom_cache_dir,
        )
        self._cache_path = self._impl.locator.get_cache_path()

        source_stamp = self._impl.locator.get_source_stamp()
        filename_base = self._impl.filename_base
        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=filename_base,
            source_stamp=source_stamp,
        )
        self.enable()

    def _index_key(self, sig, codegen):
        """Compute cache key including CuBIE-specific hashes.

        Parameters
        ----------
        sig
            Function signature tuple.
        codegen
            CUDA codegen object with magic_tuple().

        Returns
        -------
        tuple
            Composite cache key.
        """
        return (
            sig,
            codegen.magic_tuple(),
            self._system_hash,
            self._compile_settings_hash,
        )

    def enforce_cache_limit(self) -> None:
        """Evict oldest cache entries if count exceeds max_entries.

        Uses filesystem mtime for LRU ordering. Evicts .nbi/.nbc
        file pairs together.
        """
        if self._max_entries == 0:
            return  # Eviction disabled

        cache_path = Path(self._cache_path)
        if not cache_path.exists():
            return

        # Find all .nbi files (index files)
        nbi_files = list(cache_path.glob("*.nbi"))
        if len(nbi_files) < self._max_entries:
            return

        # Sort by mtime (oldest first)
        nbi_files.sort(key=lambda f: f.stat().st_mtime)

        files_to_remove = len(nbi_files) - self._max_entries + 1
        for nbi_file in nbi_files[:files_to_remove]:
            base = nbi_file.stem
            # Remove .nbi file
            try:
                nbi_file.unlink()
            except OSError as e:
                # Log warning but continue - file may be locked or deleted
                warn(
                    f"Failed to remove cache file {nbi_file}: {e}",
                    RuntimeWarning,
                )
            # Remove associated .nbc files (may be multiple)
            for nbc_file in cache_path.glob(f"{base}.*.nbc"):
                try:
                    nbc_file.unlink()
                except OSError:
                    # Silently continue - .nbc cleanup is best-effort
                    pass

    def save_overload(self, sig, data):
        """Save kernel to cache, enforcing entry limit first.

        Parameters
        ----------
        sig
            Function signature.
        data
            Kernel data to cache.
        """
        self.enforce_cache_limit()
        super().save_overload(sig, data)

    def flush_cache(self) -> None:
        """Delete all cache files in the cache directory.

        Removes all .nbi and .nbc files, then recreates an empty
        cache directory.
        """
        cache_path = Path(self._cache_path)
        if cache_path.exists():
            try:
                rmtree(cache_path)
            except OSError:
                # Another thread may have gotten there first if concurrent.
                # If so, just continue.
                pass
        cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        """Return the cache directory path."""
        return Path(self._cache_path)

    def update_from_config(self, config) -> None:
        """Update cache settings from CacheConfig.

        Parameters
        ----------
        config
            CacheConfig instance with updated settings.
        """
        self._max_entries = config.max_cache_entries
        self._mode = config.cache_mode
        self._system_hash = config.system_hash

        # Note: Changing cache_dir requires recreating the cache.
        current_locator_path = self._impl.locator.get_cache_path()
        config_root = (
            config.cache_dir
            if config.cache_dir
            else GENERATED_DIR / config.system_name
        )
        config_specified = (
            Path(config_root) / f"CUDA_cache_{config.system_hash[:8]}"
        )

        if config_specified != current_locator_path:
            # Recreate impl with new cache directory
            self._impl = CUBIECacheImpl(
                self._system_name,
                config.system_hash,
                self._compile_settings_hash,
                custom_cache_dir=config.cache_dir,
            )
            self._cache_path = self._impl.locator.get_cache_path()
            self._cache_file = IndexDataCacheFile(
                cache_path=self._cache_path,
                filename_base=self._impl.filename_base,
                source_stamp=self._impl.locator.get_source_stamp(),
            )

    def set_hashes(
        self,
        system_hash: Optional[str] = None,
        compile_settings_hash: Optional[str] = None,
    ) -> None:
        """Update system and compile settings hashes.

        Parameters
        ----------
        system_hash
            New system hash to set.
        compile_settings_hash
            New compile settings hash to set.
        """
        if system_hash is not None:
            self._system_hash = system_hash
        if compile_settings_hash is not None:
            self._compile_settings_hash = compile_settings_hash
        self._impl.set_hashes(system_hash, compile_settings_hash)


@define
class CacheConfig(_CubieConfigBase):
    """Configuration for file-based kernel caching.

    Parameters
    ----------
    cache_enabled
        Whether file-based caching is enabled.
    cache_mode
        Caching mode: 'hash' for content-addressed caching,
        'flush_on_change' to clear cache when settings change.
    max_cache_entries
        Maximum number of cache entries before LRU eviction.
        Set to 0 to disable eviction.
    cache_dir
        Custom cache directory. None uses default location.
    system_hash
        Hash representing the ODE system definition. This should persist
        over multiple sessions.
    system_name
        Name of the ODE system for directory organization.
    """

    cache_enabled: bool = field(
        default=False,
        validator=val.instance_of(bool),
    )
    cache_mode: str = field(
        default="hash",
        validator=val.in_(("hash", "flush_on_change")),
    )
    max_cache_entries: int = field(
        default=10,
        validator=getype_validator(int, 0),
    )
    cache_dir: Optional[Path] = field(
        default=None,
        validator=val.optional(val.instance_of((str, Path))),
        converter=converters.optional(Path),
    )
    system_hash: str = field(
        default="",
        validator=val.instance_of(str),
    )
    system_name: str = field(
        default="",
        validator=val.instance_of(str),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @classmethod
    def params_from_user_kwarg(cls, cache_arg: Union[bool, str, Path]):
        """Parse single-entry "cache" argument from solver interface.

        Parameters
        ----------
        cache_arg
            Cache configuration:
            - True: Enable caching with default path
            - False or None: Disable caching
            - str or Path: Enable caching at specified path

        Returns
        -------
        Dict[str, Any]
            Configured cache params.
        """
        cache_enabled = cache_arg not in [False, None]
        cache_mode = "hash"
        cache_path = None
        if isinstance(cache_arg, str):
            if cache_arg == "flush_on_change":
                cache_mode = "flush_on_change"
            else:
                cache_path = Path(cache_arg)
        elif isinstance(cache_arg, Path):
            cache_path = cache_arg

        return {
            "cache_enabled": cache_enabled,
            "cache_dir": cache_path,
            "cache_mode": cache_mode,
        }


class CubieCacheHandler:
    """Handler for managing CuBIE kernel cache."""

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        cache_arg: Union[bool, str, Path] = None,
        **kwargs,
    ) -> None:
        # Convert single cache arg into cache_enabled, path kwargs
        config_params = CacheConfig.params_from_user_kwarg(cache_arg)
        # Let user kwargs override default config_params
        config_params.update(kwargs)

        # Build CacheConfig using build_config utility
        _config = build_config(
            CacheConfig,
            {"system_name": system_name, "system_hash": system_hash},
            **config_params,
        )
        self.config = _config

        # Create cache if enabled
        if _config.cache_enabled:
            self._cache = CUBIECache(
                system_name=system_name,
                system_hash=system_hash,
                config_hash="UNINITIALIZED",  # Set at run time via configured_cache
                max_entries=_config.max_cache_entries,
                mode=_config.cache_mode,
                custom_cache_dir=_config.cache_dir,
            )
        else:
            self._cache = None

    @property
    def cache(self) -> Optional[CUBIECache]:
        """Return the managed CUBIECache instance."""
        return self._cache

    def flush(self) -> None:
        """Flush the managed cache."""
        if self._cache is not None:
            self._cache.flush_cache()

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update cache configuration and recreate cache if needed.

        Parameters
        ----------
        updates_dict
            Dictionary of configuration updates.
        silent
            Suppress errors for unrecognized parameters.
        **kwargs
            Additional configuration overrides.

        Returns
        -------
        Set[str]
            Set of recognized parameter names.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)

        if not updates_dict:
            return set()

        recognized, changed = self.config.update(updates_dict)

        # Update cache if it exists and settings changed
        if self._cache is not None and changed:
            self._cache.update_from_config(self.config)

        # Handle cache being enabled via update
        if "cache_enabled" in changed:
            if self.config.cache_enabled:
                self._cache = CUBIECache(
                    system_name=self.config.system_name,
                    system_hash=self.config.system_hash,
                    config_hash="UNINITIALIZED",
                    max_entries=self.config.max_cache_entries,
                    mode=self.config.cache_mode,
                    custom_cache_dir=self.config.cache_dir,
                )
            else:
                self._cache = None

        unrecognized = set(updates_dict.keys()) - recognized

        if unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {unrecognized}")

        return recognized

    def configured_cache(
        self, compile_settings_hash: str
    ) -> Optional[CUBIECache]:
        """Return a CUBIECache configured with current hashes.

        Parameters
        ----------
        compile_settings_hash
            Hash of compile settings for cache disambiguation.

        Returns
        -------
        CUBIECache or None
            Configured cache instance if enabled, else None.
        """
        if self._cache is None:
            return None

        self._cache.set_hashes(
            system_hash=self.config.system_hash,
            compile_settings_hash=compile_settings_hash,
        )
        return self._cache

    def invalidate(self) -> None:
        """Invalidate the managed cache if in flush_on_change mode."""
        if self._cache is None:
            return
        if self.config.cache_mode != "flush_on_change":
            return
        self.flush()

    @property
    def cache_enabled(self) -> bool:
        """Return whether caching is enabled."""
        return self.config.cache_enabled
