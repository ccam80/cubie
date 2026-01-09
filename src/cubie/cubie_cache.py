"""File-based caching infrastructure for CuBIE compiled kernels.

Provides cache classes that persist compiled CUDA kernels to disk,
enabling faster startup on subsequent runs with identical settings.
Cache files are stored in ``generated/<system_name>/cache/`` within
the configured GENERATED_DIR.

Notes
-----
This module depends on numba-cuda internal classes and may require
updates when numba-cuda versions change. When running under CUDA
simulator mode, caching is disabled and stub classes are provided.
"""

from pathlib import Path
from typing import Optional

from attrs import field, validators as val, define, converters

from cubie._utils import getype_validator
from cubie.cuda_simsafe import is_cudasim_enabled
from cubie.odesystems.symbolic.odefile import GENERATED_DIR

if not is_cudasim_enabled():
    from numba.cuda.core.caching import (
        _CacheLocator,
        CacheImpl,
        IndexDataCacheFile,
    )
    from numba.cuda.dispatcher import CUDACache

    _CACHING_AVAILABLE = True
else:
    # Stub classes for simulator mode
    _CacheLocator = object
    CacheImpl = object
    IndexDataCacheFile = None
    CUDACache = object
    _CACHING_AVAILABLE = False


@define
class CacheConfig:
    """Configuration for file-based kernel caching.

    Parameters
    ----------
    enabled
        Whether file-based caching is enabled.
    mode
        Caching mode: 'hash' for content-addressed caching,
        'flush_on_change' to clear cache when settings change.
    max_entries
        Maximum number of cache entries before LRU eviction.
        Set to 0 to disable eviction.
    cache_dir
        Custom cache directory. None uses default location.
    """

    enabled: bool = field(
        default=True,
        validator=val.instance_of(bool),
    )
    mode: str = field(
        default="hash",
        validator=val.in_(("hash", "flush_on_change")),
    )
    max_entries: int = field(
        default=10,
        validator=getype_validator(int, 0),
    )
    cache_dir: Optional[Path] = field(
        default=None,
        validator=val.optional(val.instance_of((str, Path))),
        converter=converters.optional(Path),
    )


class CUBIECacheLocator(_CacheLocator):
    """Locate cache files in CuBIE's generated directory structure.

    Directs cache files to ``generated/<system_name>/cache/`` instead
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
        if custom_cache_dir is not None:
            self._cache_path = Path(custom_cache_dir)
        else:
            self._cache_path = GENERATED_DIR / system_name / "CUDA_cache"

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
        # Create CUBIECacheLocator directly (not via from_function)
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
        return kernel._reduce_states()

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
        from numba.cuda.dispatcher import _Kernel

        return _Kernel._rebuild(**payload)

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
        config_hash: Optional[str] = None,
        max_entries: int = 10,
        mode: str = "hash",
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        # Caching not available in CUDA simulator mode
        # if not _CACHING_AVAILABLE:
        #     raise RuntimeError(
        #         "CUBIECache is not available in CUDA simulator mode. "
        #         "File-based caching requires a real CUDA environment."
        #     )

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
        # super().__init__() # Doesn't work as super().__init__ needs py_func

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
        # AI review note: this method is implementing a lot of our own
        # logic. Instead, we should use exiting Numba IndexCacheFile
        # mechanics to remove certain entries by index, or resave only
        # certain indices.
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
                import warnings

                warnings.warn(
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
        # AI review note: Can't we just use existing Numba cache flush logic?
        import shutil

        cache_path = Path(self._cache_path)
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
            except OSError:
                pass
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
