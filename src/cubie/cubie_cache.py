"""File-based caching infrastructure for CuBIE compiled kernels.

Provides cache classes that persist compiled CUDA kernels to disk,
enabling faster startup on subsequent runs with identical settings.
Cache files are stored in
``<cache root>/<system_name>/CUDA_cache_{system_hash[:8]}`` under the
shared cache root (:mod:`cubie.cache_root`) or in
``custom_dir/CUDA_cache_{system_hash[:8]}`` if a custom directory is
provided as an argument to "cache".

Notes
-----
This module depends on CUDA backend internal classes and may require
updates when backend versions change. The cache base classes are
imported from cubie.cuda_simsafe, which maps them per backend:
numba-cuda's ``CacheImpl``/``CUDACache`` (a vendored ``CUDACache``
under CUDASIM), or numba-cuda-mlir's ``MLIRCacheImpl``/``MLIRCache``
whose compile-result scheme (cubin/PTX payloads) carries its own
serialization.
"""

import marshal
import os
from contextlib import AbstractContextManager
from functools import cache
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
from shutil import rmtree
from sys import implementation, version_info
from time import monotonic, sleep
from types import CodeType
from typing import Any, Dict, Optional, Set, Union
from warnings import warn

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from attrs import field, validators as val, define, converters

from cubie.CUDAFactory import _CubieConfigBase
from cubie._env import kernel_cache_dir_default, max_cache_entries_default
from cubie._utils import getype_validator, build_config
from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import (  # noqa: F401
    _CacheLocator,  # noqa: F401
    CacheImpl,  # noqa: F401
    CUDACache,
    IndexDataCacheFile,  # noqa: F401
    cache_dumps,
    is_cudasim_enabled,
)
from cubie.cache_root import get_cache_root
from cubie.time_logger import default_timelogger
from cubie._utils import package_source_hash

# Register compile timing event with custom messages
default_timelogger.register_event(
    "compile_cuda_kernel",
    "compile",
    "CUDA kernel compilation time",
    start_message="Compiling CUDA kernel...",
    stop_message=" Compilation complete in {duration:.3f}s",
)


class _CacheFileLock(AbstractContextManager):
    """Cross-process lock for one cache index."""

    def __init__(self, path: Path, timeout: float = 120.0) -> None:
        self._path = path
        self._timeout = timeout
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        try:
            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()

            deadline = monotonic() + self._timeout
            while True:
                try:
                    self._handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(
                            self._handle.fileno(), msvcrt.LK_NBLCK, 1
                        )
                    else:
                        fcntl.flock(
                            self._handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    return self
                except OSError:
                    if monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for cache lock {self._path}."
                        ) from None
                    sleep(0.05)
        except BaseException:
            try:
                self._handle.close()
            finally:
                self._handle = None
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is None:
            return False
        try:
            self._handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self._handle.close()
            finally:
                self._handle = None
        return False


ALL_CACHE_PARAMETERS: Set[str] = {
    "cache_enabled",
    "cache_mode",
    "max_cache_entries",
    "cache_dir",
}
"""All keyword arguments accepted by :class:`CubieCacheHandler`.

These parameters can be passed to :class:`CubieCacheHandler` or to
:meth:`CubieCacheHandler.update`. Parent components use this set to
filter kwargs before forwarding.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``cache_enabled``
     - :class:`CacheConfig`
     - Whether file-based caching is enabled.
   * - ``cache_mode``
     - :class:`CacheConfig`
     - ``'hash'`` or ``'flush_on_change'``.
   * - ``max_cache_entries``
     - :class:`CacheConfig`
     - Maximum entries before LRU eviction (0 disables).
   * - ``cache_dir``
     - :class:`CacheConfig`
     - Custom cache directory path or ``None``.
"""


def _environment_entries() -> list[str]:
    """List the Python and installed-package versions."""
    entries = [
        f"python=={implementation.name}-{version_info.major}."
        f"{version_info.minor}.{version_info.micro}"
    ]
    for distribution in distributions():
        name = distribution.metadata["Name"] or "unknown"
        version = distribution.metadata["Version"] or "unknown"
        entries.append(f"{name}=={version}")
    return sorted(entries)


@cache
def environment_hash() -> str:
    """Hash Python and all installed package versions."""
    joined = "\n".join(_environment_entries())
    return sha256(joined.encode("utf-8")).hexdigest()


def _portable_code(code: CodeType) -> CodeType:
    """Remove checkout paths from a code object and its children."""
    constants = tuple(
        _portable_code(value) if isinstance(value, CodeType) else value
        for value in code.co_consts
    )
    return code.replace(co_filename="", co_consts=constants)


def _stable_value_key(value, active: set[int]):
    """Return a process-stable key for a captured value."""
    py_func = getattr(value, "py_func", None)
    if py_func is not None:
        return (
            "dispatcher",
            _function_key(py_func, active),
            _stable_value_key(value.targetoptions, active),
        )
    if isinstance(value, CodeType):
        code_hash = sha256(marshal.dumps(_portable_code(value))).hexdigest()
        return ("code", code_hash)
    if value.__class__.__name__ == "FastMathOptions":
        return ("fastmath", tuple(sorted(value.flags)))
    if isinstance(value, tuple):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("tuple", items)
    if isinstance(value, list):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("list", items)
    if isinstance(value, dict):
        items = (
            (_stable_value_key(key, active), _stable_value_key(item, active))
            for key, item in value.items()
        )
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        items = (_stable_value_key(item, active) for item in value)
        return ("set", tuple(sorted(items, key=repr)))
    return ("serialized", sha256(cache_dumps(value)).hexdigest())


def _stable_value_hash(value) -> str:
    """Hash a value without process-specific serialization order."""
    key = _stable_value_key(value, set())
    return sha256(cache_dumps(key)).hexdigest()


def _function_key(py_func, active: Optional[set[int]] = None):
    """Identify function code and compile-time captures."""
    if active is None:
        active = set()
    identity = id(py_func)
    if identity in active:
        return ("recursive", py_func.__module__, py_func.__qualname__)
    active.add(identity)
    try:
        closure = py_func.__closure__ or ()
        closure_key = tuple(
            _stable_value_key(cell.cell_contents, active) for cell in closure
        )
        defaults_key = _stable_value_key(
            (py_func.__defaults__, py_func.__kwdefaults__), active
        )
        code_hash = sha256(
            marshal.dumps(_portable_code(py_func.__code__))
        ).hexdigest()
        return (
            py_func.__module__,
            py_func.__qualname__,
            sha256(cache_dumps(closure_key)).hexdigest(),
            code_hash,
            sha256(cache_dumps(defaults_key)).hexdigest(),
        )
    finally:
        active.remove(identity)


def _portable_magic(value):
    """Normalize backend target values used in cache keys."""
    if hasattr(value, "major") and hasattr(value, "minor"):
        return (int(value.major), int(value.minor))
    if isinstance(value, tuple):
        return tuple(_portable_magic(item) for item in value)
    return value


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
            cache_root = get_cache_root() / system_name
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
            The system hash combined with the environment hash, so a
            change to any installed package invalidates the cache.
        """
        return f"{self._system_hash}-{environment_hash()}"

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


if IS_MLIR:

    class _KernelSerialization:
        """The MLIR compile-result scheme serializes itself.

        ``MLIRCacheImpl`` (the ``CacheImpl`` base on this backend)
        provides ``reduce``/``rebuild``/``check_cachable`` for
        cubin/PTX compile-result payloads.
        """

else:

    class _KernelSerialization:
        """numba-cuda kernel serialization via ``_Kernel`` methods."""

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
            else:  # pragma: no cover - simulated
                raise RuntimeError(
                    "CUBIECacheImpl.reduce() was called inside "
                    "CUDASIM mode, indicating a cache miss when "
                    "there are no compiled kernels available. This "
                    "indicates a config error; it should not be "
                    "reachable if CUDASIM mode was properly enabled."
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
            else:  # pragma: no cover - simulated
                raise RuntimeError(
                    "CUBIECacheImpl.rebuild() was called inside "
                    "CUDASIM mode, indicating a cache hit when "
                    "there are no compiled kernels available. This "
                    "indicates a config error; it should not be "
                    "reachable if CUDASIM mode was properly enabled."
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


class CUBIECacheImpl(_KernelSerialization, CacheImpl):
    """Serialization logic for CuBIE compiled kernels.

    Delegates actual serialization to the backend's kernel or
    compile-result methods
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
        system_name = self._locator._system_name
        disambiguator = self._locator.get_disambiguator()
        self._filename_base = f"{system_name}-{disambiguator}"
        return self._filename_base

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
        Set to 0 to disable eviction. ``None`` reads the
        ``CUBIE_MAX_CACHE_ENTRIES`` environment default.
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
        max_entries: Optional[int] = None,
        mode: str = "hash",
        custom_cache_dir: Optional[Path] = None,
        py_func=None,
    ) -> None:
        """Initialize CUBIECache with system and compile info.

        Note: Does not call inherited init using __super__(), absorbs the
        responsibilities directly due to  a different set of config parameters.
        """

        self._system_name = system_name
        self._system_hash = system_hash

        self._compile_settings_hash = config_hash

        self._name = f"CUBIECache({system_name})"
        if max_entries is None:
            max_entries = max_cache_entries_default()
        if custom_cache_dir is None:
            custom_cache_dir = kernel_cache_dir_default()
        self._max_entries = max_entries
        self._mode = mode
        self._function_key = None
        if py_func is not None:
            self._function_key = _function_key(py_func)

        self._impl = CUBIECacheImpl(
            system_name,
            system_hash,
            self._compile_settings_hash,
            custom_cache_dir=custom_cache_dir,
        )
        self._cache_path = self._impl.locator.get_cache_path()

        # numba-cuda's CUDACache gained launch-config state (PR #804):
        # the dispatcher calls is_launch_config_sensitive() on cached
        # launches. This __init__ intentionally does not chain to
        # CUDACache.__init__, so replicate that state here for the
        # inherited launch-config methods to work.
        self._launch_config_key = None
        self._launch_config_sensitive_flag = None
        marker_name = f"{self._impl.filename_base}.lcs"
        self._launch_config_marker_path = os.path.join(
            self._cache_path, marker_name
        )
        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=self._impl.filename_base,
            source_stamp=self._impl.locator.get_source_stamp(),
        )
        cache_path = Path(self._cache_path)
        self._write_lock_path = cache_path.with_name(
            f"{cache_path.name}.lock"
        )
        self.enable()

    def _index_key(self, sig, codegen):
        """Return the CuBIE and launch-specific cache key."""
        key = (
            sig,
            _portable_magic(codegen.magic_tuple()),
            self._system_hash,
            self._compile_settings_hash,
            package_source_hash(),
            self._function_key,
        )
        if self._launch_config_key is not None:
            key += (("launch_config", self._launch_config_key),)
        return key

    def load_overload(self, sig, target_context):
        """Load cached kernel, starting compile timer on cache miss.

        Parameters
        ----------
        sig
            Function signature.
        target_context
            CUDA target context for kernel reconstruction.

        Returns
        -------
        _Kernel or None
            Reconstructed CUDA kernel if cache hit, None if miss.
        """
        with _CacheFileLock(self._write_lock_path):
            result = super().load_overload(sig, target_context)

        if result is not None:
            # Cache hit - notify via TimeLogger
            default_timelogger.print_message(
                f"Matching compiled function found at: "
                f"{self._cache_path}. Skipping compile!"
            )
        else:
            # Cache miss - start compile timing via TimeLogger
            default_timelogger.print_message(
                "No cached file found. Beginning compilation... "
                "This can take several minutes for larger (n>30) systems."
            )
            default_timelogger.start_event("compile_cuda_kernel")

        return result

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
        if len(nbi_files) <= self._max_entries:
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
        """Save kernel to cache, stopping compile timer.

        Parameters
        ----------
        sig
            Function signature.
        data
            Kernel data to cache.
        """
        # Stop compile timing - TimeLogger handles the message
        default_timelogger.stop_event("compile_cuda_kernel")

        with _CacheFileLock(self._write_lock_path):
            self.enforce_cache_limit()
            super().save_overload(sig, data)

    def flush_cache(self) -> None:
        """Delete all cache files in the cache directory.

        Removes all .nbi and .nbc files, then recreates an empty
        cache directory.
        """
        cache_path = Path(self._cache_path)
        with _CacheFileLock(self._write_lock_path):
            if cache_path.exists():
                try:
                    rmtree(cache_path)
                except OSError:
                    pass
            cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        """Return the cache directory path."""
        return Path(self._cache_path)

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
        factory=max_cache_entries_default,
        validator=getype_validator(int, 0),
    )
    cache_dir: Optional[Path] = field(
        factory=kernel_cache_dir_default,
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
    """Handler for managing CuBIE kernel cache.

    Parameters
    ----------
    system_name
        Name of the ODE system for directory organization.
    system_hash
        Hash representing the ODE system definition.
    cache_arg
        Cache configuration shorthand: ``True`` enables caching with
        default path, ``False``/``None`` disables, a string or
        ``Path`` enables at that directory.
    **kwargs
        Additional overrides forwarded to :class:`CacheConfig`.
        See :data:`ALL_CACHE_PARAMETERS` for accepted keywords.

    See Also
    --------
    :class:`CacheConfig`
        Configuration container for cache settings.
    :data:`ALL_CACHE_PARAMETERS`
        Complete set of accepted keyword arguments.
    """

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
        if changed:
            if (
                self._cache is not None
                and self.config.cache_mode == "flush_on_change"
            ):
                self._cache.flush_cache()
            self._cache = None

        unrecognized = set(updates_dict.keys()) - recognized

        if unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {unrecognized}")

        return recognized

    def configured_cache(
        self, system_hash: str, compile_settings_hash: str
    ) -> Optional[CUBIECache]:
        """Return a CUBIECache configured with current hashes.

        Parameters
        ----------
        system_hash
            Hash representing the ODE system definition.
        compile_settings_hash
            Hash of compile settings for cache disambiguation.

        Returns
        -------
        CUBIECache or None
            Configured cache instance if enabled, else None.
        """
        if not self.config.cache_enabled:
            return None
        if system_hash != self.config.system_hash:
            self.config.update({"system_hash": system_hash})
        self._cache = CUBIECache(
            system_name=self.config.system_name,
            system_hash=system_hash,
            config_hash=compile_settings_hash,
            max_entries=self.config.max_cache_entries,
            mode=self.config.cache_mode,
            custom_cache_dir=self.config.cache_dir,
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
