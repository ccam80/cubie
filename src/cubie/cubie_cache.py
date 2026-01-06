"""File-based caching infrastructure for CuBIE compiled kernels.

Provides cache classes that persist compiled CUDA kernels to disk,
enabling faster startup on subsequent runs with identical settings.
Cache files are stored in ``generated/<system_name>/cache/`` within
the configured GENERATED_DIR.

Notes
-----
This module depends on numba-cuda internal classes and may require
updates when numba-cuda versions change.
"""

import hashlib
from typing import Any

from attrs import fields, has
from numba.cuda.core.caching import (
    _CacheLocator,
    Cache,
    CacheImpl,
    IndexDataCacheFile,
)
from numpy import ndarray

from cubie.odesystems.symbolic.odefile import GENERATED_DIR


def hash_compile_settings(obj: Any) -> str:
    """Compute a stable hash from attrs compile settings.

    Traverses attrs class fields and computes a deterministic hash
    of all field values suitable for cache key construction.

    Parameters
    ----------
    obj
        An attrs class instance containing compile settings.

    Returns
    -------
    str
        SHA256 hash string of the serialized settings.

    Raises
    ------
    TypeError
        If obj is not an attrs class instance.

    Notes
    -----
    Fields marked with ``eq=False`` (typically callables) are skipped.
    Numpy arrays are hashed via ``tobytes()`` for determinism.
    Nested attrs classes are recursively processed.
    """
    if not has(type(obj)):
        raise TypeError(
            "obj must be an attrs class instance, "
            f"got {type(obj).__name__}"
        )

    parts = []
    for field in fields(type(obj)):
        # Skip fields with eq=False (e.g., callables, device functions)
        if field.eq is False:
            continue

        value = getattr(obj, field.name)
        serialized = _serialize_value(value)
        parts.append(f"{field.name}={serialized}")

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _serialize_value(value: Any) -> str:
    """Serialize a value to a string for hashing.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    str
        String representation suitable for hashing.
    """
    if value is None:
        return "None"
    elif isinstance(value, ndarray):
        # Hash array bytes for deterministic result
        array_hash = hashlib.sha256(value.tobytes()).hexdigest()
        return f"ndarray:{array_hash}"
    elif has(type(value)):
        # Recursively hash nested attrs classes
        return f"attrs:{hash_compile_settings(value)}"
    else:
        return str(value)


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
    """

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
    ) -> None:
        self._system_name = system_name
        self._system_hash = system_hash
        self._compile_settings_hash = compile_settings_hash
        self._cache_path = GENERATED_DIR / system_name / "cache"

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
    """

    # Override locator classes to use only CuBIE locator
    _locator_classes = []

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
    ) -> None:
        # Create CUBIECacheLocator directly (not via from_function)
        self._locator = CUBIECacheLocator(
            system_name, system_hash, compile_settings_hash
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


class CUBIECache(Cache):
    """File-based cache for CuBIE compiled kernels.

    Coordinates loading and saving of cached kernels, incorporating
    ODE system hash and compile settings hash into cache keys.

    Parameters
    ----------
    system_name
        Name of the ODE system.
    system_hash
        Hash representing the ODE system definition.
    compile_settings
        Attrs class instance of compile settings.

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
        compile_settings: Any,
    ) -> None:
        self._system_name = system_name
        self._system_hash = system_hash
        self._compile_settings_hash = hash_compile_settings(compile_settings)
        self._name = f"CUBIECache({system_name})"

        self._impl = CUBIECacheImpl(
            system_name,
            system_hash,
            self._compile_settings_hash,
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

    @property
    def cache_path(self) -> str:
        """Return the directory where cache files are stored.

        Returns
        -------
        str
            Absolute path to the cache directory.
        """
        return self._cache_path

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

    def load_overload(self, sig, target_context):
        """Load cached kernel with CUDA context handling.

        Parameters
        ----------
        sig
            Function signature.
        target_context
            CUDA target context.

        Returns
        -------
        Optional[_Kernel]
            Cached kernel or None if not found.
        """
        from numba.cuda import utils
        with utils.numba_target_override():
            return super().load_overload(sig, target_context)
