"""Base classes for constructing cached CUDA device functions with Numba."""

import hashlib
from abc import ABC, abstractmethod
from typing import Set, Any, Tuple

from attrs import define, field, fields, has
from numpy import (
    any as np_any,
    array,
    float16,
    float32,
    float64,
    int8,
    int32,
    int64,
    ones,
    array_equal,
    asarray,
    ndarray,
)
from numba import cuda
from numba import types as numba_types
from numba import float64 as numba_float64
from numba import float32 as numba_float32
from numba import int64 as numba_int64
from numba import int32 as numba_int32

from cubie._utils import in_attr
from cubie.time_logger import default_timelogger


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
        if hasattr(value, 'values_hash'):
            return f"config:{value.values_hash}"
        else:
            # Fallback for non-CUDAFactoryConfig attrs classes
            return f"attrs:{_hash_attrs_object(value)}"
    else:
        return str(value)


def _hash_attrs_object(obj: Any) -> str:
    """Compute a hash for an attrs object without values_hash method.

    Parameters
    ----------
    obj
        An attrs class instance.

    Returns
    -------
    str
        SHA256 hash string of the serialized fields.
    """
    parts = []
    for fld in fields(type(obj)):
        if fld.eq is False:
            continue
        value = getattr(obj, fld.name)
        serialized = _serialize_value(value)
        parts.append(f"{fld.name}={serialized}")

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


@define
class CUDAFactoryConfig:
    """Base class for CUDAFactory compile settings containers.

    Provides infrastructure for tracking configuration values and computing
    stable hashes for cache key generation. Subclasses should be defined
    with @attrs.define decorator.

    .. warning::

        **All field modifications MUST be done via the :meth:`update` method.**

        Direct attribute assignment (e.g., ``config.field = value``) will
        break cache invalidation and hashing logic. The ``update()`` method
        ensures ``values_tuple`` and ``values_hash`` are regenerated after
        any change, which is required for correct cache key generation.

    Notes
    -----
    The values_tuple and values_hash properties enable efficient cache
    invalidation by comparing configuration states. Fields with eq=False
    are excluded from hashing (typically callables or device functions).
    """

    _values_tuple: Tuple = field(
        default=None, init=False, repr=False, eq=False
    )
    _values_hash: str = field(default="", init=False, repr=False, eq=False)

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple[Set[str], Set[str]]:
        """Update configuration fields with new values.

        Parameters
        ----------
        updates_dict
            Mapping of setting names to new values. Keys should be
            non-underscored field names (e.g., ``"precision"`` not
            ``"_precision"``).
        **kwargs
            Additional settings to update.

        Returns
        -------
        tuple[set[str], set[str]]
            recognized: Names of settings that matched known fields.
            changed: Names of settings whose values were updated.

        Notes
        -----
        Checks field names and field aliases in a single pass. For fields
        with underscore-prefixed names (e.g., ``_precision``), the key in
        updates_dict should be the non-underscored form (``precision``),
        which matches the field's alias. After updates, values_tuple and
        values_hash are regenerated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if not updates_dict:
            return set(), set()

        recognized = set()
        changed = set()

        # Build a map of field names and aliases to field objects
        # Non-underscored keys map through: key -> field.alias -> field
        # or key -> field.name -> field
        field_map = {}
        for fld in fields(type(self)):
            field_map[fld.name] = fld
            if fld.alias is not None:
                field_map[fld.alias] = fld
            # Also map non-underscored name for underscore-prefixed fields
            if fld.name.startswith("_"):
                field_map[fld.name[1:]] = fld

        for key, value in updates_dict.items():
            fld = field_map.get(key)
            if fld is None:
                continue

            recognized.add(key)
            old_value = getattr(self, fld.name)

            # Determine if value changed, handling arrays
            if isinstance(old_value, ndarray) or isinstance(value, ndarray):
                value_changed = not array_equal(
                    asarray(old_value), asarray(value)
                )
            else:
                try:
                    value_changed = old_value != value
                except ValueError:
                    # Fallback for array-like comparisons
                    value_changed = not array_equal(
                        asarray(old_value), asarray(value)
                    )

            # Handle boolean or array result from comparison
            if isinstance(value_changed, ndarray):
                changed_flag = bool(np_any(value_changed))
            else:
                changed_flag = bool(value_changed)

            if changed_flag:
                setattr(self, fld.name, value)
                changed.add(key)

        # Regenerate hash after updates
        if changed:
            self._regenerate_hash()

        return recognized, changed

    def _regenerate_hash(self) -> None:
        """Regenerate values_tuple and values_hash from current field values.

        Called automatically after update() modifies any field values.
        """
        values = []
        for fld in fields(type(self)):
            if fld.eq is False:
                continue
            # Skip internal tracking fields
            if fld.name in ("_values_tuple", "_values_hash"):
                continue
            value = getattr(self, fld.name)
            if has(type(value)) and hasattr(value, 'values_hash'):
                # Nested CUDAFactoryConfig: use its hash for consistency
                values.append(f"config:{value.values_hash}")
            else:
                values.append(_serialize_value(value))

        self._values_tuple = tuple(values)
        combined = "|".join(str(v) for v in self._values_tuple)
        self._values_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def values_tuple(self) -> Tuple:
        """Tuple of serialized values for all eq=True fields.

        Returns
        -------
        tuple
            Ordered serialized values of configuration fields. For nested
            CUDAFactoryConfig instances, includes their hash representation
            (as "config:<hash>") for consistency with the serialization
            used elsewhere.
        """
        if self._values_tuple is None:
            self._regenerate_hash()
        return self._values_tuple

    @property
    def values_hash(self) -> str:
        """SHA256 hexdigest of the values_tuple.

        Returns
        -------
        str
            64-character hex string representing the configuration state.
        """
        if not self._values_hash:
            self._regenerate_hash()
        return self._values_hash


@define
class CUDAFunctionCache:
    """Base class for CUDAFactory cache containers."""
    pass


class CUDAFactory(ABC):
    """Factory for creating and caching CUDA device functions.

    Subclasses implement :meth:`build` to construct Numba CUDA device functions
    or other cached outputs. Compile settings are stored as attrs classes and
    any change invalidates the cache to ensure functions are rebuilt when
    needed.

    .. warning::

        **All compile settings modifications MUST be done via
        :meth:`update_compile_settings`.**

        Direct attribute assignment on compile_settings (e.g.,
        ``factory.compile_settings.field = value``) will break cache
        invalidation and hashing logic. The ``update_compile_settings()``
        method ensures the cache is properly invalidated and hash values
        are regenerated.

    Attributes
    ----------
    _compile_settings : attrs class or None
        Current compile settings.
    _cache_valid : bool
        Indicates whether cached outputs are valid.
    _cache : attrs class or None
        Container for cached outputs (CUDAFunctionCache subclass).

    Notes
    -----
    There is potential for a cache mismatch when doing the following:

    ```python
    device_function = self.device_function  # calls build if settings updated
    self.update_compile_settings(new_setting=value)  # updates settings but
    does not rebuild

    device_function(argument_derived_from_new_setting)  # this will use the
    old device function, not the new one
    ```

    The lesson is: Always use CUDAFactory.device_function at the point of
    use, otherwise you'll defeat the cache invalidation logic.

    If your build function returns multiple cached items, create a cache
    class decorated with @attrs.define. For example:
    ```python
    @attrs.define
    class MyCache:
        device_function: callable
        other_output: int
    ```
    Then, in your build method, return an instance of this class:
    ```python
    def build(self):
        return MyCache(device_function=my_device_function, other_output=42)
    ```

    The current cache validity can be checked using the `cache_valid` property,
    which will return True if the cache
    is valid and False otherwise.
    """

    def __init__(self):
        """Initialize the CUDA factory.

        Notes
        -----
        Uses the global default time logger from cubie.time_logger.
        Configure timing via solve_ivp(time_logging_level=...) or
        Solver(time_logging_level=...).
        """
        self._compile_settings = None
        self._cache_valid = True
        self._cache = None

        # Use global default logger callbacks
        self._timing_start = default_timelogger.start_event
        self._timing_stop = default_timelogger.stop_event
        self._timing_progress = default_timelogger.progress

    @abstractmethod
    def build(self):
        """Build and return the CUDA device function.

        This method must be overridden by subclasses.

        Returns
        -------
        callable or attrs class
            Compiled CUDA function or container of cached outputs.
        """
        return None

    def setup_compile_settings(self, compile_settings):
        """Attach a container of compile-critical settings to the object.

        Parameters
        ----------
        compile_settings : attrs class
            Settings object used to configure the CUDA function.

        Notes
        -----
        Any existing settings are replaced.
        """
        if not has(compile_settings):
            raise TypeError(
                "Compile settings must be an attrs class instance."
            )
        self._compile_settings = compile_settings
        self._invalidate_cache()

    @property
    def cache_valid(self):
        """bool: ``True`` if cached outputs are up to date."""

        return self._cache_valid

    @property
    def device_function(self):
        """Return the compiled CUDA device function.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output('device_function')

    @property
    def compile_settings(self):
        """Return the current compile settings object."""
        return self._compile_settings

    @property
    def config_hash(self) -> str:
        """Return the hash of the current compile settings.

        Returns
        -------
        str
            SHA256 hexdigest of the compile settings. Returns empty string
            if compile_settings is None.

        Notes
        -----
        For CUDAFactoryConfig-based settings, uses values_hash directly.
        For legacy attrs classes, falls back to _hash_attrs_object.
        Subclasses with child CUDAFactory objects should override this
        to combine their hashes.
        """
        if self._compile_settings is None:
            return ""
        if hasattr(self._compile_settings, 'values_hash'):
            return self._compile_settings.values_hash
        # Fallback for legacy attrs classes
        return _hash_attrs_object(self._compile_settings)

    def update_compile_settings(
        self, updates_dict=None, silent=False, **kwargs
    ) -> Set[str]:
        """Update compile settings with new values.

        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of setting names to new values.
        silent : bool, default=False
            Suppress errors for unrecognised parameters.
        **kwargs
            Additional settings to update.

        Returns
        -------
        set[str]
            Names of settings that were successfully updated.

        Raises
        ------
        ValueError
            If compile settings have not been set up.
        KeyError
            If an unrecognised parameter is supplied and ``silent`` is ``False``.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        if self._compile_settings is None:
            raise ValueError(
                "Compile settings must be set up using "
                "self.setup_compile_settings before updating."
            )

        recognized_params = set()
        updated_params = set()

        # Use CUDAFactoryConfig.update() if available
        if hasattr(self._compile_settings, 'update'):
            recognized, changed = self._compile_settings.update(updates_dict)
            recognized_params.update(recognized)
            updated_params.update(changed)
            # Remove recognized keys from updates_dict for nested checking
            remaining_dict = {
                k: v for k, v in updates_dict.items()
                if k not in recognized_params
            }
        else:
            # Legacy path: check individual fields
            remaining_dict = updates_dict.copy()
            for key, value in updates_dict.items():
                recognized, updated = self._check_and_update(f"_{key}", value)
                # Check for non-underscored name if no private attr found
                if not recognized:
                    r, u = self._check_and_update(key, value)
                    recognized |= r
                    updated |= u

                if recognized:
                    recognized_params.add(key)
                    remaining_dict.pop(key, None)
                if updated:
                    updated_params.add(key)

        # Check nested attrs classes and dicts for remaining keys
        for key, value in remaining_dict.items():
            r, u = self._check_nested_update(key, value)
            if r:
                recognized_params.add(key)
            if u:
                updated_params.add(key)

        unrecognised_params = set(updates_dict.keys()) - recognized_params
        if unrecognised_params and not silent:
            invalid = ", ".join(sorted(unrecognised_params))
            raise KeyError(
                f"'{invalid}' is not a valid compile setting for this "
                "object, and so was not updated.",
            )
        if updated_params:
            self._invalidate_cache()

        return recognized_params

    def _check_and_update(self,
                          key: str,
                          value: Any):
        """Check a single compile setting and update if changed.

        More permissive than !=, as it catches arrays too and registers a
        mismatch for incompatible types instead of raising an error.

        Parameters
        ----------
        key
            Attribute name in the compile_settings object
        value
            New value for the attribute

        Returns
        -------
        tuple (bool, bool)
            recognized: The key appears in the compile_settings object
            updated: The value has changed.
        """
        updated = False
        recognized = False
        if in_attr(key, self._compile_settings):
            old_value = getattr(self._compile_settings, key)
            try:
                value_changed = (
                    old_value != value
                )
            except ValueError:
                # Maybe the size of an array has changed?
                value_changed = not array_equal(
                    asarray(old_value), asarray(value)
                )
            if np_any(value_changed):  # Arrays will return an array of bools
                setattr(self._compile_settings, key, value)
                updated = True
            recognized = True

        return recognized, updated

    def _check_nested_update(self, key: str, value: Any) -> Tuple[bool, bool]:
        """Check nested attrs classes and dicts for a matching key.

        Searches one level of nesting within compile_settings attributes.
        If an attribute is an attrs class or dict, checks whether the key
        exists as a field/key within it. Uses CUDAFactoryConfig.update()
        for nested CUDAFactoryConfig instances, or falls back to direct
        attribute comparison.

        Parameters
        ----------
        key
            Attribute name to search for in nested structures
        value
            New value for the attribute

        Returns
        -------
        tuple (bool, bool)
            recognized: The key was found in a nested structure
            updated: The value has changed and was updated

        Notes
        -----
        Only updates values when the new value is type-compatible with the
        existing attribute. This prevents accidental type mismatches when
        a key name collides across different nested structures.
        """
        for fld in fields(type(self._compile_settings)):
            nested_obj = getattr(self._compile_settings, fld.name)

            # Check if nested object is an attrs class
            if has(type(nested_obj)):
                # Use CUDAFactoryConfig.update() if available
                if hasattr(nested_obj, 'update'):
                    recognized, changed = nested_obj.update({key: value})
                    if recognized:
                        return True, bool(changed)
                else:
                    # Legacy path: check with underscore prefix first
                    for attr_key in (f"_{key}", key):
                        if in_attr(attr_key, nested_obj):
                            old_value = getattr(nested_obj, attr_key)
                            value_changed = old_value != value

                            updated = False
                            if np_any(value_changed):
                                setattr(nested_obj, attr_key, value)
                                updated = True
                            return True, updated

            # Check if nested object is a dict
            elif isinstance(nested_obj, dict):
                if key in nested_obj:
                    old_value = nested_obj[key]
                    value_changed = old_value != value

                    updated = False
                    if np_any(value_changed):
                        nested_obj[key] = value
                        updated = True
                    return True, updated

        return False, False

    def _invalidate_cache(self):
        """Mark cached outputs as invalid."""
        self._cache_valid = False

    def _build(self):
        """Rebuild cached outputs if they are invalid."""
        build_result = self.build()

        if not isinstance(build_result, CUDAFunctionCache):
            raise TypeError(
                "build() must return an attrs class (CUDAFunctionCache "
                "subclass)"
            )

        self._cache = build_result
        self._cache_valid = True

    def get_cached_output(self, output_name):
        """Return a named cached output.

        Parameters
        ----------
        output_name : str
            Name of the cached item to retrieve.

        Returns
        -------
        Any
            Cached value associated with ``output_name``.

        Raises
        ------
        KeyError
            If ``output_name`` is not present in the cache.
        NotImplementedError
            If a cache has been filled with a "-1" integer, this indicates
            that the requested object is not implemented in the subclass.
        """
        if not self.cache_valid:
            self._build()
        if self._cache is None:
            raise RuntimeError("Cache has not been initialized by build().")
        if not in_attr(output_name, self._cache):
            raise KeyError(
                f"Output '{output_name}' not found in cached outputs."
            )
        cache_contents = getattr(self._cache, output_name)
        if type(cache_contents) is int and cache_contents == -1:
            raise NotImplementedError(
                f"Output '{output_name}' is not implemented in this class."
            )
        return cache_contents
