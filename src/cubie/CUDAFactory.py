"""Base classes for constructing cached CUDA device functions with Numba."""

from abc import ABC, abstractmethod
from typing import Set, Any

import attrs
import numpy as np
from numpy import array_equal, asarray

from cubie._utils import in_attr, is_attrs_class


class CUDAFactory(ABC):
    """Factory for creating and caching CUDA device functions.

    Subclasses implement :meth:`build` to construct Numba CUDA device functions
    or other cached outputs. Compile settings are stored as attrs classes and
    any change invalidates the cache to ensure functions are rebuilt when
    needed.

    Attributes
    ----------
    _compile_settings : attrs class or None
        Current compile settings.
    _cache_valid : bool
        Indicates whether cached outputs are valid.
    _device_function : callable or None
        Cached CUDA device function.
    _cache : attrs class or None
        Container for additional cached outputs.

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
        self._compile_settings = None
        self._cache_valid = True
        self._device_function = None
        self._cache = None

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
        if not attrs.has(compile_settings):
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
        if not self._cache_valid:
            self._build()
        return self._device_function

    @property
    def compile_settings(self):
        """Return the current compile settings object."""
        return self._compile_settings

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
                "Compile settings must be set up using self.setup_compile_settings before updating."
            )

        recognized_params = []
        updated_params = []

        for key, value in updates_dict.items():
            recognized, updated = self._check_and_update(f"_{key}", value)
            # Only check for a non-underscored name if there's no private attr
            if not recognized:
                r, u = self._check_and_update(key, value)
                recognized |= r
                updated |= u

            if recognized:
                recognized_params.append(key)
            if updated:
                updated_params.append(key)

        unrecognised_params = set(updates_dict.keys()) - set(recognized_params)
        if unrecognised_params and not silent:
            invalid = ", ".join(sorted(unrecognised_params))
            raise KeyError(
                f"'{invalid}' is not a valid compile setting for this "
                "object, and so was not updated.",
            )
        if updated_params:
            self._invalidate_cache()

        return set(recognized_params)

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
                    getattr(self._compile_settings, key) != value
                )
            except ValueError:
                # Maybe the size of an array has changed?
                value_changed = not array_equal(
                    asarray(old_value), asarray(value)
                )
            if np.any(value_changed): # Arrays will return an array of bools
                setattr(self._compile_settings, key, value)
                updated = True
            recognized = True

        return recognized, updated

    def _invalidate_cache(self):
        """Mark cached outputs as invalid."""
        self._cache_valid = False

    def _build(self):
        """Rebuild cached outputs if they are invalid."""
        build_result = self.build()

        # Multi-output case
        if is_attrs_class(build_result):
            self._cache = build_result
            # If 'device_function' is in the dict, make it an attribute
            if in_attr("device_function", build_result):
                self._device_function = build_result.device_function
        else:
            self._device_function = build_result

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
        if output_name == "device_function":
            return self._device_function
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
