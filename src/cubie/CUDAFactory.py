"""Base classes for constructing cached CUDA device functions with Numba."""

from abc import ABC, abstractmethod
from typing import Set, Any, Tuple
import inspect

import attrs
import numpy as np
from numpy import array_equal, asarray
from numba import cuda
import numba

from cubie._utils import in_attr, is_devfunc
from cubie.time_logger import _default_timelogger
from cubie.cuda_simsafe import CUDA_SIMULATION


@attrs.define
class CUDAFunctionCache:
    """Base class for CUDAFactory cache containers.
    
    Automatically registers compilation timing events for device functions
    by introspecting attrs fields during initialization.
    """
    
    def __attrs_post_init__(self):
        """Register compilation events for all device function fields.

        Notes
        -----
        Iterates through all attrs fields and registers compilation timing
        events for fields containing CUDA device functions (identified by
        presence of 'py_func' attribute). Event names follow the pattern
        'compile_{field_name}' and are automatically associated with the
        'compile' category.
        """
        for field in attrs.fields(self.__class__):
            device_func = getattr(self, field.name)
            if device_func is None or device_func == -1:
                continue
            # Presuming dispatchers exists after "build", before this fn called
            if not hasattr(device_func, 'py_func'):
                continue
            
            event_name = f"compile_{field.name}"
            description = f"Compilation time for {field.name}"
            _default_timelogger.register_event(event_name, "compile",
                                             description)

def _create_placeholder_args(
    device_function: Any, precision: type
) -> tuple: # pragma: no cover
    """Create minimal placeholder arguments for device function.
    
    Parameters
    ----------
    device_function
        CUDA device function to create arguments for
    precision
        Numerical precision for scalar and array arguments
    
    Returns
    -------
    tuple
        Tuple of arguments with appropriate types and shapes
    
    Notes
    -----
    Creates minimal arrays for all parameters. If the device function has
    a `critical_shapes` attribute, those shapes are used instead of size-1
    arrays to avoid illegal memory access during dummy execution. If the
    device function has a `critical_values` attribute, those values are
    used for scalar parameters to avoid infinite loops in adaptive algorithms.
    """
    # Check if critical_shapes and critical_values attributes exist
    has_critical_shapes = hasattr(device_function, 'critical_shapes')
    critical_shapes = getattr(device_function, 'critical_shapes', None)
    has_critical_values = hasattr(device_function, 'critical_values')
    critical_values = getattr(device_function, 'critical_values', None)
    
    args_out = tuple()
    if hasattr(device_function, 'signatures'):
        sigs = device_function.signatures
        if sigs is not None:
            for sig_idx, sig in enumerate(sigs):
                args = tuple()
                for param_idx, item in enumerate(sig):
                    if isinstance(item, numba.types.Array):
                        # Determine shape - use critical_shapes if available
                        shape_available = (
                            has_critical_shapes and critical_shapes and
                            param_idx < len(critical_shapes)
                        )
                        if shape_available:
                            shape = critical_shapes[param_idx]
                            if isinstance(shape, int):
                                shape = (shape,) * item.ndim
                            if shape is None:
                                shape = (1,) * item.ndim
                        else:
                            shape = (1,) * item.ndim

                        # Create array with appropriate dtype and shape
                        if item.dtype == numba.float64:
                            args += (
                                cuda.to_device(
                                    np.ones(shape, dtype=np.float64)
                                ),
                            )
                        elif item.dtype == numba.float32:
                            args += (
                                cuda.to_device(
                                    np.ones(shape, dtype=np.float32)
                                ),
                            )
                        elif item.dtype == numba.types.float16:
                            args += (
                                cuda.to_device(
                                    np.ones(shape, dtype=np.float16)
                                ),
                            )
                        elif item.dtype == numba.int64:
                            args += (
                                cuda.to_device(np.ones(shape, dtype=np.int64)),
                            )
                        elif item.dtype == numba.int32:
                            args += (
                                cuda.to_device(np.ones(shape, dtype=np.int32)),
                            )
                        elif item.dtype == numba.types.int32:
                            args += (
                                cuda.to_device(np.ones(shape, dtype=np.int32)),
                            )
                    elif isinstance(item, numba.types.Integer):
                        # Use critical_values if available for this parameter
                        use_critical = (
                            has_critical_values and critical_values and
                            param_idx < len(critical_values) and
                            critical_values[param_idx] is not None
                        )
                        if use_critical:
                            value = critical_values[param_idx]
                        else:
                            value = 1

                        if item.bitwidth <= 8:
                            args += (np.int8(value),)
                        elif item.bitwidth <= 16:
                            args += (np.int32(value),)
                        elif item.bitwidth <= 32:
                            args += (np.int32(value),)
                        elif item.bitwidth <= 64:
                           args += (np.int64(value),)
                    elif isinstance(item, numba.types.Float):
                        # Use critical_values if available for this parameter
                        use_critical = (
                            has_critical_values and critical_values and
                            param_idx < len(critical_values) and
                            critical_values[param_idx] is not None
                        )
                        if use_critical:
                            value = critical_values[param_idx]
                        else:
                            value = 1.0

                        if item.bitwidth <= 16:
                            args += (np.float16(value),)
                        elif item.bitwidth <= 32:
                            args += (np.float32(value),)
                        elif item.bitwidth <= 64:
                            args += (np.float64(value),)
                    else:
                        raise TypeError(
                            f"Unsupported parameter type {item} in "
                            f"device function."
                        )
                args_out += (args,)
            return args_out
    # Fallback when no signatures available yet
    sig = inspect.signature(device_function.py_func)
    params = list(sig.parameters.keys())
    param_count = len(params)

    # Use critical_shapes and critical_values if available
    if has_critical_shapes and critical_shapes:
        args = tuple()
        for param_idx in range(param_count):
            shape_available = (
                param_idx < len(critical_shapes) and
                critical_shapes[param_idx] is not None
            )
            if shape_available:
                shape = critical_shapes[param_idx]
                args += (np.ones(shape, dtype=precision),)
            else:
                # Scalar or unknown - use critical_values if available
                use_critical = (
                    has_critical_values and critical_values and
                    param_idx < len(critical_values) and
                    critical_values[param_idx] is not None
                )
                if use_critical:
                    value = critical_values[param_idx]
                else:
                    value = 1.0
                args += (np.array(value, dtype=precision),)
        return (args,)
    else:
        default_args = tuple(
            np.array(1.0, dtype=precision) for _ in range(param_count)
        )
        return (default_args,)

def _run_placeholder_kernel(device_func: Any, placeholder_args: Tuple) -> \
        None: # pragma: no cover - device code
    """Create minimal CUDA kernel to trigger device function compilation.

    Parameters
    ----------
    device_func
        CUDA device function to wrap in kernel
    placeholder_args
        Tuple of placeholder arrays to pass to 

    Returns
    -------
    None
        Compiled CUDA kernel is called in routine, nothing returned

    Notes
    -----
    Calls numba.cuda.synchronize() with no stream; will hang until compilation
    and run are complete.
    """
    for signature in placeholder_args:

        param_count = len(signature)
        if param_count == 0:
            @cuda.jit
            def kernel():
                if cuda.grid(1) == 0:
                    device_func()
        elif param_count == 1:
            @cuda.jit
            def kernel(a1):
                if cuda.grid(1) == 0:
                    device_func(a1)
        elif param_count == 2:
            @cuda.jit
            def kernel(a1, a2):
                if cuda.grid(1) == 0:
                    device_func(a1, a2)
        elif param_count == 3:
            @cuda.jit
            def kernel(a1, a2, a3):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3)
        elif param_count == 4:
            @cuda.jit
            def kernel(a1, a2, a3, a4):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4)
        elif param_count == 5:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5)
        elif param_count == 6:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6)
        elif param_count == 7:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7)
        elif param_count == 8:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7, a8)
        elif param_count == 9:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9)
        elif param_count == 10:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
        elif param_count == 11:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
        elif param_count == 12:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
                if cuda.grid(1) == 0:
                    device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
        elif param_count == 13:
            @cuda.jit
            def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13
                    )
        elif param_count == 14:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14
                    )
        elif param_count == 15:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15
                    )
        elif param_count == 16:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16
                    )
        elif param_count == 17:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17
                    )
        elif param_count == 18:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18
                    )
        elif param_count == 19:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19
                    )
        elif param_count == 20:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19, a20
                    )
        elif param_count == 21:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20, a21
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19, a20, a21
                    )
        elif param_count == 22:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20, a21, a22
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19, a20, a21, a22
                    )
        elif param_count == 23:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20, a21, a22, a23
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23
                    )
        elif param_count == 24:
            @cuda.jit
            def kernel(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                a15, a16, a17, a18, a19, a20, a21, a22, a23, a24
            ):
                if cuda.grid(1) == 0:
                    device_func(
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
                        a13, a14, a15, a16, a17, a18, a19, a20, a21, a22,
                        a23, a24
                    )
        else:
            # Fallback for very large parameter counts
            raise ValueError(
                "CUDA device function has more than 24 parameters. "
                "Extend _run_placeholder_kernel to support this function."
            )
        # Give it a bunch of shared memory (actual number arbitrary)
        kernel[1, 1, 0, 32768](*signature)
        cuda.synchronize()

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
        self._timing_start = _default_timelogger.start_event
        self._timing_stop = _default_timelogger.stop_event
        self._timing_progress = _default_timelogger.progress

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
        return self.get_cached_output('device_function')

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

            # Check nested attrs classes and dicts if not found at top level
            r, u = self._check_nested_update(key, value)
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
                    old_value != value
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

    def _check_nested_update(self, key: str, value: Any) -> Tuple[bool, bool]:
        """Check nested attrs classes and dicts for a matching key.

        Searches one level of nesting within compile_settings attributes.
        If an attribute is an attrs class or dict, checks whether the key
        exists as a field/key within it. Uses the same comparison logic
        as _check_and_update.

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
        for field in attrs.fields(type(self._compile_settings)):
            nested_obj = getattr(self._compile_settings, field.name)

            # Check if nested object is an attrs class
            if attrs.has(type(nested_obj)):
                # Check with underscore prefix first, then without
                for attr_key in (f"_{key}", key):
                    if in_attr(attr_key, nested_obj):
                        old_value = getattr(nested_obj, attr_key)
                        value_changed = old_value != value

                        updated = False
                        if np.any(value_changed):
                            setattr(nested_obj, attr_key, value)
                            updated = True
                        return True, updated

            # Check if nested object is a dict
            elif isinstance(nested_obj, dict):
                if key in nested_obj:
                    old_value = nested_obj[key]
                    value_changed = old_value != value

                    updated = False
                    if np.any(value_changed):
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
        
        # Store cache and trigger auto-registration of compilation events
        self._cache = build_result
        self._cache_valid = True
        
        # Trigger compilation by running a placeholder kernel
        if _default_timelogger.verbosity is not None:
            for field in attrs.fields(type(self._cache)):
                device_func = getattr(self._cache, field.name)
                if device_func is None or device_func == -1:
                    continue
                if hasattr(device_func, 'py_func'):
                    event_name = f"compile_{field.name}"
                    self.specialize_and_compile(device_func, event_name)

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

    def specialize_and_compile(
        self, device_function: Any, event_name: str
    ) -> None:
        """Trigger compilation of device function and record timing.

        Parameters
        ----------
        device_function
            Numba CUDA device function to compile
        event_name
            Name of timing event to record (must be pre-registered)

        Notes
        -----
        Creates a minimal CUDA kernel that calls the device function
        with appropriately typed arguments. The kernel launch triggers
        Numba's JIT compilation, which is timed and recorded.

        Called automatically by _build() for all device functions
        returned from build(). Manual invocation is not needed.

        In CUDA simulator mode, timing is skipped silently as
        compilation does not occur.
        """
        if CUDA_SIMULATION:
            return
        precision = self._compile_settings.precision
        
        # Start timing
        self._timing_start(event_name)

        # Create placeholder arguments
        placeholder_args = _create_placeholder_args(device_function, precision)
        if is_devfunc(device_function):
            # Create and launch placeholder kernel
            _run_placeholder_kernel(device_function, placeholder_args)
        else:
            # If function is a kernel, just run it directly
            for signature in placeholder_args:
                # Give it a bunch of shared memory (actual number arbitrary)
                device_function[1,1,0,32768](*signature)

        cuda.synchronize()

        # Stop timing
        self._timing_stop(event_name)
