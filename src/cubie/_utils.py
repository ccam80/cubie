"""Utility helpers used throughout :mod:`cubie`.

This module provides general-purpose helpers for array slicing, dictionary
updates and CUDA utilities that are shared across the code base.
"""

from functools import wraps
from time import time
from typing import Callable
from warnings import warn

import numpy as np
from numba import cuda, float32, float64, from_dtype, int32
from numba.cuda.random import (
    xoroshiro128p_dtype,
    xoroshiro128p_normal_float32,
    xoroshiro128p_normal_float64,
)
from attrs import fields, has

xoro_type = from_dtype(xoroshiro128p_dtype)


def slice_variable_dimension(slices, indices, ndim):
    """Create a combined slice for selected dimensions.

    Parameters
    ----------
    slices : slice or list[slice]
        Slice to apply to each index in ``indices``.
    indices : int or list[int]
        Dimension indices corresponding to ``slices``.
    ndim : int
        Total number of dimensions of the target array.

    Returns
    -------
    tuple
        Tuple of slice objects with ``slices`` applied to ``indices``.

    Raises
    ------
    ValueError
        If ``slices`` and ``indices`` differ in length or indices exceed
        ``ndim``.
    """
    if isinstance(slices, slice):
        slices = [slices]
    if isinstance(indices, int):
        indices = [indices]
    if len(slices) != len(indices):
        raise ValueError("slices and indices must have the same length")
    if max(indices) >= ndim:
        raise ValueError("indices must be less than ndim")

    outslice = [slice(None)] * ndim
    for i, s in zip(indices, slices):
        outslice[i] = s

    return tuple(outslice)


def in_attr(name, attrs_class_instance):
    """Check whether a field exists on an attrs class instance.

    Parameters
    ----------
    name : str
        Field name to query.
    attrs_class_instance : attrs class
        Instance whose fields are inspected.

    Returns
    -------
    bool
        ``True`` if ``name`` or ``_name`` is a field of the instance.
    """
    field_names = {
        field.name for field in fields(attrs_class_instance.__class__)
    }
    return name in field_names or ("_" + name) in field_names


def is_attrs_class(putative_class_instance):
    """Return ``True`` if the object is an attrs class instance.

    Parameters
    ----------
    putative_class_instance : Any
        Object to check.

    Returns
    -------
    bool
        Whether the object is an attrs class instance.
    """
    return has(putative_class_instance)


def update_dicts_from_kwargs(dicts: list | dict, **kwargs):
    """Update dictionaries with supplied keyword arguments.

    Scans the provided dictionaries for keys matching ``kwargs`` and updates
    their values. The function returns ``True`` if any dictionary was modified
    so callers can determine when cached objects require rebuilding.

    Parameters
    ----------
    dicts : list or dict
        Dictionaries to update.
    **kwargs
        Key-value pairs to apply to ``dicts``.

    Returns
    -------
    bool
        ``True`` if any dictionary entries were changed.

    Warns
    -----
    UserWarning
        If a key is not found or appears in multiple dictionaries.
    """
    if isinstance(dicts, dict):
        dicts = [dicts]

    dicts_modified = False

    for key, value in kwargs.items():
        kwarg_found = False
        for d in dicts:
            if key in d:
                if kwarg_found:
                    warn(
                        f"The parameter {key} was found in multiple dictionaries, and was updated in both.",
                        UserWarning,
                    )
                else:
                    kwarg_found = True

                if d[key] != value:
                    d[key] = value
                    dicts_modified = True

        if kwarg_found is False:
            warn(
                f"The parameter {key} was not found in the ODE algorithms dictionary"
                "of parameters",
                UserWarning,
            )

    return dicts_modified


def timing(_func=None, *, nruns=1):
    """Decorator for printing execution time statistics.

    Parameters
    ----------
    _func : callable, optional
        Function to decorate. Used when the decorator is applied without
        arguments.
    nruns : int, default=1
        Number of executions used to compute timing statistics.

    Returns
    -------
    callable
        Wrapped function or decorator.
    """

    def decorator(func):
        @wraps(func)
        def wrap(*args, **kw):
            durations = np.empty(nruns)
            for i in range(nruns):
                t0 = time()
                result = func(*args, **kw)
                durations[i] = time() - t0
            print(
                "func:%r took:\n %2.6e sec avg\n %2.6e max\n %2.6e min\n over %d runs"
                % (
                    func.__name__,
                    durations.mean(),
                    durations.max(),
                    durations.min(),
                    nruns,
                )
            )
            return result

        return wrap

    return decorator if _func is None else decorator(_func)


@cuda.jit(
    float64(
        float64,
        float64,
    ),
    device=True,
    inline=True,
)
def clamp_64(
    value,
    clip_value,
):
    """Clamp a 64-bit float to ``[-clip_value, clip_value]``.

    Parameters
    ----------
    value : float64
        Value to clamp.
    clip_value : float64
        Maximum absolute value allowed.

    Returns
    -------
    float64
        Clamped value.
    """
    # no cover: start
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value
    # no cover: end


@cuda.jit(
    float32(
        float32,
        float32,
    ),
    device=True,
    inline=True,
)
def clamp_32(
    value,
    clip_value,
):
    """Clamp a 32-bit float to ``[-clip_value, clip_value]``.

    Parameters
    ----------
    value : float32
        Value to clamp.
    clip_value : float32
        Maximum absolute value allowed.

    Returns
    -------
    float32
        Clamped value.
    """
    # no cover: start
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value
    # no cover: end


@cuda.jit(
    (float64[:], float64[:], int32, xoro_type[:]),
    device=True,
    inline=True,
)
def get_noise_64(
    noise_array,
    sigmas,
    idx,
    RNG,
):
    """Fill ``noise_array`` with Gaussian noise (float64).

    Parameters
    ----------
    noise_array : float64[:]
        Output array to populate.
    sigmas : float64[:]
        Standard deviations for each element.
    idx : int32
        Thread index used for RNG.
    RNG : xoro_type[:]
        RNG state array.
    """
    # no cover: start
    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float64(RNG, idx) * sigmas[i]
    # no cover: end


@cuda.jit(
    (float32[:], float32[:], int32, xoro_type[:]),
    device=True,
    inline=True,
)
def get_noise_32(
    noise_array,
    sigmas,
    idx,
    RNG,
):
    """Fill ``noise_array`` with Gaussian noise (float32).

    Parameters
    ----------
    noise_array : float32[:]
        Output array to populate.
    sigmas : float32[:]
        Standard deviations for each element.
    idx : int32
        Thread index used for RNG.
    RNG : xoro_type[:]
        RNG state array.
    """
    # no cover: start
    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float32(RNG, idx) * sigmas[i]
    # no cover: end


def round_sf(num, sf):
    """Round a number to a given number of significant figures.

    Parameters
    ----------
    num : float
        Number to round.
    sf : int
        Desired significant figures.

    Returns
    -------
    float
        ``num`` rounded to ``sf`` significant figures.
    """
    if num == 0.0:
        return 0.0
    else:
        return round(num, sf - 1 - int(np.floor(np.log10(abs(num)))))


def round_list_sf(list, sf):
    """Round each number in a list to significant figures.

    Parameters
    ----------
    list : Sequence[float]
        Numbers to round.
    sf : int
        Desired significant figures.

    Returns
    -------
    list[float]
        Rounded numbers.
    """
    return [round_sf(num, sf) for num in list]


def get_readonly_view(array):
    """Return a read-only view of ``array``.

    Parameters
    ----------
    array : numpy.ndarray
        Array to make read-only.

    Returns
    -------
    numpy.ndarray
        Read-only view of ``array``.
    """
    view = array.view()
    view.flags.writeable = False
    return view

def is_devfunc(func: Callable):
    """Test whether a callable is a Numba-generated CUDA device function.

    Parameters
    ----------
    func: Callable
        The function to inspect.

    Returns
    -------
    bool
        Whether the function is a Numba CUDA device function.
    """
    is_device = False
    if hasattr(func, 'targetoptions'):
        if func.targetoptions.get('device', False):
            is_device = True
    return is_device