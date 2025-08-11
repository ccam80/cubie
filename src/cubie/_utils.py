# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:24:23 2024

@author: cca79
"""
from functools import wraps
from time import time
from warnings import warn
from contextlib import contextmanager


import numpy as np
from numba import cuda
from numba import float64, int32, from_dtype, float32
from numba.cuda.random import xoroshiro128p_normal_float64, \
    xoroshiro128p_normal_float32, xoroshiro128p_dtype

xoro_type = from_dtype(xoroshiro128p_dtype)
from attrs import fields, has


def in_attr(name, attrs_class_instance):
    """Checks if a name is in the attributes of a class instance."""
    field_names = {field.name for field in
                   fields(attrs_class_instance.__class__)}
    return name in field_names or ("_" + name) in field_names


def is_attrs_class(putative_class_instance):
    """Checks if the given object is an attrs class instance."""
    return has(putative_class_instance)


def pinned_zeros(self, shape, dtype):
    """Returns a pinned array of zeros with the given shape and dtype."""
    npary = np.zeros(shape, dtype=dtype)
    return cuda.pinned_array_like(npary)


def update_dicts_from_kwargs(dicts: list | dict, **kwargs):
    """Helper function to update specific keys in the parameter d of classes
    which contain compiled objects -
    this function scans through the dicts to find any keys that match kwargs, and updates the values if they're
    different. The function returns True if any of the dicts were modified, to set a "needs rebuild" flag in the class
    if the d is used for compilation.

    Raises a UserWarning if any of the keys in kwargs were not found in the dicts. This doesn't error/stop code.

    Args:
        dicts (list[d): A list of dictionaries to update.
        **kwargs: Key-value pairs to update in the dictionaries.
    Returns:
        was_modified (bool): a flag that indicates if any d items were updated

    """
    if isinstance(dicts, dict):
        dicts = [dicts]

    dicts_modified = False

    for key, value in kwargs.items():
        kwarg_found = False
        for d in dicts:
            if key in d:
                if kwarg_found:
                    warn(f"The parameter {key} was found in multiple dictionaries, and was updated in both.",
                            UserWarning, )
                else:
                    kwarg_found = True

                if d[key] != value:
                    d[key] = value
                    dicts_modified = True

        if kwarg_found is False:
            warn(f"The parameter {key} was not found in the ODE algorithms dictionary"
                 "of parameters", UserWarning, )

    return dicts_modified


def timing(_func=None, *, nruns=1):
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kw):
            durations = np.empty(nruns)
            for i in range(nruns):
                t0 = time()
                result = func(*args, **kw)
                durations[i] = time() - t0
            print('func:%r took:\n %2.6e sec avg\n %2.6e max\n %2.6e min\n over %d runs' % (
                func.__name__, durations.mean(), durations.max(),
                durations.min(), nruns))
            return result

        return wrap

    return decorator if _func is None else decorator(_func)


@cuda.jit(float64(float64, float64, ), device=True, inline=True, )
def clamp_64(value, clip_value, ):
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value


@cuda.jit(float32(float32, float32, ), device=True, inline=True, )
def clamp_32(value, clip_value, ):
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value


@cuda.jit((float64[:], float64[:], int32, xoro_type[:]), device=True,
          inline=True, )
def get_noise_64(noise_array, sigmas, idx, RNG, ):
    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float64(RNG, idx) * sigmas[i]


@cuda.jit((float32[:], float32[:], int32, xoro_type[:]), device=True,
          inline=True, )
def get_noise_32(noise_array, sigmas, idx, RNG, ):
    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float32(RNG, idx) * sigmas[i]


def round_sf(num, sf):
    if num == 0.0:
        return 0.0
    else:
        return round(num, sf - 1 - int(np.floor(np.log10(abs(num)))))


def round_list_sf(list, sf):
    return [round_sf(num, sf) for num in list]


def get_readonly_view(array):
    view = array.view()
    view.flags.writeable = False
    return view

