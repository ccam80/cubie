"""Utility helpers used throughout :mod:`cubie`.

This module provides general-purpose helpers for array slicing, dictionary
updates and CUDA utilities that are shared across the code base.
"""
import inspect
from functools import wraps
from time import time
from typing import Any, Mapping, Tuple, Union, Optional, Iterable
from warnings import warn

import numpy as np
from numba import cuda, float32, float64, from_dtype, int32
from numba.cuda.random import (
    xoroshiro128p_dtype,
    xoroshiro128p_normal_float32,
    xoroshiro128p_normal_float64,
)
from attrs import fields, has, validators, Attribute
from cubie.cuda_simsafe import is_devfunc, selp

xoro_type = from_dtype(xoroshiro128p_dtype)

PrecisionDType = Union[
    type[np.float16],
    type[np.float32],
    type[np.float64],
    np.dtype[np.float16],
    np.dtype[np.float32],
    np.dtype[np.float64],
]

ALLOWED_PRECISIONS = {
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def precision_converter(value: PrecisionDType) -> type[np.floating]:
    """Return a canonical NumPy scalar type for precision configuration."""

    dtype = np.dtype(value)
    if dtype not in ALLOWED_PRECISIONS:
        raise ValueError(
            "precision must be one of float16, float32, or float64",
        )
    return dtype.type


def precision_validator(
    _: object,
    __: Attribute,
    value: PrecisionDType,
) -> None:
    """Validate that ``value`` resolves to a supported precision."""

    if np.dtype(value) not in ALLOWED_PRECISIONS:
        raise ValueError(
            "precision must be one of float16, float32, or float64",
        )


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


def split_applicable_settings(
    target: Any,
    settings: Mapping[str, Any],
    warn_on_unused: bool = True,
) -> Tuple[dict[str, Any], set[str], set[str]]:
    """Partition ``settings`` into accepted, missing, and unused entries.

    Uses the signature of ``target`` to determine which settings are
    applicable, then divides a mapping of arguments into three sets:

    - accepted: settings that are accepted by ``target``
    - missing: settings that are required by ``target`` but not provided
    - unused: settings that were provided but are not applicable to ``target``

    Parameters
    ----------
    target : Any
        Callable or class whose signature defines accepted keywords.
    settings : Mapping[str, Any]
        Mapping containing candidate configuration entries.
    warn_for_unused : bool, default=True
        If true, issue a warning for unused settings.
    Returns
    -------
    tuple
        Three-element tuple containing the filtered settings dictionary,
        the set of missing required keys, and the set of unused keys.
    """

    if inspect.isclass(target):
        signature = inspect.signature(target.__init__)
    else:
        signature = inspect.signature(target)

    accepted: set[str] = set()
    required: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        accepted.add(name)
        if parameter.default is inspect._empty:
            required.add(name)

    filtered = {
        key: value
        for key, value in settings.items()
        if key in accepted
    }
    missing = required - filtered.keys()
    unused = set(settings.keys()) - accepted
    if warn_on_unused and unused:
        warn(f"The following settings were ignored: {unused}", stacklevel=3)
    return filtered, missing, unused


def merge_kwargs_into_settings(
    kwargs: dict[str, object],
    valid_keys: Iterable[str],
    user_settings: Optional[dict[str, object]] = None,
) -> Tuple[dict[str, object], set[str]]:
    """Merge component settings from ``kwargs`` and ``user_settings``.

    Parameters
    ----------
    kwargs
        Keyword arguments supplied directly to a component.
    user_settings
        Explicit settings dictionary supplied by the caller. When provided,
        these values supply defaults that keyword arguments may override.
    valid_keys
        Iterable of keys recognised by the component. Only these keys are
        extracted from ``kwargs``.

    Returns
    -------
    merged
        Dictionary containing recognised settings with keyword arguments
        overriding values from ``user_settings``.
    unused
        Set of keys in ``kwargs`` that were not consumed.
    """

    allowed = set(valid_keys)
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    user_settings = {} if user_settings is None else user_settings.copy()
    duplicates = {key for key in filtered if key in user_settings}
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        warn(
            (
                "Duplicate settings were provided for keys "
                f"{{{joined}}}; values from keyword arguments take "
                "precedence over the explicit settings dictionary."
            ),
            UserWarning,
            stacklevel=2,
        )

    user_settings.update(filtered)
    recognized = set(filtered)
    return user_settings, recognized


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


def clamp_factory(precision):
    precision = from_dtype(precision)

    @cuda.jit(
        precision(precision, precision, precision),
        device=True,
        inline=True,
    )
    def clamp(value, minimum, maximum):
        clamped_high = selp(value > maximum, maximum, value)
        clamped = selp(clamped_high < minimum, minimum, clamped_high)
        return clamped

    return clamp


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


def is_device_validator(instance, attribute, value):
    """Validate that a value is a Numba CUDA device function."""
    if not is_devfunc(value):
        raise TypeError(
            f"{attribute} must be a Numba CUDA device function,"
            f"got {type(value)}."
        )


def float_array_validator(instance, attribute, value):
    """Validate that a value is a NumPy floating-point array with finite values.

    Raises a TypeError if the value is not a NumPy ndarray of floats, and a
    ValueError if any elements are NaN or infinite.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{attribute} must be a numpy array of floats, got {type(value)}.")
    if value.dtype.kind != 'f':
        raise TypeError(f"{attribute} must be a numpy array of floats, got dtype {value.dtype}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{attribute} must not contain NaNs or infinities.")


def inrangetype_validator(dtype, min_, max_):
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.ge(min_),
        validators.le(max_)
)

# Helper: expand Python dtype to accept corresponding NumPy scalar hierarchy
# e.g. float -> (float, np.floating), int -> (int, np.integer)
# Unknown types are returned unchanged.

def _expand_dtype(dtype):
    if dtype is float:
        return (float, np.floating)
    if dtype is int:
        return (int, np.integer)
    return dtype

def lttype_validator(dtype, max_):
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.lt(max_)
    )


def gttype_validator(dtype, min_):
    # Accept both built-in and NumPy scalar types
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.gt(min_)
    )

def letype_validator(dtype, max_):
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.le(max_)
    )


def getype_validator(dtype, min_):
    return validators.and_(
        validators.instance_of(_expand_dtype(dtype)),
        validators.ge(min_)
    )


def ensure_nonzero_size(
    value: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    """
    Replace zero-size shape with a one-size shape to ensure non-zero sizes.

    Parameters
    ----------
    value : Union[int, Tuple[int, ...]]
        Input value or tuple of values to process.

    Returns
    -------
    Union[int, Tuple[int, ...]]
        The input value with any zeros replaced by ones. For integers,
        returns max(1, value). For tuples, if any element is zero,
        returns a tuple of all ones with the same length.

    Examples
    --------
    >>> ensure_nonzero_size(0)
    1
    >>> ensure_nonzero_size(5)
    5
    >>> ensure_nonzero_size((0, 2, 0))
    (1, 1, 1)
    >>> ensure_nonzero_size((2, 3, 4))
    (2, 3, 4)
    """
    if isinstance(value, int):
        return max(1, value)
    elif isinstance(value, tuple):
        if any(v == 0 for v in value):
            return tuple(1 for v in value)
        else:
            return value
    else:
        return value
