"""
Batch Solving Utilities Module.

This module provides utility functions for batch solving operations, including
size validation, CUDA array detection, and array validation functions for
use with attrs-based classes.
"""

from os import environ
from typing import Union, Tuple


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


if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":

    def is_cuda_array(value):
        """
        Check if value is a CUDA array (simulation mode).

        In CUDA simulation mode, any object with a 'shape' attribute
        is considered a CUDA array.

        Parameters
        ----------
        value : object
            Object to check.

        Returns
        -------
        bool
            True if the value has a 'shape' attribute, False otherwise.
        """
        return hasattr(value, "shape")
else:
    from numba.cuda import is_cuda_array


def cuda_array_validator(instance, attribute, value, dimensions=None):
    """
    Validate that a value is a CUDA array with optional dimension checking.

    This function is designed to be used as an attrs validator for
    CUDA array attributes.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object
        The value to validate.
    dimensions : int, optional
        If provided, also check that the array has this many dimensions.

    Returns
    -------
    bool
        True if value is a CUDA array and optionally has the correct
        number of dimensions.

    Notes
    -----
    This function is intended for use with the attrs library's validation
    system. The instance and attribute parameters are required by the attrs
    interface but are not used in the validation logic.
    """
    if dimensions is None:
        return is_cuda_array(value)
    else:
        return is_cuda_array(value) and len(value.shape) == dimensions


def optional_cuda_array_validator(instance, attribute, value, dimensions=None):
    """
    Validate that a value is None or a CUDA array with optional dimension checking.

    This function is designed to be used as an attrs validator for
    optional CUDA array attributes.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object or None
        The value to validate.
    dimensions : int, optional
        If provided, also check that the array has this many dimensions.

    Returns
    -------
    bool
        True if value is None or is a CUDA array and optionally has the
        correct number of dimensions.

    Notes
    -----
    This function is intended for use with the attrs library's validation
    system. The instance and attribute parameters are required by the attrs
    interface but are not used in the validation logic.
    """
    if value is None:
        return True
    return cuda_array_validator(instance, attribute, value, dimensions)


def optional_cuda_array_validator_3d(instance, attribute, value):
    """
    Validate that a value is None or a 3D CUDA array.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object or None
        The value to validate.

    Returns
    -------
    bool
        True if value is None or is a 3D CUDA array.

    Notes
    -----
    This is a convenience function that calls optional_cuda_array_validator
    with dimensions=3.
    """
    return optional_cuda_array_validator(
        instance, attribute, value, dimensions=3
    )


def optional_cuda_array_validator_2d(instance, attribute, value):
    """
    Validate that a value is None or a 2D CUDA array.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object or None
        The value to validate.

    Returns
    -------
    bool
        True if value is None or is a 2D CUDA array.

    Notes
    -----
    This is a convenience function that calls optional_cuda_array_validator
    with dimensions=2.
    """
    return optional_cuda_array_validator(
        instance, attribute, value, dimensions=2
    )


def cuda_array_validator_3d(instance, attribute, value):
    """
    Validate that a value is a 3D CUDA array.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object
        The value to validate.

    Returns
    -------
    bool
        True if value is a 3D CUDA array.

    Notes
    -----
    This is a convenience function that calls cuda_array_validator
    with dimensions=3.
    """
    return cuda_array_validator(instance, attribute, value, dimensions=3)


def cuda_array_validator_2d(instance, attribute, value):
    """
    Validate that a value is a 2D CUDA array.

    Parameters
    ----------
    instance : object
        The instance containing the attribute (unused but required by attrs).
    attribute : attr.Attribute
        The attribute being validated (unused but required by attrs).
    value : object
        The value to validate.

    Returns
    -------
    bool
        True if value is a 2D CUDA array.

    Notes
    -----
    This is a convenience function that calls cuda_array_validator
    with dimensions=2.
    """
    return cuda_array_validator(instance, attribute, value, dimensions=2)
