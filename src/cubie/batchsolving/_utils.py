"""
Batch Solving Utilities Module.

This module provides utility functions for batch solving operations, including
size validation, CUDA array detection, and array validation functions for
use with attrs-based classes.
"""

from cubie.cuda_simsafe import is_cuda_array


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
