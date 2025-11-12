"""Array validation helpers for batch solver components."""

from typing import Any, Optional

from cubie.cuda_simsafe import is_cuda_array


def cuda_array_validator(
    instance: Any,
    attribute: Any,
    value: Any,
    dimensions: Optional[int] = None,
) -> bool:
    """Validate that a value is a CUDA array with optional dimension checks.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.
    dimensions
        Required dimensionality when provided.

    Returns
    -------
    bool
        ``True`` when ``value`` is a CUDA array and, if requested, has the
        specified number of dimensions.

    Notes
    -----
    The ``instance`` and ``attribute`` parameters are accepted for attrs
    compatibility and are not used in the validation logic.
    """
    if dimensions is None:
        return is_cuda_array(value)
    return is_cuda_array(value) and len(value.shape) == dimensions


def optional_cuda_array_validator(
    instance: Any,
    attribute: Any,
    value: Any,
    dimensions: Optional[int] = None,
) -> bool:
    """Validate that a value is ``None`` or a CUDA array with optional checks.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.
    dimensions
        Required dimensionality when provided.

    Returns
    -------
    bool
        ``True`` when ``value`` is ``None`` or passes
        :func:`cuda_array_validator`.

    Notes
    -----
    The ``instance`` and ``attribute`` parameters are accepted for attrs
    compatibility and are not used in the validation logic.
    """
    if value is None:
        return True
    return cuda_array_validator(instance, attribute, value, dimensions)


def optional_cuda_array_validator_3d(
    instance: Any, attribute: Any, value: Any
) -> bool:
    """Validate that a value is ``None`` or a three-dimensional CUDA array.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.

    Returns
    -------
    bool
        ``True`` when ``value`` is ``None`` or is a three-dimensional CUDA
        array.

    Notes
    -----
    Delegates to :func:`optional_cuda_array_validator` with ``dimensions`` set
    to ``3``.
    """
    return optional_cuda_array_validator(
        instance, attribute, value, dimensions=3
    )


def optional_cuda_array_validator_2d(
    instance: Any, attribute: Any, value: Any
) -> bool:
    """Validate that a value is ``None`` or a two-dimensional CUDA array.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.

    Returns
    -------
    bool
        ``True`` when ``value`` is ``None`` or is a two-dimensional CUDA
        array.

    Notes
    -----
    Delegates to :func:`optional_cuda_array_validator` with ``dimensions`` set
    to ``2``.
    """
    return optional_cuda_array_validator(
        instance, attribute, value, dimensions=2
    )


def cuda_array_validator_3d(
    instance: Any, attribute: Any, value: Any
) -> bool:
    """Validate that a value is a three-dimensional CUDA array.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.

    Returns
    -------
    bool
        ``True`` when ``value`` is a three-dimensional CUDA array.

    Notes
    -----
    Delegates to :func:`cuda_array_validator` with ``dimensions`` set to ``3``.
    """
    return cuda_array_validator(instance, attribute, value, dimensions=3)


def cuda_array_validator_2d(
    instance: Any, attribute: Any, value: Any
) -> bool:
    """Validate that a value is a two-dimensional CUDA array.

    Parameters
    ----------
    instance
        Instance containing the attribute (required by attrs, unused here).
    attribute
        Attribute metadata supplied by attrs.
    value
        Value provided for validation.

    Returns
    -------
    bool
        ``True`` when ``value`` is a two-dimensional CUDA array.

    Notes
    -----
    Delegates to :func:`cuda_array_validator` with ``dimensions`` set to ``2``.
    """
    return cuda_array_validator(instance, attribute, value, dimensions=2)
