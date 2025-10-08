"""Structured array allocation requests and responses for GPU memory.

This module defines lightweight data containers that describe array allocation
requirements and report allocation outcomes. Requests capture shape, precision,
and memory placement details, while responses track allocated buffers and any
chunking performed by the memory manager.
"""

from typing import Optional

import attrs
import attrs.validators as val
import numpy as np

from cubie.cuda_simsafe import DeviceNDArrayBase


@attrs.define
class ArrayRequest:
    """Specification for requesting array allocation.

    Parameters
    ----------
    shape
        Tuple describing the requested array shape. Defaults to ``(1, 1, 1)``.
    dtype
        NumPy precision constructor used to produce the allocation. Defaults to
        :func:`numpy.float64`. Integer status buffers use :func:`numpy.int32`.
    memory
        Memory placement option. Must be one of ``"device"``, ``"mapped"``,
        ``"pinned"``, or ``"managed"``.
    stride_order
        Optional tuple describing logical dimension labels in stride order. When
        omitted, the initializer selects an order based on dimensionality.
    unchunkable
        Whether the memory manager is allowed to chunk the allocation.

    Attributes
    ----------
    shape
        Tuple describing the requested array shape.
    dtype
        NumPy precision constructor used to produce the allocation.
    memory
        Memory placement option.
    stride_order
        Tuple describing logical dimension labels in stride order.
    unchunkable
        Flag indicating that chunking should be disabled.

    Notes
    -----
    When ``stride_order`` is ``None``, it is set automatically during
    initialization:

    * For 3D arrays, ``("time", "run", "variable")`` is selected.
    * For 2D arrays, ``("variable", "run")`` is selected.
    """

    shape: tuple[int, ...] = attrs.field(
        default=(1, 1, 1),
        validator=val.deep_iterable(
            val.instance_of(int), val.instance_of(tuple)
        ),
    )
    # the np.float64 object being passed around is a "getset_descriptor",
    # not a dtype, and a type hint here just adds confusion or shows warnings.
    dtype = attrs.field(
        default=np.float64,
        validator=val.in_([np.float64, np.float32, np.int32]),
    )
    memory: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed"]),
    )
    stride_order: Optional[tuple[str, ...]] = attrs.field(
        default=None, validator=val.optional(val.instance_of(tuple))
    )
    unchunkable: bool = attrs.field(default=False, validator=val.instance_of(bool))

    def __attrs_post_init__(self) -> None:
        """
        Set cubie-native stride order if not set already.

        Returns
        -------
        None
            ``None``.
        """
        if self.stride_order is None:
            if len(self.shape) == 3:
                self.stride_order = ("time", "run", "variable")
            elif len(self.shape) == 2:
                self.stride_order = ("variable", "run")

    @property
    def size(self) -> int:
        """Total size of the array in bytes."""
        return np.prod(self.shape, dtype=np.int64) * self.dtype().itemsize


@attrs.define
class ArrayResponse:
    """Result of an array allocation containing buffers and chunking data.

    Parameters
    ----------
    arr
        Dictionary mapping array labels to allocated device arrays.
    chunks
        Mapping that records how many chunks each allocation was divided into.
    chunk_axis
        Axis label along which chunking was performed. Defaults to ``"run"``.

    Attributes
    ----------
    arr
        Dictionary mapping array labels to allocated device arrays.
    chunks
        Mapping that records how many chunks each allocation was divided into.
    chunk_axis
        Axis label along which chunking was performed.
    """

    arr: dict[str, DeviceNDArrayBase] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
    chunks: int = attrs.field(
        default=1,
    )
    chunk_axis: str = attrs.field(
        default="run", validator=val.in_(["run", "variable", "time"])
    )
