"""
Array request and response data structures for memory management.

This module provides data structures for requesting and managing GPU memory
allocations with specific shapes, data types, and memory location requirements.
"""

from os import environ
from typing import Optional

import attrs
import attrs.validators as val
import numpy as np

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import (
        FakeCUDAArray as DeviceNDArrayBase,
    )
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase


@attrs.define
class ArrayRequest:
    """
    Specification for requesting array allocation with shape, dtype, and memory location.

    Parameters
    ----------
    shape : tuple of int, default (1, 1, 1)
        The shape of the requested array.
    dtype : numpy.dtype, default np.float64
        The data type for the array elements.
    memory : str, default "device"
        Memory location type. Must be one of "device", "mapped", "pinned", or "managed".
    stride_order : tuple of str or None, default None
        The order of strides for the array dimensions. If None, will be set
        automatically based on shape length.

    Attributes
    ----------
    shape : tuple of int
        The shape of the requested array.
    dtype : numpy.dtype
        The data type for the array elements.
    memory : str
        Memory location type.
    stride_order : tuple of str or None
        The order of strides for the array dimensions.

    Properties
    ----------
    size : int
        Total size of the array in bytes.

    Notes
    -----
    When stride_order is None, it will be automatically set during initialization:
    - For 3D arrays: ("time", "run", "variable")
    - For 2D arrays: ("variable", "run")
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
        default=np.float64, validator=val.in_([np.float64, np.float32])
    )
    memory: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed"]),
    )
    stride_order: Optional[tuple[str, ...]] = attrs.field(
        default=None, validator=val.optional(val.instance_of(tuple))
    )

    def __attrs_post_init__(self):
        """Set cubie-native stride order if not set already."""
        if self.stride_order is None:
            if len(self.shape) == 3:
                self.stride_order = ("time", "run", "variable")
            elif len(self.shape) == 2:
                self.stride_order = ("variable", "run")

    @property
    def size(self):
        """
        Calculate the total size of the array in bytes.

        Returns
        -------
        int
            Total size in bytes including element size and shape.
        """
        return np.prod(self.shape, dtype=np.int64) * self.dtype().itemsize


@attrs.define
class ArrayResponse:
    """
    Result of an array allocation containing arrays and chunking information.

    Parameters
    ----------
    arr : dict of str to DeviceNDArrayBase, default empty dict
        Dictionary mapping array labels to allocated device arrays.
    chunks : int, default empty dict
        Number of chunks the allocation was divided into.
    chunk_axis : str, default "run"
        The axis along which chunking was performed.

    Attributes
    ----------
    arr : dict of str to DeviceNDArrayBase
        Dictionary mapping array labels to allocated device arrays.
    chunks : int
        Number of chunks the allocation was divided into.
    chunk_axis : str
        The axis along which chunking was performed.
    """

    arr: dict[str, DeviceNDArrayBase] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
    chunks: int = attrs.field(
        default=attrs.Factory(dict),
    )
    chunk_axis: str = attrs.field(
        default="run", validator=val.in_(["run", "variable", "time"])
    )
