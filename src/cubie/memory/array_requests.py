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

from cubie._utils import opt_getype_validator
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
    unchunkable
        Whether the memory manager is allowed to chunk the allocation.
    total_runs
        Total number of runs for chunking calculations. When ``None``, the
        array is not intended for run-axis chunking (e.g., driver_coefficients).
        Memory manager extracts this value to determine chunk parameters.
        Defaults to ``None``.

    Attributes
    ----------
    shape
        Tuple describing the requested array shape.
    dtype
        NumPy precision constructor used to produce the allocation.
    memory
        Memory placement option.
    chunk_axis_index
        Axis index along which chunking may occur.
    unchunkable
        Flag indicating that chunking should be disabled.
    total_runs
        Total number of runs for chunking calculations, or ``None`` if not
        applicable.
    """

    dtype = attrs.field(
        validator=val.in_([np.float64, np.float32, np.int32]),
    )
    shape: tuple[int, ...] = attrs.field(
        default=(1, 1, 1),
        validator=val.deep_iterable(
            val.instance_of(int), val.instance_of(tuple)
        ),
    )
    memory: str = attrs.field(
        default="device",
        validator=val.in_(["device", "mapped", "pinned", "managed"]),
    )
    chunk_axis_index: Optional[int] = attrs.field(
        default=2,
        validator=opt_getype_validator(int, 0),
    )
    unchunkable: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    total_runs: Optional[int] = attrs.field(
        default=None,
        validator=opt_getype_validator(int, 1),
    )

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
        Number of chunks the allocation was divided into.
    chunk_length
        Length of each chunk along the run axis (except possibly last).
    chunked_shapes
        Mapping from array labels to their per-chunk shapes. Empty dict when
        no chunking occurs.

    Attributes
    ----------
    arr
        Dictionary mapping array labels to allocated device arrays.
    chunks
        Number of chunks the allocation was divided into.
    chunk_length
        Length of each chunk along the run axis.
    chunked_shapes
        Mapping from array labels to their per-chunk shapes.
    """

    arr: dict[str, DeviceNDArrayBase] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
    chunks: int = attrs.field(
        default=1,
    )
    chunk_length: int = attrs.field(
        default=1,
    )
    chunked_shapes: dict[str, tuple[int, ...]] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )
