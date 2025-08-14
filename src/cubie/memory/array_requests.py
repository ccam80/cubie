from os import environ
from typing import Optional

import attrs
import attrs.validators as val
import numpy as np

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import (
        FakeCUDAArray as DeviceNDArrayBase)
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase

@attrs.define
class ArrayRequest:
    """Requested array spec: shape, dtype, memory location, stride order."""
    shape: tuple[int,...] = attrs.field(
            default=(1, 1, 1),
            validator=val.deep_iterable(
                    val.instance_of(int), val.instance_of(tuple)))
    # the np.float64 object being passed around is a "getset_descriptor",
    # not a dtype, and a type hint here just adds confusion or shows warnings.
    dtype = attrs.field(
            default=np.float64,
            validator=val.in_([np.float64, np.float32]))
    memory: str = attrs.field(
            default="device",
            validator=val.in_(["device", "mapped", "pinned", "managed"]))
    stride_order: Optional[tuple[str, ...]] = attrs.field(
            default=None,
            validator=val.optional(val.instance_of(tuple)))

    def __attrs_post_init__(self):
        """Set cubie-native stride order if not set already."""
        if self.stride_order is None:
            if len(self.shape) == 3:
                self.stride_order = ("time", "run", "variable")
            elif len(self.shape) == 2:
                self.stride_order = ("variable", "run")

    @property
    def size(self):
        return np.prod(self.shape, dtype=np.int64) * self.dtype().itemsize

@attrs.define
class ArrayResponse:
    """ Result of an array allocation: an array, and a number of chunks """
    arr: dict[str, DeviceNDArrayBase] = attrs.field(
            default=attrs.Factory(dict),
            validator=val.instance_of(dict))
    chunks: int = attrs.field(
            default=attrs.Factory(dict),
    )
    chunk_axis: str = attrs.field(default="run",
                                  validator=val.in_(["run", "variable", "time"]))