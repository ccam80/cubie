"""Batch solving utilities for GPU-accelerated integrations."""

from os import environ
if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import (
        FakeCUDAArray as DeviceNDArrayBase,
    )
    from numba.cuda.simulator.cudadrv.devicearray import (
        FakeCUDAArray as MappedNDArray,
    )
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase, MappedNDArray

from typing import Optional, Union
from numpy.typing import NDArray

ArrayTypes = Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]

from cubie.outputhandling import summary_metrics  # noqa: E402
from cubie.batchsolving.solver import Solver, solve_ivp  # noqa: E402



__all__ = ["summary_metrics", "ArrayTypes", "Solver", "solve_ivp"]
