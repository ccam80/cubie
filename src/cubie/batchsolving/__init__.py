from os import environ
from typing import Optional, Union
from numpy.typing import NDArray

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import \
        FakeCUDAArray as DeviceNDArrayBase
    from numba.cuda.simulator.cudadrv.devicearray import (FakeCUDAArray as
                                                          MappedNDArray)
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase, MappedNDArray

ArrayTypes = Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]
# Import and re-export for top-level access
from cubie.outputhandling import summary_metrics

__all__ = ['summary_metrics', 'ArrayTypes']
