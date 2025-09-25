"""Batch solving utilities for GPU-accelerated integrations."""

from cubie.cuda_simsafe import DeviceNDArrayBase, MappedNDArray

from typing import Optional, Union
from numpy.typing import NDArray

ArrayTypes = Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]

from cubie.outputhandling import summary_metrics  # noqa: E402
from cubie.batchsolving.solver import Solver, solve_ivp  # noqa: E402



__all__ = ["summary_metrics", "ArrayTypes", "Solver", "solve_ivp"]
