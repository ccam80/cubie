"""Coordinate GPU batch ODE solves and expose supporting infrastructure.

The package surfaces the :class:`Solver` class alongside :func:`solve_ivp`, a
convenience wrapper for configuring batch integrations. Supporting modules
provide grid construction, kernel compilation, system interfaces, and result
containers. The :mod:`cubie.batchsolving.arrays` subpackage hosts array
managers for host and device buffers used throughout the workflow.
"""

from typing import Optional, Union

from numpy.typing import NDArray

from cubie.cuda_simsafe import DeviceNDArrayBase, MappedNDArray

ArrayTypes = Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]

from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder  # noqa: E402
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig  # noqa: E402
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel  # noqa: E402
from cubie.batchsolving.SystemInterface import SystemInterface  # noqa: E402
from cubie.batchsolving._utils import (  # noqa: E402
    cuda_array_validator,
    cuda_array_validator_2d,
    cuda_array_validator_3d,
    optional_cuda_array_validator,
    optional_cuda_array_validator_2d,
    optional_cuda_array_validator_3d,
)
from cubie.batchsolving.arrays.BaseArrayManager import (  # noqa: E402
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
)
from cubie.batchsolving.arrays.BatchInputArrays import (  # noqa: E402
    InputArrayContainer,
    InputArrays,
)
from cubie.batchsolving.arrays.BatchOutputArrays import (  # noqa: E402
    ActiveOutputs,
    OutputArrayContainer,
    OutputArrays,
)
from cubie.batchsolving.solver import Solver, solve_ivp  # noqa: E402
from cubie.batchsolving.solveresult import SolveResult, SolveSpec  # noqa: E402
from cubie.outputhandling import summary_metrics  # noqa: E402


__all__ = [
    "ActiveOutputs",
    "ArrayContainer",
    "ArrayTypes",
    "BatchGridBuilder",
    "BatchSolverConfig",
    "BatchSolverKernel",
    "BaseArrayManager",
    "InputArrayContainer",
    "InputArrays",
    "ManagedArray",
    "OutputArrayContainer",
    "OutputArrays",
    "Solver",
    "SolveResult",
    "SolveSpec",
    "SystemInterface",
    "cuda_array_validator",
    "cuda_array_validator_2d",
    "cuda_array_validator_3d",
    "optional_cuda_array_validator",
    "optional_cuda_array_validator_2d",
    "optional_cuda_array_validator_3d",
    "solve_ivp",
    "summary_metrics",
]
