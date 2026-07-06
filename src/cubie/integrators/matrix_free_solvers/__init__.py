"""Matrix-free solver factories used by integrator kernels.

The package exposes CUDA device function factories for linear and nonlinear
solvers that are consumed by modules in :mod:`cubie.integrators`.
"""

from cubie.result_codes import CUBIE_RESULT_CODES
from .base_solver import MatrixFreeSolverConfig
from .linear_solver import (
    LinearSolver,
    LinearSolverConfig,
    LinearSolverCache,
)
from .newton_krylov import (
    NewtonKrylov,
    NewtonKrylovConfig,
    NewtonKrylovCache,
)


__all__ = [
    "LinearSolver",
    "LinearSolverConfig",
    "LinearSolverCache",
    "MatrixFreeSolverConfig",
    "NewtonKrylov",
    "NewtonKrylovConfig",
    "NewtonKrylovCache",
    "CUBIE_RESULT_CODES",
]
