"""Matrix-free solver factories used by integrator kernels.

The package exposes CUDA device function factories for linear and nonlinear
solvers that are consumed by modules in :mod:`cubie.integrators`.
"""

from enum import IntEnum

from .linear_solver import linear_solver_factory, linear_solver_cached_factory
from .newton_krylov import newton_krylov_solver_factory


class SolverRetCodes(IntEnum):
    """Enumerate outcomes returned by Newton--Krylov solvers.

    The integer codes flag convergence failures in the lower 16 bits of the
    status word emitted by :func:`newton_krylov_solver_factory`.
    """

    SUCCESS = 0
    NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1
    MAX_NEWTON_ITERATIONS_EXCEEDED = 2
    MAX_LINEAR_ITERATIONS_EXCEEDED = 4


__all__ = [
    "linear_solver_factory",
    "linear_solver_cached_factory",
    "newton_krylov_solver_factory",
    "SolverRetCodes",
]
