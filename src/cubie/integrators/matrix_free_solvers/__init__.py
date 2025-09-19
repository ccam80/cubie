"""Factories for matrix-free nonlinear solvers."""
from enum import IntEnum
from .linear_solver import (
    linear_solver_factory,
)
from .newton_krylov import newton_krylov_solver_factory


class SolverRetCodes(IntEnum):
    SUCCESS = 0
    NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1 # backtracking failed
    MAX_NEWTON_ITERATIONS_EXCEEDED = 2       # Newton loop hit max_iters
    MAX_LINEAR_ITERATIONS_EXCEEDED = 4       # inner linear solver did not
    # converge

__all__ = [
    "linear_solver_factory",
    "newton_krylov_solver_factory",
    'SolverRetCodes',
]
