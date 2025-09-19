"""
Numerical integration algorithms and settings for ODE solving.

This module provides a collection of numerical integration algorithms and
configuration classes for solving ordinary differential equations (ODEs) on
CUDA-enabled devices. It includes both the high-level configuration classes
and low-level algorithm implementations.

The module contains:
- Integration algorithm implementations (Euler method, etc.)
- Runtime and timing configuration classes
- Single integrator run coordination
- CUDA device function management

Examples
--------
The integrators are typically used through higher-level solver classes,
but can be accessed directly through the ImplementedAlgorithms registry.
"""
from enum import IntEnum

from cubie.integrators.algorithms import *

from cubie.integrators.matrix_free_solvers import (
    newton_krylov_solver_factory,
    linear_solver_factory
)

# Note: integer codes for linear solvers are the same to allow direct
# pass-through, but IntegratorReturnCodes.SUCCESS != SolverRetCodes.SUCCESS.
# Enums can not be subclassed.
class IntegratorReturnCodes(IntEnum):
    SUCCESS = 0,
    NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1   # backtracking failed
    MAX_NEWTON_ITERATIONS_EXCEEDED = 2         # Newton loop hit max_iters
    MAX_LINEAR_ITERATIONS_EXCEEDED = 4         # inner solver did not converge
    STEP_TOO_SMALL = 8                         # Step size < dt_min
    MAX_LOOP_ITERS_EXCEEDED = 16

__all__ = ["ImplementedAlgorithms",
           "newton_krylov_solver_factory",
           "linear_solver_factory",
           "IntegratorReturnCodes"]
