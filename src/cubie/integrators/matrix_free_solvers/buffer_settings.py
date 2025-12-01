"""DEPRECATED: Buffer settings moved to linear_solver.py.

Import from cubie.integrators.matrix_free_solvers.linear_solver instead.
This module is kept for backwards compatibility only.
"""

# Re-export from new location for backwards compatibility
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings,
    LinearSolverLocalSizes,
    LinearSolverSliceIndices,
)

__all__ = [
    "LinearSolverBufferSettings",
    "LinearSolverLocalSizes",
    "LinearSolverSliceIndices",
]
