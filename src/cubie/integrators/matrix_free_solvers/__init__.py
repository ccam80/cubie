"""Factories for matrix-free nonlinear solvers."""

from .linear_solver import minimal_residual_solver_factory
from .newton_krylov import (
    neumann_preconditioner_factory,
    newton_krylov_solver_factory,
)
from .rosenbrock import rosenbrock_solver_factory
from .general_irk import general_irk_solver_factory
from .radauiia5 import radauiia5_solver_factory

__all__ = [
    "minimal_residual_solver_factory",
    "newton_krylov_solver_factory",
    "neumann_preconditioner_factory",
    "rosenbrock_solver_factory",
    "general_irk_solver_factory",
    "radauiia5_solver_factory",
]
