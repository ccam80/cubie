"""Factories for matrix-free nonlinear solvers."""

from .linear_solver import (
    linear_solver_factory,
    neumann_preconditioner_factory,
)
from .newton_krylov import newton_krylov_solver_factory
from .rosenbrock import rosenbrock_solver_factory
from .general_irk import general_irk_solver_factory
from .radauiia5 import radauiia5_solver_factory

__all__ = [
    "linear_solver_factory",
    "neumann_preconditioner_factory",
    "newton_krylov_solver_factory",
    "rosenbrock_solver_factory",
    "general_irk_solver_factory",
]
