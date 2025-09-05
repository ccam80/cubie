"""Radau IIA order 5 solver factory."""

from typing import Callable

from numba import cuda

from .linear_solver import minimal_residual_solver_factory
from .newton_krylov import (
    neumann_preconditioner_factory,
    newton_krylov_solver_factory,
)


def radauiia5_solver_factory(
    system,
    newton_tolerance: float,
    newton_max_iters: int,
    linear_tolerance: float,
    linear_max_iters: int,
    preconditioner_order: int = 1,
) -> Callable:
    """Create a Radau IIA 5th-order nonlinear solver.

    Parameters
    ----------
    system : SymbolicODE
        System providing solver helpers.
    newton_tolerance : float
        Residual norm required for Newton convergence.
    newton_max_iters : int
        Maximum number of Newton iterations.
    linear_tolerance : float
        Residual norm required for the linear solves.
    linear_max_iters : int
        Maximum number of linear iterations.
    preconditioner_order : int, default=1
        Order of the Neumann polynomial preconditioner.

    Returns
    -------
    callable
        CUDA device function performing the solve.
    """

    linear_solver = minimal_residual_solver_factory(
        linear_tolerance, linear_max_iters
    )
    preconditioner = neumann_preconditioner_factory(preconditioner_order)
    newton_solver = newton_krylov_solver_factory(
        newton_tolerance, newton_max_iters, linear_solver
    )
    i_minus_hj = system.get_solver_helper("i-hj")
    residual_plus_i_minus_hj = system.get_solver_helper("r+i-hj")

    @cuda.jit(device=True)
    def radauiia5_solver(
        state,
        parameters,
        drivers,
        h,
        residual,
        rhs,
        delta,
        z_vec,
        v_vec,
        precond_temp,
    ) -> None:
        """Solve the Radau IIA stage equations."""

        newton_solver(
            i_minus_hj,
            residual_plus_i_minus_hj,
            state,
            parameters,
            drivers,
            h,
            rhs,
            delta,
            residual,
            z_vec,
            v_vec,
            preconditioner,
            precond_temp,
        )

    return radauiia5_solver
