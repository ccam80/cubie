"""Radau IIA order 5 solver factory."""

from typing import Callable

from numba import cuda

from .linear_solver import linear_solver_factory
from .newton_krylov import newton_krylov_solver_factory


def radauiia5_solver_factory(
    system,
    newton_tolerance: float,
    newton_max_iters: int,
    linear_tolerance: float,
    linear_max_iters: int,
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
    Returns
    -------
    callable
        CUDA device function performing the solve.
    """
    i_minus_hj = system.get_solver_helper("i-hj")
    residual_plus_i_minus_hj = system.get_solver_helper("r+i-hj")

    linear_solver = linear_solver_factory(
        i_minus_hj,
        correction_type="minimal_residual",
        tolerance=linear_tolerance,
        max_iters=linear_max_iters,
    )
    newton_solver = newton_krylov_solver_factory(
        residual_plus_i_minus_hj,
        linear_solver,
        newton_tolerance,
        newton_max_iters,
    )

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
    ) -> None:
        """Solve the Radau IIA stage equations."""

        newton_solver(
            state,
            parameters,
            drivers,
            h,
            rhs,
            delta,
            residual,
            z_vec,
            v_vec,
        )

    return radauiia5_solver
