"""Radau IIA order 5 solver factory."""

from typing import Callable

from numba import cuda

from cubie.systemmodels.symbolic.operator_apply import residual_end_state_factory
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
    system.build()
    base_state = cuda.to_device(
        system.initial_values.values_array.astype(system.precision)
    )
    dxdt = system.dxdt_function
    residual = residual_end_state_factory(base_state, dxdt)
    operator = system.get_solver_helper("operator")

    linear_solver = linear_solver_factory(
        operator,
        correction_type="minimal_residual",
        tolerance=linear_tolerance,
        max_iters=linear_max_iters,
    )
    newton_solver = newton_krylov_solver_factory(
        residual,
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
