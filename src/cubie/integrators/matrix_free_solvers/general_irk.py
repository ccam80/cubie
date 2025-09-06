"""General implicit Runge--Kutta solver factory."""

from typing import Callable, Optional

from numba import cuda

from cubie.systemmodels.symbolic.operator_apply import residual_end_state_factory
from .newton_krylov import newton_krylov_solver_factory


def general_irk_solver_factory(
    system,
    tolerance: float,
    max_iters: int,
    linear_solver: Callable,
    preconditioner: Optional[Callable] = None,
) -> Callable:
    """Create a solver for implicit Runge--Kutta stages.

    Parameters
    ----------
    system : SymbolicODE
        System providing solver helpers.
    tolerance : float
        Residual norm required for convergence.
    max_iters : int
        Maximum number of Newton iterations.
    linear_solver : callable
        Device function solving ``J x = rhs``.
    preconditioner : callable or None, optional
        Optional preconditioner device function.

    Returns
    -------
    callable
        CUDA device function performing Newton--Krylov iterations.
    """

    system.build()
    base_state = cuda.to_device(
        system.initial_values.values_array.astype(system.precision)
    )
    dxdt = system.dxdt_function
    residual = residual_end_state_factory(base_state, dxdt)
    newton_solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        tolerance=tolerance,
        max_iters=max_iters,
    )

    @cuda.jit(device=True)
    def general_irk_solver(
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
        """Solve the IRK nonlinear system."""

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

    return general_irk_solver
