"""General implicit Runge--Kutta solver factory."""

from typing import Callable, Optional

from numba import cuda

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

    i_minus_hj = system.get_solver_helper("i-hj")
    residual_plus_i_minus_hj = system.get_solver_helper("r+i-hj")
    newton_solver = newton_krylov_solver_factory(
        tolerance, max_iters, linear_solver
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

    return general_irk_solver
