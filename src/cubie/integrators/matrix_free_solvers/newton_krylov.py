"""Device function factories for Newton-Krylov solvers."""
from typing import Callable

from numba import cuda


def newton_krylov_solver_factory(
    tolerance: float, max_iters: int, linear_solver: Callable
) -> Callable:
    """Create a Newton-Krylov solver device function.

    Parameters
    ----------
    tolerance : float
        Residual norm required for convergence.
    max_iters : int
        Maximum number of Newton iterations.
    linear_solver : callable
        Device function solving ``J x = rhs``.

    Returns
    -------
    callable
        CUDA device function implementing a Newton-Krylov method.
    """

    @cuda.jit(device=True)
    def newton_krylov_solver(
        jvp_function,
        residual_function,
        state,
        parameters,
        drivers,
        h,
        residual,
        rhs,
        delta,
        z_vec,
        v_vec,
        preconditioner,
        precond_temp,
    ):
        """Solve ``F(state) = 0`` using a Newton-Krylov iteration."""

        for _ in range(max_iters):
            residual_function(state, parameters, drivers, h, rhs, residual)
            norm = 0.0
            for i in range(residual.shape[0]):
                norm += residual[i] * residual[i]
            if norm ** 0.5 <= tolerance:
                return
            for i in range(residual.shape[0]):
                rhs[i] = -residual[i]
                delta[i] = 0.0
                residual[i] = rhs[i]
            linear_solver(
                jvp_function,
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
            for i in range(state.shape[0]):
                state[i] += delta[i]

    return newton_krylov_solver

