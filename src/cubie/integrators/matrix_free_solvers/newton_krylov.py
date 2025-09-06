"""Device function factories for Newton-Krylov solvers."""
from typing import Callable

from numba import cuda


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
) -> Callable:
    """Create a Newton-Krylov solver device function.

    Parameters
    ----------
    residual_function : callable
        Device function evaluating the nonlinear residual ``F(state)``.
    linear_solver : callable
        Device function solving ``J x = rhs`` for ``x``.
    tolerance : float
        Residual norm required for convergence.
    max_iters : int
        Maximum number of Newton iterations.
    damping : float, optional
        Step shrink factor used during backtracking, default ``0.5``.
    max_backtracks : int, optional
        Maximum number of damping attempts.

    Returns
    -------
    callable
        CUDA device function implementing a damped Newton-Krylov method.
    """
    tol_squared = tolerance*tolerance

    @cuda.jit(device=True)
    def newton_krylov_solver(
        state,
        parameters,
        drivers,
        h,  # Operator context
        delta,
        residual,
        preconditioned_vec,
        work_vec,
    ):
        n = state.shape[0]

        # Build initial rhs = -F(state) and norm in one pass
        residual_function(state, parameters, drivers, h, work_vec, residual)
        norm2_prev = 0.0
        for i in range(n):
            ri = residual[i]
            residual[i] = -ri
            delta[i] = 0.0
            norm2_prev += ri * ri
        if norm2_prev <= tol_squared:
            return True
        for _ in range(max_iters):
            # Solve J * delta = rhs  (rhs currently holds -F(state))
            linear_solver(
                state, parameters, drivers, h,
                residual,          # in: rhs, out: linear residual
                delta,             # in: initial guess out: Newton direction
                preconditioned_vec,
                work_vec,
            )

            # Backtrack loop - if the full step doesn't reduce the residual,
            # try smaller steps until we either reduce the residual or run out
            # of attempts
            scale = 1.0
            s_applied = 0.0

            for _bt in range(max_backtracks + 1):
                # Add difference in step size since last attempt
                coeff = scale - s_applied
                for i in range(n):
                    state[i] += coeff * delta[i]
                s_applied = scale

                # Residual function calculates guess - F(guess) - for example,
                # in a single backward Euler step, this is guess - step
                # start state - h*f(guess);
                residual_function(state, parameters, drivers, h,
                                  work_vec, residual)
                norm2_new = 0.0
                for i in range(n):
                    ri = residual[i]
                    norm2_new += ri * ri

                if norm2_new <= tol_squared:
                    return True
                if norm2_new < norm2_prev:
                    # Accept: prepare rhs = -F(state) in-place for next Newton iteration
                    norm2_prev = norm2_new
                    for i in range(n):
                        residual[i] = -residual[i]
                    break
                scale *= damping
            else:
                # No acceptable step: revert net update once and fail
                for i in range(n):
                    state[i] -= s_applied * delta[i]
                return False
            # Accepted but not converged; continue with prepared rhs

    return newton_krylov_solver