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
        state, parameters, drivers, h,    # Operator context
        linear_rhs,
        delta,
        residual,
        trial_state,
        work_vec,
    ):
        n = state.shape[0]

        for _ in range(max_iters):
            accepted = False
            scale = 1.0
            norm2_prev = 0.0

            # Evaluate residual and check convergence
            residual_function(state, parameters, drivers, h, linear_rhs, residual)
            for i in range(n):
                ri = residual[i]
                linear_rhs[i] = -residual[i]
                delta[i] = 0.0
                norm2_prev += ri * ri
            if norm2_prev <= tol_squared:
                return True

            # Run Krylov solver on linearized (by matrix-free Jv) system to get
            # a reasonable step direction
            linear_solver(
                state, parameters, drivers, h,
                linear_rhs,
                delta,
                residual,
                trial_state,
                work_vec,
            )

            # Backtrack loop - if the full step doesn't reduce the residual,
            # try smaller steps until we either reduce the residual or run out
            # of attempts
            for _ in range(max_backtracks):
                for i in range(n):
                    trial_state[i] = state[i] + scale * delta[i]
                # Residual function calculates guess - F(guess) - for example,
                # in a single backward Euler step, this is guess - step
                # start state - h*f(guess);
                residual_function(trial_state, parameters, drivers, h, linear_rhs, residual)
                norm2_new = 0.0
                for i in range(n):
                    ri = residual[i]
                    norm2_new += ri * ri

                if norm2_new <= tol_squared or norm2_new < norm2_prev:
                    for i in range(n):
                        state[i] = trial_state[i]
                    if norm2_new <= tol_squared:
                        return True
                    accepted = True
                    break
                scale *= damping
            return accepted

    return newton_krylov_solver