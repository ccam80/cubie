"""Device function factory for Newton-Krylov solvers."""
from typing import Callable

from numba import cuda, int32


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
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
    n : int
        Size of 1d system state/rhs vector
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

    Notes on return code
    --------------------
    The device function returns an int status:
      0: success
      1: no suitable step found (backtracking failed)
      2: max Newton iterations exceeded
      3: inner linear solver did not converge (propagated)
    """
    tol_squared = tolerance*tolerance

    #no cover: start
    @cuda.jit(device=True)
    def newton_krylov_solver(
        state,
        parameters,
        drivers,
        h,
        a_ij,
        base_state,
        delta,
        residual,
        preconditioned_vec,
        work_vec,
    ):
        """ Damped Newton-Krylov solver.

        Parameters
        ----------
        state: device array
            On entry, the initial guess; on exit, the solution if
            convergence was achieved.
        parameters: device array
            Model parameters.
        drivers: device array
            Model drivers.
        h: float
            Timestep size (if applicable).
        a_ij: float
            stage weight (if multi-stage; set to 1.0 for single-stage).
        base_state: device array
            Base state for residual evaluation (e.g., previous step y_n).
        delta: device array
            Workspace for Newton step.
        residual: device array
            Workspace for residual evaluation.
        preconditioned_vec: device array
            Workspace for preconditioned vector in linear solver.
        work_vec: device array
            Workspace for residual evaluation.

        Returns
        -------
        Int (0=success, 1=no step found, 2=max iterations reached, 3=linear solver failed)

        Notes
        -----
        - Scratch space required: 4*n where n = len(state).
        - The linear solver is expected to solve J*delta = rhs where rhs
          is provided in-place in the residual array, and delta is used
          as an initial guess and returns the solution in-place.
        - The residual function is expected to compute F(state) in-place
            in the residual array.
        - The state is updated in-place and reverted if no acceptable
        """
        # Build initial rhs = -F(state) and norm in one pass
        residual_function(state, parameters, drivers, h, a_ij, base_state,
                          work_vec, residual)
        norm2_prev = 0.0
        for i in range(n):
            ri = residual[i]
            residual[i] = -ri
            delta[i] = 0.0
            norm2_prev += ri * ri
        if norm2_prev <= tol_squared:
            return int32(0)

        for _ in range(max_iters):
            # Solve J * delta = rhs  (rhs currently holds -F(state))
            lin_return = linear_solver(
                state, parameters, drivers, h,
                residual,          # in: rhs, out: linear residual
                delta,             # in: initial guess out: Newton direction
                preconditioned_vec,
                work_vec,
            )
            if lin_return != int32(0): #is there a way to avoid this branch?
                return lin_return # 3: max linear iters

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
                residual_function(state, parameters, drivers, h, a_ij,
                                  base_state, work_vec, residual)
                norm2_new = 0.0
                for i in range(n):
                    ri = residual[i]
                    norm2_new += ri * ri

                if norm2_new <= tol_squared:
                    return int32(0)
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
                return int32(1)
            # Accepted but not converged; fall through with prepared rhs
        return int32(2)  # Newton iterations exhausted

    # no cover: end
    return newton_krylov_solver
