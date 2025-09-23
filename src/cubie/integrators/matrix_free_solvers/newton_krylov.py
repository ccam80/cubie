"""Device function factory for Newton-Krylov solvers."""
from typing import Callable

from numba import cuda, int32, from_dtype
from numpy import float32 as np_float32
from cubie.cudasim_utils import activemask, all_sync


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision = np_float32,
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
    precision : np.float32 or np.float64, optional
        Floating-point data type to use. Default is np.float32.

    Returns
    -------
    callable
        CUDA device function implementing a damped Newton-Krylov method.

    Notes on return code
    --------------------
    The device function returns an int status whose lower 16 bits encode the
    outcome code:
      0: success
      1: no suitable step found (backtracking failed)
      2: max Newton iterations exceeded
      3: inner linear solver did not converge (propagated)
    The upper 16 bits contain the number of Newton iterations completed.
    """
    tol_squared = precision(tolerance*tolerance)
    precision = from_dtype(precision)
    typed_zero = precision(0.0)
    typed_one = precision(1.0)
    typed_damping = precision(damping)
    status_start = int32(-1)
    #no cover: start
    @cuda.jit(device=True, inline=True)
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
            Additional workspace for linear solver.

        Returns
        -------
        Int (0=success, 1=no step found, 2=max iterations reached, 3=linear solver failed)

        Notes
        -----
        - Scratch space required: 4*n where n = len(state).
        - The linear solver is expected to solve J*delta = rhs where rhs
          is provided in-place in the residual array, and delta is used
          as an initial guess and returns the solution in-place.
        - The state is updated in-place and reverted if no acceptable step found.
        """
        # Build initial rhs = -F(state) and norm in one pass

        residual_function(state,
                          parameters,
                          drivers,
                          h,
                          a_ij,
                          base_state,
                          residual)
        norm2_prev = typed_zero
        for i in range(n):
            ri = residual[i]
            residual[i] = -ri
            delta[i] = typed_zero
            norm2_prev += ri * ri

        # Sticky per-lane status: -1 active, 0 success, 1 no step, 2 max iters,
        # 3 inner fail
        status = status_start
        if norm2_prev <= tol_squared:
            status = int32(0)

        iters_count = int32(0)
        mask = activemask()
        for iters in range(max_iters):
            # Warp-coherent exit if all lanes done
            if all_sync(mask, status >= 0):
                break

            iters_count += int32(1)
            # Unavoidable branch - solver modifies residual, delta in-place,
            # which we don't want for already-converged guesses.
            if status < 0:
                # Solve J * delta = rhs  (rhs currently holds -F(state))
                lin_return = linear_solver(
                    state, parameters, drivers, h,
                    residual,          # in: rhs, out: linear residual
                    delta,             # in: initial guess out: Newton direction
                    preconditioned_vec,
                    work_vec,
                )
                if lin_return != int32(0):
                    status = int32(lin_return)

            # Backtrack loop - if the full step doesn't reduce the residual,
            # try smaller steps until we either reduce the residual or run out
            # of attempts
            scale = typed_one
            s_applied = typed_zero
            found_step = False

            for _bt in range(max_backtracks + 1):
                # Add difference in step size since last attempt
                if status < 0:

                    coeff = scale - s_applied
                    for i in range(n):
                        state[i] += coeff * delta[i]
                    s_applied = scale

                    # Evaluate residual at tentative state
                    residual_function(state,
                                      parameters,
                                      drivers,
                                      h,
                                      a_ij,
                                      base_state,
                                      residual)

                    norm2_new = typed_zero
                    for i in range(n):
                        ri = residual[i]
                        norm2_new += ri * ri

                    if norm2_new <= tol_squared:
                        status = int32(0)

                    accept =  (status < 0) and (norm2_new < norm2_prev)
                    found_step = found_step or accept

                    # Prepare rhs and norm for next iterate ONLY if accepted
                    for i in range(n):
                        residual[i] = cuda.selp(accept,
                                                -residual[i],
                                                residual[i])
                    norm2_prev = cuda.selp(accept, norm2_new, norm2_prev)

                # Warp-coherent break when all lanes have accepted
                if all_sync(mask, (found_step or status >= 0)):
                    break
                # Not accepted yet; try again with a smaller scale.
                scale *= typed_damping

            # Backtrack exhausted without a step → revert and mark status=1
            if (status < 0) and (not found_step):
                for i in range(n):
                    state[i] -= s_applied * delta[i]
                status = int32(1)
        if status < 0:
            # Lanes still active after max_iters
            status = int32(2)
        status |= (iters_count + 1) << 16
        return status
    # no cover: end
    return newton_krylov_solver
