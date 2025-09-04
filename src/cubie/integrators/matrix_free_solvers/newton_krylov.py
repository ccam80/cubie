"""Device function factories for Newton-Krylov solvers."""
from numba import cuda


def newton_krylov_solver_factory(tolerance, max_iters, linear_solver):
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
            residual_function(state, residual)
            norm = 0.0
            for i in range(residual.shape[0]):
                norm += residual[i] * residual[i]
            if norm ** 0.5 <= tolerance:
                return
            for i in range(residual.shape[0]):
                rhs[i] = -residual[i]
                delta[i] = 0.0
            linear_solver(
                jvp_function,
                state,
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


def neumann_preconditioner_factory(order=1):
    """Create a Neumann polynomial preconditioner device function.

    Parameters
    ----------
    order : int, default=1
        Polynomial order.

    Returns
    -------
    callable
        CUDA device function implementing the preconditioner.
    """

    @cuda.jit(device=True)
    def neumann_preconditioner(
        jvp_function, state, vector, out, temp_vec
    ):
        """Approximate ``(I - J)^{-1}`` with a Neumann polynomial.

        Assumes state, vector, out, and temp_vec are sized consistently for the
        Jacobian-vector product. The implementation performs:
        out = (I + J + J^2) vector for order=2, or out = (I + J) vector for order=1.
        """

        # Initialize output with input vector
        for i in range(out.shape[0]):
            out[i] = vector[i]

        # Add J^n.v terms to output vector
        for _ in range(order):
            jvp_function(state, out, temp_vec)
            for i in range(out.shape[0]):
                out[i] += temp_vec[i]

    return neumann_preconditioner