"""Device function factories for Newton-Krylov solvers."""
from numba import cuda


def newton_krylov_solver_factory(tolerance, max_iters):
    """Create a Newton-Krylov solver device function.

    Parameters
    ----------
    tolerance : float
        Residual norm required for convergence.
    max_iters : int
        Maximum number of Newton iterations.

    Returns
    -------
    callable
        CUDA device function implementing a simple Newton-Krylov method.
    """

    @cuda.jit(device=True)
    def newton_krylov_solver(
        jvp_function,
        residual_function,
        state,
        residual,
        direction,
        temp_vec,
        jvp_temp,
        preconditioner,
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
                direction[i] = -residual[i]
            if preconditioner is not None:
                preconditioner(
                    jvp_function, state, direction, temp_vec, jvp_temp
                )
                for i in range(state.shape[0]):
                    state[i] += temp_vec[i]
            else:
                for i in range(state.shape[0]):
                    state[i] += direction[i]

    return newton_krylov_solver


def neumann_preconditioner_factory(order=1, stage_decoupled=False):
    """Create a Neumann polynomial preconditioner device function.

    Parameters
    ----------
    order : int, default=1
        Polynomial order (1 or 2).
    stage_decoupled : bool, default=False
        Treat the input vector as stage-decoupled.

    Returns
    -------
    callable
        CUDA device function implementing the preconditioner.
    """

    @cuda.jit(device=True)
    def neumann_preconditioner(
        jvp_function, state, vector, out, temp_vec
    ):
        """Approximate ``(I - J)^{-1}`` with a Neumann polynomial."""

        for i in range(vector.shape[0]):
            out[i] = vector[i]

        if stage_decoupled:
            stages = state.shape[0] // vector.shape[0]
            width = vector.shape[0]
            for _ in range(order):
                jvp_function(state, out, temp_vec)
                for i in range(width * stages):
                    out[i] += temp_vec[i]
        else:
            jvp_function(state, out, temp_vec)
            for i in range(vector.shape[0]):
                out[i] += temp_vec[i]
            if order == 2:
                jvp_function(state, temp_vec, vector)
                for i in range(vector.shape[0]):
                    out[i] += vector[i]

    return neumann_preconditioner

if __name__ == "__main__":
    from numba import float64
    from numba.cuda import jit

    @jit(
        (
            float64[:],
            float64[:],
            float64[:],
        ),
        device=True,
        inline=True,
    )
    def jvp_example(state, vector, out):
        for i in range(state.shape[0]):
            out[i] = state[i] * vector[i]

    @jit(
        (
            float64[:],
            float64[:],
        ),
        device=True,
        inline=True,
    )
    def residual_example(state, out):
        for i in range(state.shape[0]):
            out[i] = state[i] ** 2 - 1.0

    newton_solver = newton_krylov_solver_factory(tolerance=1e-6, max_iters=10)
    preconditioner = neumann_preconditioner_factory(order=2)

