"""Device function factory for minimal residual solvers."""
from numba import cuda


def minimal_residual_solver_factory(tolerance, max_iters):
    """Create a minimal residual solver device function.

    This solver applies preconditioned steepest descent to
    minimise the residual without storing a basis or
    Hessenberg matrix. It therefore uses very little memory
    but can converge slowly and is less robust than GMRES on
    ill conditioned or strongly nonsymmetric systems.
    """

    @cuda.jit(device=True)
    def minimal_residual_solver(
        jvp_function,
        state,
        rhs,
        x,
        residual,
        z_vec,
        v_vec,
        preconditioner,
        precond_temp,
    ):
        for _ in range(max_iters):
            jvp_function(state, x, v_vec)
            norm = 0.0
            for i in range(rhs.shape[0]):
                residual[i] = rhs[i] - v_vec[i]
                norm += residual[i] * residual[i]
            if norm ** 0.5 <= tolerance:
                return
            if preconditioner is not None:
                preconditioner(
                    jvp_function, state, residual, z_vec, precond_temp
                )
            else:
                for i in range(rhs.shape[0]):
                    z_vec[i] = residual[i]
            jvp_function(state, z_vec, v_vec)
            rz = 0.0
            vz = 0.0
            for i in range(rhs.shape[0]):
                rz += residual[i] * z_vec[i]
                vz += v_vec[i] * z_vec[i]
            alpha = rz / vz if vz != 0.0 else 0.0
            for i in range(rhs.shape[0]):
                x[i] += alpha * z_vec[i]

    return minimal_residual_solver