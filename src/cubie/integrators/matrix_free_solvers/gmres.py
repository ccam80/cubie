"""Device function factories for GMRES solvers."""
from numba import cuda


def gmres_solver_factory(tolerance, max_iters):
    """Create a GMRES solver device function.

    Parameters
    ----------
    tolerance : float
        Residual norm required for convergence.
    max_iters : int
        Dimension of the Krylov subspace.

    Returns
    -------
    callable
        CUDA device function implementing a simple GMRES method.
    """

    @cuda.jit(device=True)
    def gmres_solver(
        jvp_function,
        state,
        rhs,
        x,
        basis,
        hessenberg,
        givens,
        g_vec,
        w_vec,
    ):
        n = rhs.shape[0]

        jvp_function(state, x, w_vec)
        beta = 0.0
        for i in range(n):
            w_vec[i] = rhs[i] - w_vec[i]
            beta += w_vec[i] * w_vec[i]
        beta = beta ** 0.5
        if beta <= tolerance:
            return
        g_vec[0] = beta
        for i in range(n):
            basis[0, i] = w_vec[i] / beta

        for j in range(max_iters):
            jvp_function(state, basis[j], w_vec)
            for i in range(j + 1):
                h = 0.0
                for k in range(n):
                    h += basis[i, k] * w_vec[k]
                hessenberg[i, j] = h
                for k in range(n):
                    w_vec[k] -= h * basis[i, k]
            h_norm = 0.0
            for k in range(n):
                h_norm += w_vec[k] * w_vec[k]
            h_norm = h_norm ** 0.5
            hessenberg[j + 1, j] = h_norm
            if h_norm != 0.0:
                for k in range(n):
                    basis[j + 1, k] = w_vec[k] / h_norm
            else:
                for k in range(n):
                    basis[j + 1, k] = 0.0

            for i in range(j):
                c = givens[0, i]
                s = givens[1, i]
                temp = c * hessenberg[i, j] + s * hessenberg[i + 1, j]
                hessenberg[i + 1, j] = (
                    -s * hessenberg[i, j] + c * hessenberg[i + 1, j]
                )
                hessenberg[i, j] = temp

            h_ij = hessenberg[j, j]
            h_ip1j = hessenberg[j + 1, j]
            denom = (h_ij * h_ij + h_ip1j * h_ip1j) ** 0.5
            c = h_ij / denom if denom != 0.0 else 1.0
            s = h_ip1j / denom if denom != 0.0 else 0.0
            givens[0, j] = c
            givens[1, j] = s
            hessenberg[j, j] = c * h_ij + s * h_ip1j
            hessenberg[j + 1, j] = 0.0
            g_next = -s * g_vec[j]
            g_vec[j] = c * g_vec[j]
            g_vec[j + 1] = g_next
            if abs(g_vec[j + 1]) <= tolerance:
                last = j
                break
        else:
            last = max_iters - 1

        for k in range(last, -1, -1):
            val = g_vec[k]
            for m in range(k + 1, last + 1):
                val -= hessenberg[k, m] * g_vec[m]
            g_vec[k] = val / hessenberg[k, k]
        for col in range(last + 1):
            for i in range(n):
                x[i] += basis[col, i] * g_vec[col]

    return gmres_solver