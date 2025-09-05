# python
"""Matrix-free preconditioned linear solver.


Implementation notes
--------------------
- Matrix-free: only operator_apply is required.
- Low memory: keeps a few vectors and fuses simple passes to reduce traffic.
- Preconditioner interface: preconditioner(operator_apply, state, parameters,
  drivers, h, residual, z, scratch) writes z ≈ M^{-1} r; if None, z := r.

This module keeps function bodies small; each operation is factored into a helper.
"""

from typing import Callable, Optional

from numba import cuda


def linear_solver_factory(
    operator_apply: Callable,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "steepest_descent",
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> Callable:
    """Create a CUDA device function implementing preconditioned SD/MR.

    Parameters
    ----------
    operator_apply : callable(state, parameters, drivers, h, in_vec, out_vec):
        applies the linear operator F to 'in_vec', writing into 'out_vec'.
        state, parameters, drivers, and h are input parameters that are used
        to evaluate the Jacobian at the current guess.
        Generally, this operator is of the form F = β M - γ h J, where:
        - M is a mass matrix (Identity for standard ODEs)
        - J is the system Jacobian
        - h is the timestep
        - β and γ are scalars (beta is a "shift" to improve conditioning in
            e.g. Radau methods; and gamma is a stage parameter for e.g.
            Rosenbrock or IRK methods).
        In the simplest case ODE integrator, backward-Euler, F ≈ I − h J at
        the current guess.
        M, J, h, B, gamma should all be compiled-in to the operator_apply
        function.
    preconditioner : callable(state, parameters, drivers, h, residual, z),
    optional, default=None
        Preconditioner function that approximately solves M z = residual,
        writing the result into z. If None, no preconditioning is applied and
        z is simply set to the residual.
    correction_type : str
        Type of line search to perform. These affect the calculation of the
        correction step length alpha:
        - "steepest_descent": choose alpha to eliminate the component of the
            residual along the search direction z. This is effective when F is
            close to symmetric and damped (e.g., diffusion-dominated, small h).
        - "minimal residual": choose alpha to guarantee that the residual
        norm decreases. Effective for strongly nonsymmetric or indefinite
        problems, but can take longer to converge for simple systems.
    tolerance : float
        Target residual 2-norm for convergence.
    max_iters : int
        Maximum iteration count.

    Returns
    -------
    callable
        CUDA device function with signature:
        solver(state, parameters, drivers, h,
             rhs, x, residual, z_vec, v_vec)
    """
    # Setup compile-time flags to kill code branches
    SD = 1 if correction_type == "steepest_descent" else 0
    MR = 1 if correction_type == "minimal residual" else 0
    PC = 1 if preconditioner is not None else 0

    @cuda.jit(device=True)
    def linear_solver(
        state, parameters, drivers, h,            # Operator context
        rhs,                                      # rhs of linear system
        x,                                        # in: guess, out: solution
        residual, preconditioned, temp         # working vectors
    ):
        """ Linear solver: precond. steepest descent or minimal residual.

        Preconditioning, steepest descent vs minimal residual, and operator
        being applied are all configured in the factory.

        Parameters
        ----------
        state: array of floats
            Input parameter for evaluating the Jacobian in the operator.
        parameters: array of floats
            Input parameter for evaluating the Jacobian in the operator.
        drivers: array of floats
            Input parameter for evaluating the Jacobian in the operator.
        h: float
            Step size - set by outer solver, used in operator_apply.
        rhs: array of floats
            Right-hand side of the linear system.
        x: array of floats
            On input: initial guess; on output: solution.
        residual: array of floats
            Working array of size rhs.shape[0]. Holds current residual.
        preconditioned: array of floats
            Working array of size rhs.shape[0]. Holds preconditioner results.
        temp: array of floats
            Working array of size rhs.shape[0]. Holds operator_apply results.
        """
        n = rhs.shape[0]

        # Initial residual: r = rhs - F x
        operator_apply(state, parameters, drivers, h, x, temp)
        acc = 0.0
        for i in range(n):
            # z := M^{-1} r (or copy)
            r = rhs[i] - temp[i]
            residual[i] = r
            acc += r * r

        rnorm = acc ** 0.5
        if rnorm <= tolerance:
            return

        for _ in range(max_iters):
            if PC:
                preconditioner(state, parameters, drivers, h, residual, preconditioned)
            else:
                for i in range(n):
                    preconditioned[i] = residual[i]

            # v = F z and line-search dot products
            operator_apply(state, parameters, drivers, h, preconditioned, temp)
            num = 0.0
            den = 0.0
            if SD:
                for i in range(n):
                    zi = preconditioned[i]
                    num += residual[i] * zi  # (r·z)
                    den += temp[i] * zi  # (Fz·z)
            elif MR:
                for i in range(n):
                    ti = temp[i]
                    num += ti * residual[i]      # (Fz·r)
                    den += ti * ti               # (Fz·Fz)

            alpha =  cuda.selp(vz != 0, rz / vz, 0.0) # noqa
            # Check convergence (norm of updated residual)
            acc = 0.0
            for i in range(n):
                x[i] += alpha * preconditioned[i]
                residual[i] -= alpha * temp[i]
                ri = residual[i]
                acc += ri * ri
            rnorm = acc ** 0.5
            if rnorm <= tolerance:
                return

    return linear_solver


def neumann_preconditioner_factory(neumann_operator, order: int = 1) -> Callable:
    """Create a Neumann polynomial preconditioner device function.

    Parameters
    ----------
    neumann_operator: callable(state, parameters, drivers, h, residual,
    out)
        Device function applying [out = r + h*J*out]
    order : int, default=1
        Polynomial order.

    Returns
    -------
    callable
        CUDA device function implementing the preconditioner.
    """

    @cuda.jit(device=True)
    def neumann_preconditioner(
        state, parameters, drivers, h, residual, out
    ):
        """Approximate ``(I - J)^{-1}`` with a Neumann polynomial.

        The implementation performs:

        ``out = residual + h*J*residual + (h*J)^2*residual (for order=2)``.

        Parameters
        ----------
        state: array of floats
            Input parameter for evaluating the Jacobian in the operator
        parameters: array of floats
            Input parameter for evaluating the Jacobian in the operator
        drivers: array of floats
            Input parameter for evaluating the Jacobian in the operator
        h: float
            Step size - set by outer solver, used in neumann_operator
        residual: array of floats
            Input residual vector
        out: array of floats
            Output vector, to be overwritten with the preconditioned result
        """

        for i in range(out.shape[0]):
            out[i] = residual[i]

        for _ in range(order):
            neumann_operator(state, parameters, drivers, h, residual, out)

    return neumann_preconditioner
