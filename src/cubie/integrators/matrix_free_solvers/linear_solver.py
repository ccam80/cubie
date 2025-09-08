# python
"""Matrix-free preconditioned linear solver.


Implementation notes
--------------------
- Matrix-free: only operator_apply is required.
- Low memory: keeps a few vectors and fuses simple passes to reduce traffic.
- Preconditioner interface: preconditioner(state, parameters,
  drivers, h, residual, z, scratch) writes z ≈ M^{-1} r; if None, z := r.
  The solver passes a scratch vector that the preconditioner may overwrite;
  this buffer is then reused by the solver internally.

This module keeps function bodies small; each operation is factored into a helper.
"""

from typing import Callable, Optional

from numba import cuda, int32


def linear_solver_factory(
    operator_apply: Callable,
    n: int,
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
        M, J, h, beta, gamma should all be compiled-in to the operator_apply
        function.
    n : int
        length of the 1d residual/rhs vectors.
    preconditioner : callable(state, parameters, drivers, h, residual, z,
    jvp_scratch), optional, default=None
        Preconditioner function that approximately solves M z = residual,
        writing the result into z. The solver provides a scratch vector
        (\"jvp_scratch\") which the preconditioner may use and overwrite.
        If None, no preconditioning is applied and z is simply set to the residual.
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
             rhs, x, z, temp)
        where "temp" is also passed as a scratch buffer to the preconditioner.
    """
    # Setup compile-time flags to kill code branches
    SD = 1 if correction_type == "steepest_descent" else 0
    MR = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    PC = 1 if preconditioner is not None else 0

    # no cover: start
    @cuda.jit(device=True)
    def linear_solver(
        state, parameters, drivers, h,            # Operator context
        rhs,                                      # in:rhs, out: residual
        x,                                        # in: guess, out: solution
        z,                                        # working vector (pre. dir.)
        temp                                      # working vector (F z)
    ):
        """ Linear solver: precond. steepest descent or minimal residual.

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
            Right-hand side of the linear system. Updated in place with
            running residual; duplicate before calling to preserve rhs.
        x: array of floats
            On input: initial guess; on output: solution.
        z: array of floats
            Working array of size rhs.shape[0]. Holds preconditioned
            direction z.
        temp: array of floats
            Working array of size rhs.shape[0]. Holds operator_apply results;
            also passed as the scratch buffer to the preconditioner.

        Returns
        -------
        int
            0 on success; 3 if max linear iterations exceeded.

        Notes
        -----
        - Preconditioning, steepest descent vs minimal residual, and operator
          being applied are all configured in the factory.
        - rhs is overwritten with the current residual and maintained in place.
        - state, parameters, drivers are immutable inputs to operator_apply.
        - Scratch space required: 2 vectors of size rhs.shape[0]. The solver
          reuses its temporary vector ("temp") as the preconditioner scratch.


        """
        # Initial residual: r = rhs - F x
        operator_apply(state, parameters, drivers, h, x, temp)
        tol_squared = tolerance*tolerance

        acc = 0.0
        for i in range(n):
            # z := M^{-1} r (or copy)
            r = rhs[i] - temp[i]
            rhs[i] = r
            acc += r * r
        if acc <= tol_squared:
            return int32(0)

        for _ in range(max_iters):
            if PC:
                preconditioner(state, parameters, drivers, h, rhs,
                               z, temp)
            else:
                for i in range(n):
                    z[i] = rhs[i]

            # v = F z and line-search dot products
            operator_apply(state, parameters, drivers, h, z, temp)
            num = 0.0
            den = 0.0
            if SD:
                for i in range(n):
                    zi = z[i]
                    num += rhs[i] * zi  # (r·z)
                    den += temp[i] * zi  # (Fz·z)
            elif MR:
                for i in range(n):
                    ti = temp[i]
                    num += ti * rhs[i]      # (Fz·r)
                    den += ti * ti               # (Fz·Fz)

            alpha = cuda.selp(den != 0.0, num / den, 0.0)
            # Check convergence (norm of updated residual)
            acc = 0.0
            for i in range(n):
                x[i] += alpha * z[i]
                rhs[i] -= alpha * temp[i]
                ri = rhs[i]
                acc += ri * ri
            if acc <= tol_squared:
                return int32(0)

        return int32(3)
    # no cover: end
    return linear_solver
