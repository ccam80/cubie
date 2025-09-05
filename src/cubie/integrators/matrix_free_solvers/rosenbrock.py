r"""Rosenbrock linear solver device function factory."""

from typing import Callable

from numba import cuda


def rosenbrock_solver_factory(
    system, linear_solver: Callable, preconditioner: Callable | None = None
) -> Callable:
    """Create a Rosenbrock stage solver.

    Parameters
    ----------
    system : SymbolicODE
        System providing solver helpers.
    linear_solver : callable
        Device function solving ``(I - h \gamma J) x = rhs``.
    preconditioner : callable or None, optional
        Optional preconditioner device function.

    Returns
    -------
    callable
        CUDA device function computing the stage update.
    """

    i_minus_hj = system.get_solver_helper("i-hj")

    @cuda.jit(device=True)
    def rosenbrock_solver(
        state,
        parameters,
        drivers,
        h,
        rhs,
        x,
        residual,
        z_vec,
        v_vec,
        precond_temp,
    ) -> None:
        """Solve the Rosenbrock linear system for one stage."""

        linear_solver(
            i_minus_hj,
            state,
            parameters,
            drivers,
            h,
            rhs,
            x,
            residual,
            z_vec,
            v_vec,
            preconditioner,
            precond_temp,
        )

    return rosenbrock_solver
