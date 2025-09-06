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
        Device function solving ``(I - h*gamma*J) x = rhs``.
    preconditioner : callable or None, optional
        Optional preconditioner device function.

    Returns
    -------
    callable
        CUDA device function computing the stage update.
    """

    @cuda.jit(device=True)
    def rosenbrock_solver(
        state,
        parameters,
        drivers,
        h,
        rhs,
        x,
        z_vec,
        temp,
    ) -> None:
        """Solve the Rosenbrock linear system for one stage."""

        linear_solver(
            state,
            parameters,
            drivers,
            h,
            rhs,
            x,
            z_vec,
            temp,
        )

    return rosenbrock_solver
