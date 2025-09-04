"""Placeholder Rosenbrock solver device function factory."""

from numba import cuda


def rosenbrock_solver_factory():
    """Create a Rosenbrock solver device function.

    Returns
    -------
    callable
        Device function implementing a single Rosenbrock iteration.
    """

    @cuda.jit(device=True)
    def rosenbrock_solver(*args):  # pragma: no cover - placeholder
        return 0

    return rosenbrock_solver
