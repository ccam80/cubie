"""Placeholder Radau IIA 5 solver factory."""

from numba import cuda


def radauiia5_solver_factory():
    """Create a Radau IIA order 5 solver.

    Returns
    -------
    callable
        Device function performing a placeholder Radau IIA 5 step.
    """

    @cuda.jit(device=True)
    def radauiia5_solver(*args):  # pragma: no cover - placeholder
        return 0

    return radauiia5_solver
