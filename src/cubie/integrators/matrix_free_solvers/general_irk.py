"""Placeholder general IRK-like solver factory."""

from numba import cuda


def general_irk_solver_factory():
    """Create a general implicit Runge--Kutta solver.

    Returns
    -------
    callable
        Device function executing a placeholder IRK step.
    """

    @cuda.jit(device=True)
    def general_irk_solver(*args):  # pragma: no cover - placeholder
        return 0

    return general_irk_solver
