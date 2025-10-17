"""Shared utilities for CPU reference integrator implementations."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from cubie.integrators import IntegratorReturnCodes


Array = NDArray[np.floating]
STATUS_MASK = 0xFFFF


def _ensure_array(vector: Sequence[float] | Array, dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.atleast_1d(vector).astype(dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


def _neumann_preconditioner_matrix(
    operator_apply: Callable[[Array], Array],
    dimension: int,
    precision: np.dtype,
    order: int,
) -> Array:
    """Return the truncated Neumann series for ``operator_apply``."""

    identity = np.eye(dimension, dtype=precision)
    operator_matrix = np.zeros_like(identity)
    for column in range(dimension):
        basis = identity[:, column].copy()
        operator_matrix[:, column] = np.asarray(
            operator_apply(basis),
            dtype=precision,
        )

    neumann = identity.copy()
    if order <= 0:
        return neumann

    residual = identity - operator_matrix
    power = identity.copy()
    for _ in range(order):
        power = power @ residual
        neumann = neumann + power
    return neumann


@dataclass
class StepResult:
    """Container describing the outcome of a single integration step."""

    state: Array
    observables: Array
    error: Array
    status: int = 0
    niters: int = 0


def _encode_solver_status(converged: bool, niters: int) -> int:
    """Return a solver status word with the Newton iteration count encoded."""

    base_code = (
        IntegratorReturnCodes.SUCCESS
        if converged
        else IntegratorReturnCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
    )
    iter_count = max(0, min(int(niters), STATUS_MASK))
    return (iter_count << 16) | (int(base_code) & STATUS_MASK)


class DriverEvaluator:
    """Evaluate spline coefficients to recover driver samples on the CPU."""

    def __init__(
        self,
        coefficients: Optional[Array],
        dt: float,
        t0: float,
        wrap: bool,
        precision: np.dtype,
        *,
        boundary_condition: Optional[str] = "not-a-knot",
    ) -> None:
        coeffs = (
            np.zeros((0, 0, 1), dtype=precision)
            if coefficients is None
            else np.array(coefficients, dtype=precision, copy=True)
        )
        if coeffs.ndim != 3:
            raise ValueError(
                "Driver coefficients must have shape (segments, drivers, "
                "order + 1)."
            )
        self.precision = precision
        self.coefficients = coeffs
        self.dt = precision(dt)
        self.t0 = precision(t0)
        self.wrap = bool(wrap)
        self.boundary_condition = boundary_condition
        self._segments = coeffs.shape[0]
        self._width = coeffs.shape[1]
        self._zero = np.zeros(self._width, dtype=precision)
        pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        self._evaluation_start = self.t0 - (self.dt if pad_clamped else 0.0)

    def evaluate(self, time: float) -> Array:
        """Return driver values interpolated at ``time``."""

        precision = self.precision

        if self._width == 0 or self._segments == 0 or self.dt == 0.0:
            return self._zero.copy()

        inv_res = 1.0 / self.dt
        scaled = (time - self._evaluation_start) * inv_res
        scaled_floor = math.floor(scaled)
        idx = int(scaled_floor)

        if self.wrap and self._segments > 0:
            segment = idx % self.coefficients.shape[0]
            if segment < 0:
                segment += self.coefficients.shape[0]
            tau = scaled - scaled_floor
            in_range = True
        else:
            in_range = 0.0 <= scaled <= precision(self.coefficients.shape[0])
            segment = idx if idx >= 0 else 0
            if segment >= self.coefficients.shape[0]:
                segment = self.coefficients.shape[0] - 1
            tau = scaled - precision(segment)

        values = self._zero.copy()
        for driver_idx in range(self._width):
            segment_coeffs = self.coefficients[segment, driver_idx]
            acc = precision(0.0)
            for coeff in reversed(segment_coeffs):
                acc = acc * tau + precision(coeff)
            if self.wrap or in_range:
                values[driver_idx] = acc
            else:
                values = self._zero.copy()
        return values

    def __call__(self, time: float) -> Array:
        """Alias for :meth:`evaluate` so instances are callable."""

        return self.evaluate(time)

    def with_coefficients(self, coefficients: Optional[Array]) -> "DriverEvaluator":
        """Return a new evaluator with ``coefficients`` but shared timing."""

        return DriverEvaluator(
            coefficients=coefficients,
            dt=self.dt,
            t0=self.t0,
            wrap=self.wrap,
            precision=self.precision,
            boundary_condition=self.boundary_condition,
        )


def krylov_solve(
    rhs: Array,
    *,
    operator_apply: Callable[[Array], Array],
    tolerance: np.floating,
    max_iterations: int,
    precision: np.dtype,
    preconditioner: Optional[Callable[[Array], Array]] = None,
    neumann_order: int = 2,
    correction_type: str = "minimal_residual",
    initial_guess: Optional[Array] = None,
) -> tuple[Array, bool, int]:
    """Solve ``operator_apply(x) = rhs`` using matrix-free Krylov iterations.

    Parameters
    ----------
    rhs
        Right-hand side vector.
    operator_apply
        Callable returning the operator applied to its argument.
    tolerance
        Convergence tolerance on the residual norm.
    max_iterations
        Maximum iteration count for the descent loop. Zero is permitted.
    precision
        Floating-point precision to use for the iteration.
    preconditioner
        Optional callable returning a preconditioned residual direction. If
        ``None`` the truncated Neumann series of order ``neumann_order`` is
        used.
    neumann_order
        Order of the truncated Neumann series used when ``preconditioner`` is
        ``None``.
    correction_type
        Descent update to apply. ``"steepest_descent"`` or
        ``"minimal_residual"``.
    initial_guess
        Optional starting iterate for the solve. Defaults to the zero vector.

    Returns
    -------
    tuple[Array, bool, int]
        Solution vector, convergence flag, and iteration count.
    """

    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )

    vector = np.asarray(rhs, dtype=precision)

    if initial_guess is None:
        solution = np.zeros_like(vector)
    else:
        solution = np.asarray(initial_guess, dtype=precision)

    tol_value = precision(tolerance)
    tol_squared = tol_value * tol_value
    iteration_limit = int(max_iterations)
    if iteration_limit < 0:
        raise ValueError("Maximum iterations must be non-negative.")

    applied = np.asarray(operator_apply(solution), dtype=precision)
    residual = vector - applied
    residual_squared = float(np.dot(residual, residual))
    if residual_squared <= tol_squared:
        return solution, True, 0

    if preconditioner is None:
        neumann_matrix = _neumann_preconditioner_matrix(
            operator_apply,
            vector.shape[0],
            precision,
            int(neumann_order),
        )

        def default_preconditioner(vector: Array) -> Array:
            return neumann_matrix @ vector

        preconditioner_fn = default_preconditioner
    else:
        preconditioner_fn = preconditioner

    converged = False
    iteration = 0
    while iteration < iteration_limit:
        iteration += 1
        direction = np.asarray(preconditioner_fn(residual), dtype=precision)

        operator_direction = np.asarray(
            operator_apply(direction),
            dtype=precision,
        )

        if correction_type == "steepest_descent":
            numerator = float(np.dot(residual, direction))
            denominator = float(np.dot(operator_direction, direction))
        else:
            numerator = float(np.dot(operator_direction, residual))
            denominator = float(np.dot(operator_direction, operator_direction))

        if denominator == 0.0:
            return solution, False, iteration

        alpha = precision(numerator / denominator)
        if converged:
            alpha = precision(0.0)

        solution = solution + alpha * direction
        residual = residual - alpha * operator_direction
        residual_squared = float(np.dot(residual, residual))
        if residual_squared <= tol_squared:
            converged = True
            break

    return solution, converged, iteration


__all__ = [
    "Array",
    "DriverEvaluator",
    "STATUS_MASK",
    "StepResult",
    "_encode_solver_status",
    "_ensure_array",
    "krylov_solve",
]
