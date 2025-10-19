"""Shared utilities for CPU reference integrator implementations."""

import math
from dataclasses import dataclass, field
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


def dot_product(
    left: Sequence[float] | Array,
    right: Sequence[float] | Array,
    *,
    precision: np.dtype,
) -> np.floating:
    """Return the dot product of ``left`` and ``right`` in ``precision``."""

    left_vector = np.asarray(left, dtype=precision)
    right_vector = np.asarray(right, dtype=precision)
    product = np.multiply(left_vector, right_vector, dtype=precision)
    return product.sum(dtype=precision)


def squared_norm(vector: Sequence[float] | Array, *, precision: np.dtype) -> np.floating:
    """Return the squared Euclidean norm of ``vector`` in ``precision``."""

    typed = np.asarray(vector, dtype=precision)
    squared = np.multiply(typed, typed, dtype=precision)
    return squared.sum(dtype=precision)


def euclidean_norm(vector: Sequence[float] | Array, *, precision: np.dtype) -> np.floating:
    """Return the Euclidean norm of ``vector`` in ``precision``."""

    squared = squared_norm(vector, precision=precision)
    return np.sqrt(squared, dtype=precision)


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


@dataclass(slots=True)
class InstrumentedStepResult:
    """Return container with per-stage diagnostics for a CPU step."""

    state: Array
    observables: Array
    error: Array
    residuals: Array
    jacobian_updates: Array
    status: int = 0
    niters: int = 0
    stage_count: int = 0
    stage_states: Array | None = None
    stage_derivatives: Array | None = None
    stage_observables: Array | None = None
    solver_initial_guesses: Array | None = None
    solver_solutions: Array | None = None
    solver_iterations: NDArray[np.int_] | None = None
    solver_status: NDArray[np.int_] | None = None
    extra_vectors: dict[str, Array] = field(default_factory=dict)


StepResultLike = StepResult | InstrumentedStepResult


def _coerce_stage_array(
    data: Optional[Array],
    stage_count: int,
    vector_dim: int,
    dtype: np.dtype,
) -> Array:
    """Return ``data`` as a per-stage array matching ``stage_count``."""

    if stage_count <= 0:
        if data is None:
            return np.zeros((0, vector_dim), dtype=dtype)
        shaped = np.asarray(data, dtype=dtype)
        if shaped.shape != (0, vector_dim):
            raise ValueError(
                "Per-stage arrays must have shape (stage_count, state_dim)."
            )
        return shaped
    if data is None:
        return np.zeros((stage_count, vector_dim), dtype=dtype)
    shaped = np.asarray(data, dtype=dtype)
    if shaped.shape != (stage_count, vector_dim):
        raise ValueError(
            "Per-stage arrays must have shape (stage_count, state_dim)."
        )
    return shaped


def _coerce_stage_vector(
    data: Optional[Sequence[int] | NDArray[np.int_]],
    stage_count: int,
    dtype: np.dtype,
) -> NDArray[np.int_]:
    """Return a per-stage integer vector with length ``stage_count``."""

    if stage_count <= 0:
        if data is None:
            return np.zeros(0, dtype=dtype)
        shaped = np.asarray(data, dtype=dtype)
        if shaped.shape != (0,):
            raise ValueError("Per-stage vectors must have length stage_count.")
        return shaped
    if data is None:
        return np.zeros(stage_count, dtype=dtype)
    shaped = np.asarray(data, dtype=dtype)
    if shaped.shape != (stage_count,):
        raise ValueError("Per-stage vectors must have length stage_count.")
    return shaped


def make_step_result(
    *,
    instrument: bool,
    state: Array,
    observables: Array,
    error: Array,
    status: int,
    niters: int,
    stage_count: Optional[int] = None,
    residuals: Optional[Array] = None,
    jacobian_updates: Optional[Array] = None,
    stage_states: Optional[Array] = None,
    stage_derivatives: Optional[Array] = None,
    stage_observables: Optional[Array] = None,
    solver_initial_guesses: Optional[Array] = None,
    solver_solutions: Optional[Array] = None,
    solver_iterations: Optional[Sequence[int] | NDArray[np.int_]] = None,
    solver_status: Optional[Sequence[int] | NDArray[np.int_]] = None,
    extra_vectors: Optional[dict[str, Array]] = None,
) -> StepResultLike:
    """Return a step result container with optional instrumentation."""

    if not instrument:
        return StepResult(state, observables, error, status, niters)

    resolved_stage_count = int(stage_count or 0)
    if resolved_stage_count < 0:
        raise ValueError("Stage count must be non-negative when instrumenting.")
    state_dim = int(state.shape[0])
    observable_dim = int(observables.shape[0])
    dtype = state.dtype

    residual_array = _coerce_stage_array(
        residuals, resolved_stage_count, state_dim, dtype
    )
    jacobian_array = _coerce_stage_array(
        jacobian_updates, resolved_stage_count, state_dim, dtype
    )
    stage_state_array = _coerce_stage_array(
        stage_states, resolved_stage_count, state_dim, dtype
    )
    stage_derivative_array = _coerce_stage_array(
        stage_derivatives, resolved_stage_count, state_dim, dtype
    )
    stage_observable_array = _coerce_stage_array(
        stage_observables,
        resolved_stage_count,
        observable_dim,
        observables.dtype,
    )
    solver_guess_array = _coerce_stage_array(
        solver_initial_guesses, resolved_stage_count, state_dim, dtype
    )
    solver_solution_array = _coerce_stage_array(
        solver_solutions, resolved_stage_count, state_dim, dtype
    )
    solver_iteration_array = _coerce_stage_vector(
        solver_iterations,
        resolved_stage_count,
        np.dtype(np.int64),
    )
    solver_status_array = _coerce_stage_vector(
        solver_status,
        resolved_stage_count,
        np.dtype(np.int64),
    )

    extras: dict[str, Array] = {}
    if extra_vectors is not None:
        for name, values in extra_vectors.items():
            extras[name] = _coerce_stage_array(
                values, resolved_stage_count, state_dim, dtype
            )

    return InstrumentedStepResult(
        state=state,
        observables=observables,
        error=error,
        residuals=residual_array,
        jacobian_updates=jacobian_array,
        status=status,
        niters=niters,
        stage_count=resolved_stage_count,
        stage_states=stage_state_array,
        stage_derivatives=stage_derivative_array,
        stage_observables=stage_observable_array,
        solver_initial_guesses=solver_guess_array,
        solver_solutions=solver_solution_array,
        solver_iterations=solver_iteration_array,
        solver_status=solver_status_array,
        extra_vectors=extras,
    )


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

        inv_res = precision(precision(1.0) / self.dt)
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
    residual_squared = dot_product(residual, residual, precision=precision)
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
            numerator = dot_product(
                residual,
                direction,
                precision=precision,
            )
            denominator = dot_product(
                operator_direction,
                direction,
                precision=precision,
            )
        else:
            numerator = dot_product(
                operator_direction,
                residual,
                precision=precision,
            )
            denominator = dot_product(
                operator_direction,
                operator_direction,
                precision=precision,
            )

        if denominator == 0.0:
            return solution, False, iteration

        alpha = precision(numerator / denominator)
        if converged:
            alpha = precision(0.0)

        solution = solution + alpha * direction
        residual = residual - alpha * operator_direction
        residual_squared = dot_product(
            residual,
            residual,
            precision=precision,
        )
        if residual_squared <= tol_squared:
            converged = True
            break

    return solution, converged, iteration


__all__ = [
    "Array",
    "DriverEvaluator",
    "STATUS_MASK",
    "dot_product",
    "euclidean_norm",
    "StepResult",
    "_encode_solver_status",
    "_ensure_array",
    "squared_norm",
    "krylov_solve",
]
