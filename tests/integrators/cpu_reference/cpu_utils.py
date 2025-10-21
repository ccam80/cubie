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
    stage_drivers: Array
    stage_increments: Array
    status: int = 0
    niters: int = 0
    stage_count: int = 0
    stage_states: Array | None = None
    stage_derivatives: Array | None = None
    stage_observables: Array | None = None
    newton_initial_guesses: Array | None = None
    newton_iteration_guesses: Array | None = None
    newton_residuals: Array | None = None
    newton_squared_norms: Array | None = None
    newton_iteration_scale: Array | None = None
    linear_initial_guesses: Array | None = None
    linear_iteration_guesses: Array | None = None
    linear_residuals: Array | None = None
    linear_squared_norms: Array | None = None
    linear_preconditioned_vectors: Array | None = None
    extra_vectors: dict[str, Array] = field(default_factory=dict)


StepResultLike = StepResult | InstrumentedStepResult


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
    stage_drivers: Optional[Array] = None,
    stage_increments: Optional[Array] = None,
    newton_initial_guesses: Optional[Array] = None,
    newton_iteration_guesses: Optional[Array] = None,
    newton_residuals: Optional[Array] = None,
    newton_squared_norms: Optional[Array] = None,
    newton_iteration_scale: Optional[Array] = None,
    linear_initial_guesses: Optional[Array] = None,
    linear_iteration_guesses: Optional[Array] = None,
    linear_residuals: Optional[Array] = None,
    linear_squared_norms: Optional[Array] = None,
    linear_preconditioned_vectors: Optional[Array] = None,
    extra_vectors: Optional[dict[str, Array]] = None,
) -> StepResultLike:
    """Return a step result container with optional instrumentation."""

    if not instrument:
        return StepResult(state, observables, error, status, niters)

    resolved_stage_count = int(stage_count or 0)
    extras = {} if extra_vectors is None else dict(extra_vectors)

    return InstrumentedStepResult(
        state=state,
        observables=observables,
        error=error,
        residuals=residuals,
        jacobian_updates=jacobian_updates,
        stage_drivers=stage_drivers,
        stage_increments=stage_increments,
        status=status,
        niters=niters,
        stage_count=resolved_stage_count,
        stage_states=stage_states,
        stage_derivatives=stage_derivatives,
        stage_observables=stage_observables,
        newton_initial_guesses=newton_initial_guesses,
        newton_iteration_guesses=newton_iteration_guesses,
        newton_residuals=newton_residuals,
        newton_squared_norms=newton_squared_norms,
        newton_iteration_scale=newton_iteration_scale,
        linear_initial_guesses=linear_initial_guesses,
        linear_iteration_guesses=linear_iteration_guesses,
        linear_residuals=linear_residuals,
        linear_squared_norms=linear_squared_norms,
        linear_preconditioned_vectors=linear_preconditioned_vectors,
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
        self._zero_value = precision(0.0)
        self._pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        self._inv_dt = precision(precision(1.0) / self.dt)
        offset = self.dt if self._pad_clamped else self._zero_value
        self._evaluation_start = precision(self.t0 - offset)

    def evaluate(self, time: float) -> Array:
        """Return driver values interpolated at ``time``."""

        precision = self.precision

        time_value = precision(time)
        scaled = precision(
            (time_value - self._evaluation_start) * self._inv_dt
        )
        scaled_floor = math.floor(float(scaled))
        idx = int(scaled_floor)

        if self.wrap:
            segment = idx % self._segments
            if segment < 0:
                segment += self._segments
            tau = precision(scaled - precision(scaled_floor))
            in_range = True
        else:
            max_segment = self._segments - 1
            in_range = (
                scaled >= self._zero_value
                and scaled <= precision(self._segments)
            )
            if idx < 0:
                segment = 0
            elif idx >= self._segments:
                segment = max_segment
            else:
                segment = idx
            tau = precision(scaled - precision(segment))

        values = self._zero.copy()
        for driver_idx in range(self._width):
            segment_coeffs = self.coefficients[segment, driver_idx]
            acc = self._zero_value
            for coeff in reversed(segment_coeffs):
                acc = acc * tau + precision(coeff)
            values[driver_idx] = acc

        if self.wrap or in_range:
            return values
        return self._zero.copy()

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
    instrumented: bool = False,
    logging_initial_guess: Optional[Array] = None,
    logging_iteration_guesses: Optional[Array] = None,
    logging_residuals: Optional[Array] = None,
    logging_squared_norms: Optional[Array] = None,
    logging_preconditioned_vectors: Optional[Array] = None,
    stage_index: int = 0,
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
    logging_initial_guess
        Optional array recording the starting iterate. Expected shape is
        ``(stage_slots, n)`` where ``n`` matches ``rhs``.
    logging_iteration_guesses
        Optional tensor recording per-iteration iterates. Expected shape is
        ``(stage_slots, max_iterations, n)``.
    logging_residuals
        Optional tensor recording per-iteration residual vectors. Expected
        shape is ``(stage_slots, max_iterations, n)``.
    logging_squared_norms
        Optional matrix recording per-iteration squared residual norms with
        shape ``(stage_slots, max_iterations)``.
    logging_preconditioned_vectors
        Optional tensor recording preconditioned residual directions. Expected
        shape is ``(stage_slots, max_iterations, n)``.
    instrumented
        When ``True`` the logging arrays are populated on each iteration.
    stage_index
        Stage slot identifying the row in the logging arrays that should be
        updated when ``instrumented`` is ``True``.

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

    state_dim = vector.shape[0]
    if instrumented and logging_initial_guess is not None:
        logging_initial_guess[stage_index, :] = solution

    tol_value = precision(tolerance)
    tol_squared = tol_value * tol_value
    iteration_limit = int(max_iterations)

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

    def _log_iteration(
        index: int,
        iterate: Array,
        residual_vec: Array,
        squared_norm: np.floating,
        preconditioned_vec: Array,
    ) -> None:
        if not instrumented or index < 0:
            return
        if logging_iteration_guesses is not None and index < logging_iteration_guesses.shape[1]:
            logging_iteration_guesses[stage_index, index, :] = iterate
        if logging_residuals is not None and index < logging_residuals.shape[1]:
            logging_residuals[stage_index, index, :] = residual_vec
        if logging_squared_norms is not None and index < logging_squared_norms.shape[1]:
            logging_squared_norms[stage_index, index] = squared_norm
        if (
            logging_preconditioned_vectors is not None
            and index < logging_preconditioned_vectors.shape[1]
        ):
            logging_preconditioned_vectors[stage_index, index, :] = (
                preconditioned_vec
            )

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
        _log_iteration(
            iteration - 1,
            solution,
            residual,
            residual_squared,
            direction,
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
