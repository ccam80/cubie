"""Shared utilities for CPU reference integrator implementations."""

import math
import attrs
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cubie.integrators import IntegratorReturnCodes


Array = NDArray[np.floating]
STATUS_MASK = 0xFFFF


def _ensure_array(vector: Union[Sequence[float], Array], dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.atleast_1d(vector).astype(dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array

def resolve_precision_signature(
    precision: np.dtype,
) -> tuple[np.dtype, type[np.floating]]:
    """Return a canonical ``(dtype, scalar_type)`` tuple for ``precision``."""

    dtype = np.dtype(precision)
    scalar_type = dtype.type  # type: ignore[assignment]
    return dtype, scalar_type


@njit(cache=True)
def _dot_product_impl(left: Array, right: Array) -> np.floating:
    """Return the dot product of ``left`` and ``right``."""

    size = left.shape[0]
    total = left.dtype.type(0.0)
    for index in range(size):
        total = total + left[index] * right[index]
    return total


def dot_product(left: Array, right: Array, precision: np.dtype) -> np.floating:
    """Return the dot product of ``left`` and ``right`` in ``precision``."""

    left_array = np.asarray(left, dtype=precision)
    right_array = np.asarray(right, dtype=precision)
    return _dot_product_impl(left_array, right_array)


@njit(cache=True)
def _squared_norm_impl(vector: Array) -> np.floating:
    """Return the squared Euclidean norm of ``vector``."""

    size = vector.shape[0]
    total = vector.dtype.type(0.0)
    for index in range(size):
        value = vector[index]
        total = total + value * value
    return total


def squared_norm(vector: Array,precision: np.dtype) -> np.floating:
    """Return the squared Euclidean norm of ``vector`` in ``precision``."""

    array = np.asarray(vector, dtype=precision)
    return _squared_norm_impl(array)


@njit(cache=True)
def _euclidean_norm_impl(vector: Array) -> np.floating:
    """Return the Euclidean norm of ``vector``."""

    return np.sqrt(_squared_norm_impl(vector))


def euclidean_norm(
    vector: Union[Sequence[float], Array], precision: np.dtype
) -> np.floating:
    """Return the Euclidean norm of ``vector`` in ``precision``."""

    array = np.asarray(vector, dtype=precision)
    return _euclidean_norm_impl(array)

@njit(cache=True)
def _matrix_vector_product(matrix: Array, vector: Array, out: Array) -> None:
    """Store ``matrix @ vector`` into ``out`` without allocating."""

    rows, cols = matrix.shape
    for row in range(rows):
        total = matrix.dtype.type(0.0)
        for col in range(cols):
            total = total + matrix[row, col] * vector[col]
        out[row] = total

@njit(cache=True)
def _compute_neumann_preconditioner(matrix: Array, order: int) -> Array:
    """Return the truncated Neumann series for ``matrix``."""

    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    neumann = identity.copy()
    if order <= 0:
        return neumann

    residual = identity - matrix
    power = identity.copy()
    for _ in range(order):
        power = power @ residual
        neumann = neumann + power
    return neumann


@njit(cache=True)
def _log_krylov_iteration(
    instrumented: bool,
    stage_index: int,
    index: int,
    iterate: Array,
    residual: Array,
    squared_norm: np.floating,
    direction: Array,
    logging_iteration_guesses: Optional[Array],
    logging_residuals: Optional[Array],
    logging_squared_norms: Optional[Array],
    logging_preconditioned_vectors: Optional[Array],
) -> None:
    """Record iteration diagnostics when instrumentation buffers are present."""

    if not instrumented or index < 0:
        return
    if (
        logging_iteration_guesses is not None
        and index < logging_iteration_guesses.shape[1]
    ):
        logging_iteration_guesses[stage_index, index, :] = iterate
    if logging_residuals is not None and index < logging_residuals.shape[1]:
        logging_residuals[stage_index, index, :] = residual
    if (
        logging_squared_norms is not None
        and index < logging_squared_norms.shape[1]
    ):
        logging_squared_norms[stage_index, index] = squared_norm
    if (
        logging_preconditioned_vectors is not None
        and index < logging_preconditioned_vectors.shape[1]
    ):
        logging_preconditioned_vectors[stage_index, index, :] = direction


@njit(cache=True)
def _log_newton_iteration(
    instrumented: bool,
    stage_index: int,
    index: int,
    candidate: Array,
    residual: Array,
    squared_norm: np.floating,
    logging_iteration_guesses: Optional[Array],
    logging_residuals: Optional[Array],
    logging_squared_norms: Optional[Array],
) -> None:
    """Record Newton diagnostics when logging buffers are available."""

    if not instrumented or index < 0:
        return
    if (
        logging_iteration_guesses is not None
        and index < logging_iteration_guesses.shape[1]
    ):
        logging_iteration_guesses[stage_index, index, :] = candidate
    if logging_residuals is not None and index < logging_residuals.shape[1]:
        logging_residuals[stage_index, index, :] = residual
    if (
        logging_squared_norms is not None
        and index < logging_squared_norms.shape[1]
    ):
        logging_squared_norms[stage_index, index] = squared_norm


@attrs.define
class StepResult:
    """Container describing the outcome of a single integration step."""

    state: Array
    observables: Array
    error: Array
    status: int = 0
    niters: int = 0


@attrs.define(slots=True)
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
    stage_states: Union[Array, None] = None
    stage_derivatives: Union[Array, None] = None
    stage_observables: Union[Array, None] = None
    newton_initial_guesses: Union[Array, None] = None
    newton_iteration_guesses: Union[Array, None] = None
    newton_residuals: Union[Array, None] = None
    newton_squared_norms: Union[Array, None] = None
    newton_iteration_scale: Union[Array, None] = None
    linear_initial_guesses: Union[Array, None] = None
    linear_iteration_guesses: Union[Array, None] = None
    linear_residuals: Union[Array, None] = None
    linear_squared_norms: Union[Array, None] = None
    linear_preconditioned_vectors: Union[Array, None] = None
    extra_vectors: Optional[dict[str, Array]] = None


StepResultLike = Union[StepResult, InstrumentedStepResult]


@njit(cache=True)
def _krylov_solve_dense_impl(
    rhs: Array,
    operator_matrix: Array,
    tolerance: np.floating,
    max_iterations: int,
    initial_guess: Array,
    has_initial_guess: bool,
    neumann_order: int,
    minimal_residual: bool,
    instrumented: bool,
    logging_initial_guess: Optional[Array],
    logging_iteration_guesses: Optional[Array],
    logging_residuals: Optional[Array],
    logging_squared_norms: Optional[Array],
    logging_preconditioned_vectors: Optional[Array],
    stage_index: int,
) -> tuple[Array, bool, int]:
    """Return the Krylov solution for a dense operator matrix."""

    solution = np.empty_like(rhs)
    if has_initial_guess:
        solution[:] = initial_guess
    else:
        zero = operator_matrix.dtype.type(0.0)
        for index in range(solution.shape[0]):
            solution[index] = zero

    if instrumented and logging_initial_guess is not None:
        logging_initial_guess[stage_index, :] = solution

    tol_squared = tolerance * tolerance
    iteration_limit = max_iterations

    operator_buffer = np.empty_like(rhs)
    residual = np.empty_like(rhs)
    _matrix_vector_product(operator_matrix, solution, operator_buffer)
    for index in range(residual.shape[0]):
        residual[index] = rhs[index] - operator_buffer[index]
    residual_squared = _dot_product_impl(residual, residual)
    if residual_squared <= tol_squared:
        return solution, True, 0

    preconditioner_matrix = _compute_neumann_preconditioner(
        operator_matrix,
        neumann_order,
    )
    direction = np.empty_like(rhs)

    converged = False
    iteration = 0

    while iteration < iteration_limit:
        iteration += 1

        _matrix_vector_product(preconditioner_matrix, residual, direction)
        _matrix_vector_product(operator_matrix, direction, operator_buffer)

        if minimal_residual:
            numerator = _dot_product_impl(operator_buffer, residual)
            denominator = _dot_product_impl(operator_buffer, operator_buffer)
        else:
            numerator = _dot_product_impl(residual, direction)
            denominator = _dot_product_impl(operator_buffer, direction)

        if denominator == operator_matrix.dtype.type(0.0):
            return solution, False, iteration

        alpha = operator_matrix.dtype.type(numerator / denominator)
        if converged:
            alpha = operator_matrix.dtype.type(0.0)

        for index in range(solution.shape[0]):
            solution[index] = solution[index] + alpha * direction[index]
            residual[index] = residual[index] - alpha * operator_buffer[index]
        residual_squared = _dot_product_impl(residual, residual)

        _log_krylov_iteration(
            instrumented=instrumented,
            stage_index=stage_index,
            index=iteration - 1,
            iterate=solution,
            residual=residual,
            squared_norm=residual_squared,
            direction=direction,
            logging_iteration_guesses=logging_iteration_guesses,
            logging_residuals=logging_residuals,
            logging_squared_norms=logging_squared_norms,
            logging_preconditioned_vectors=logging_preconditioned_vectors,
        )

        if residual_squared <= tol_squared:
            converged = True
            break

    return solution, converged, iteration


def newton_solve(
    initial_guess: Array,
    precision: np.dtype,
    residual_fn: Callable[[Array], Array],
    jacobian_fn: Callable[[Array], Array],
    linear_solver: Callable[..., tuple[Array, bool, int]],
    newton_tol: np.floating,
    newton_max_iters: int,
    newton_damping: np.floating,
    newton_max_backtracks: int,
    stage_index: int = 0,
    instrumented: bool = False,
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
) -> tuple[Array, bool, int]:
    """Return the Newton update for ``residual_fn`` starting at ``initial_guess``.

    Parameters
    ----------
    initial_guess
        Starting candidate for the Newton iteration.
    precision
        Floating-point dtype used for all computations and logging buffers.
    residual_fn
        Callable returning the residual vector for a candidate state.
    jacobian_fn
        Callable returning the Jacobian matrix for a candidate state.
    linear_solver
        Callable solving the linear system produced on each iteration.
    newton_tol
        Convergence tolerance on the residual norm.
    newton_max_iters
        Maximum number of Newton updates to attempt.
    newton_damping
        Multiplicative factor applied to the step size during backtracking.
    newton_max_backtracks
        Maximum number of damping attempts per iteration.
    stage_index
        Index of the stage recorded in the logging buffers.
    instrumented
        When ``True`` the logging arrays are populated with diagnostics.
    newton_initial_guesses
        Optional tensor recording the starting iterate per stage.
    newton_iteration_guesses
        Optional tensor recording iterates per Newton update.
    newton_residuals
        Optional tensor recording residual vectors per Newton update.
    newton_squared_norms
        Optional matrix storing squared residual norms per update.
    newton_iteration_scale
        Optional matrix storing accepted damping factors per iteration.
    linear_initial_guesses
        Optional tensor recording initial guesses for the linear solver.
    linear_iteration_guesses
        Optional tensor recording per-iteration linear iterates.
    linear_residuals
        Optional tensor recording per-iteration linear residual vectors.
    linear_squared_norms
        Optional matrix storing per-iteration linear residual norms.
    linear_preconditioned_vectors
        Optional tensor storing preconditioned vectors per linear iteration.

    Returns
    -------
    tuple[Array, bool, int]
        Final iterate, convergence flag, and iteration count.
    """

    dtype, scalar_type = resolve_precision_signature(precision)
    tol_value = scalar_type(newton_tol)
    damping_value = scalar_type(newton_damping)
    iteration_limit = int(newton_max_iters)
    backtrack_limit = int(newton_max_backtracks)

    state = np.asarray(initial_guess, dtype=dtype).copy()
    residual = np.asarray(residual_fn(state), dtype=dtype)
    norm = euclidean_norm(residual, precision=dtype)
    norm_squared = norm * norm
    direction = np.zeros_like(residual)
    if instrumented and newton_initial_guesses is not None:
        newton_initial_guesses[stage_index, :] = state

    log_index = 0
    _log_newton_iteration(
        instrumented=instrumented,
        stage_index=stage_index,
        index=log_index,
        candidate=state,
        residual=residual,
        squared_norm=norm_squared,
        logging_iteration_guesses=newton_iteration_guesses,
        logging_residuals=newton_residuals,
        logging_squared_norms=newton_squared_norms,
    )
    log_index += 1

    if norm <= tol_value:
        return state, True, 0

    for iteration in range(iteration_limit):
        jacobian = np.asarray(jacobian_fn(state), dtype=dtype)

        linear_kwargs: dict[str, Any] = {}
        if instrumented:
            slot = stage_index * max(iteration_limit, 1) + iteration
            linear_kwargs = {
                "stage_index": slot,
                "initial_guess": direction,
                "instrumented": True,
                "logging_initial_guess": linear_initial_guesses,
                "logging_iteration_guesses": linear_iteration_guesses,
                "logging_residuals": linear_residuals,
                "logging_squared_norms": linear_squared_norms,
                "logging_preconditioned_vectors": (
                    linear_preconditioned_vectors
                ),
            }

        direction, converged, _ = linear_solver(
            jacobian,
            -residual,
            **linear_kwargs,
        )
        if not converged:
            return state, False, iteration + 1

        step = np.asarray(direction, dtype=dtype)
        scale = scalar_type(1.0)
        accepted = False

        for _ in range(backtrack_limit + 1):
            trial_state = state + scale * step
            trial_residual = np.asarray(residual_fn(trial_state), dtype=dtype)
            trial_norm = euclidean_norm(trial_residual, precision=dtype)
            trial_squared = trial_norm * trial_norm

            _log_newton_iteration(
                instrumented=instrumented,
                stage_index=stage_index,
                index=log_index,
                candidate=trial_state,
                residual=trial_residual,
                squared_norm=trial_squared,
                logging_iteration_guesses=newton_iteration_guesses,
                logging_residuals=newton_residuals,
                logging_squared_norms=newton_squared_norms,
            )
            log_index += 1

            if trial_norm <= tol_value:
                return trial_state, True, iteration + 1
            if trial_norm < norm:
                state = trial_state
                residual = trial_residual
                norm = trial_norm
                norm_squared = trial_squared
                accepted = True
                break
            scale = scalar_type(scale * damping_value)

        if not accepted:
            return state, False, iteration + 1

        if (
            instrumented
            and newton_iteration_scale is not None
            and iteration < newton_iteration_scale.shape[1]
        ):
            newton_iteration_scale[stage_index, iteration] = scale

    return state, False, iteration_limit


def make_step_result(
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

    iter_count = max(0, min(int(niters) + 1, STATUS_MASK))
    if not instrument:
        return StepResult(state, observables, error, status, iter_count)

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
        niters=iter_count,
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
    iter_count = max(0, min(int(niters) + 1, STATUS_MASK))
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
        self._order = coeffs.shape[2] - 1 if coeffs.size else -1
        self._zero = np.zeros(self._width, dtype=precision)
        self._zero_value = precision(0.0)
        self._pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        self._inv_dt = precision(precision(1.0) / self.dt)
        offset = self.dt if self._pad_clamped else self._zero_value
        self._evaluation_start = precision(self.t0 - offset)

    def _evaluate(
        self,
        time: float,
        coefficients: Array,
        *,
        segments: Optional[int] = None,
        order: Optional[int] = None,
    ) -> tuple[Array, bool]:
        """Return Horner evaluations and range flag for ``coefficients``."""

        segment_count = self._segments if segments is None else int(segments)
        if segment_count <= 0 or self._width == 0:
            return self._zero.copy(), False

        poly_order = coefficients.shape[-1] - 1 if order is None else int(order)
        if poly_order < 0:
            return self._zero.copy(), False

        precision = self.precision
        time_value = precision(time)
        scaled = precision(
            (time_value - self._evaluation_start) * self._inv_dt
        )
        scaled_floor = math.floor(float(scaled))
        idx = int(scaled_floor)

        if self.wrap:
            segment = idx % segment_count
            if segment < 0:
                segment += segment_count
            tau = precision(scaled - precision(scaled_floor))
            in_range = True
        else:
            max_segment = segment_count - 1
            if idx < 0:
                segment = 0
            elif idx >= segment_count:
                segment = max_segment
            else:
                segment = idx
            tau = precision(scaled - precision(segment))
            in_range = (
                scaled >= self._zero_value
                and scaled <= precision(segment_count)
            )

        limit = poly_order + 1
        values = self._zero.copy()
        for driver_idx in range(self._width):
            segment_coeffs = coefficients[segment, driver_idx, :limit]
            acc = self._zero_value
            for coeff in reversed(segment_coeffs):
                acc = acc * tau + precision(coeff)
            values[driver_idx] = acc
        return values, in_range

    def evaluate(self, time: float) -> Array:
        """Return driver values interpolated at ``time``."""

        values, in_range = self._evaluate(
            time,
            self.coefficients,
            segments=self._segments,
            order=self._order,
        )
        if self.wrap or in_range:
            return values
        return self._zero.copy()

    def __call__(self, time: float) -> Array:
        """Alias for :meth:`evaluate` so instances are callable."""

        return self.evaluate(time)

    def derivative(self, time: float) -> Array:
        """Return time derivatives of the drivers evaluated at ``time``."""

        if self._segments == 0 or self._width == 0 or self._order <= 0:
            return self._zero.copy()

        derivative_coeffs = self.coefficients[..., 1:].copy()
        powers = np.arange(
            1, derivative_coeffs.shape[-1] + 1, dtype=self.precision
        ).reshape(1, 1, -1)
        derivative_coeffs *= powers

        values, in_range = self._evaluate(
            time,
            derivative_coeffs,
            segments=self._segments,
            order=self._order - 1,
        )
        if not self.wrap and not in_range:
            return self._zero.copy()
        return values * self._inv_dt

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
    operator_matrix: Array,
    rhs: Array,
    tolerance: np.floating,
    max_iterations: int,
    precision: np.dtype,
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
    """Solve ``operator_matrix @ x = rhs`` using dense Krylov iterations.

    Parameters
    ----------
    operator_matrix
        Dense linear operator expressed as a matrix.
    rhs
        Right-hand side vector.
    tolerance
        Convergence tolerance on the residual norm.
    max_iterations
        Maximum iteration count for the descent loop. Zero is permitted.
    precision
        Floating-point precision to use for the iteration.
    neumann_order
        Order of the truncated Neumann-series left preconditioner.
    correction_type
        Descent update to apply. ``"steepest_descent"`` or
        ``"minimal_residual"``.
    initial_guess
        Optional starting iterate for the solve. Defaults to the zero vector.
    instrumented
        When ``True`` the logging arrays are populated on each iteration.
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

    dtype, scalar_type = resolve_precision_signature(precision)
    matrix = np.asarray(operator_matrix, dtype=dtype)
    vector = np.asarray(rhs, dtype=dtype)

    minimal_residual = correction_type == "minimal_residual"
    tol_value = scalar_type(tolerance)
    iteration_limit = int(max_iterations)
    order = int(neumann_order)

    if initial_guess is None:
        guess = np.zeros_like(vector)
    else:
        guess = np.asarray(initial_guess, dtype=dtype)

    initial_provided = initial_guess is not None
    solution, converged, iteration = _krylov_solve_dense_impl(
        vector,
        matrix,
        tol_value,
        iteration_limit,
        guess,
        initial_provided,
        order,
        minimal_residual,
        instrumented,
        logging_initial_guess,
        logging_iteration_guesses,
        logging_residuals,
        logging_squared_norms,
        logging_preconditioned_vectors,
        int(stage_index),
    )
    return (
        np.asarray(solution, dtype=dtype),
        bool(converged),
        int(iteration),
    )


__all__ = [
    "Array",
    "DriverEvaluator",
    "STATUS_MASK",
    "resolve_precision_signature",
    "newton_solve",
    "dot_product",
    "euclidean_norm",
    "StepResult",
    "_encode_solver_status",
    "_ensure_array",
    "squared_norm",
    "krylov_solve",
]
