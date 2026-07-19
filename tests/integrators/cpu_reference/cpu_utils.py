"""Shared utilities for CPU reference integrator implementations."""

import math
import attrs
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cubie.integrators import CUBIE_RESULT_CODES


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
def _scaled_norm_impl(
    values: Array,
    reference: Array,
    atol: np.floating,
    rtol: np.floating,
) -> np.floating:
    """Return ``sum((|values[i]| / tol_i)^2) / n`` with
    ``tol_i = max(atol + rtol * |reference[i]|, 1e-16)``; <= 1.0 is
    converged.
    """

    size = values.shape[0]
    zero = values.dtype.type(0.0)
    nrm2 = zero
    floor = values.dtype.type(1e-16)
    inv_n = values.dtype.type(1.0 / size)
    for index in range(size):
        ref_value = reference[index]
        abs_ref = ref_value if ref_value >= zero else -ref_value
        tol = atol + rtol * abs_ref
        tol = tol if tol > floor else floor
        value = values[index]
        abs_val = value if value >= zero else -value
        ratio = abs_val / tol
        nrm2 = nrm2 + ratio * ratio
    return nrm2 * inv_n


def correction_norm_reference(
    update: Array,
    stage_state: Array,
    step_start: Array,
    atol: np.floating,
    rtol: np.floating,
) -> np.floating:
    """Return the scaled correction norm against the stage state.

    Mirrors the device correction norms: the update is scaled by
    ``atol + rtol * max(|stage_state|, |step_start|)``.
    """

    reference = np.maximum(np.abs(stage_state), np.abs(step_start))
    return _scaled_norm_impl(update, reference, atol, rtol)


def scaled_norm(
    values: Union[Sequence[float], Array],
    reference: Union[Sequence[float], Array],
    atol: np.floating,
    rtol: np.floating,
    precision: np.dtype,
) -> np.floating:
    """Return the mean squared scaled norm in ``precision``."""

    dtype, scalar_type = resolve_precision_signature(precision)
    values_array = np.asarray(values, dtype=dtype)
    reference_array = np.asarray(reference, dtype=dtype)
    return _scaled_norm_impl(
        values_array,
        reference_array,
        scalar_type(atol),
        scalar_type(rtol),
    )

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
    rtol: np.floating,
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
    """Return the Krylov solution for a dense operator matrix.

    Converged when the scaled residual norm is <= 1.
    """

    solution = np.empty_like(rhs)
    if has_initial_guess:
        solution[:] = initial_guess
    else:
        zero = operator_matrix.dtype.type(0.0)
        for index in range(solution.shape[0]):
            solution[index] = zero

    if instrumented and logging_initial_guess is not None:
        logging_initial_guess[stage_index, :] = solution

    typed_one = operator_matrix.dtype.type(1.0)
    iteration_limit = max_iterations

    operator_buffer = np.empty_like(rhs)
    residual = np.empty_like(rhs)
    _matrix_vector_product(operator_matrix, solution, operator_buffer)
    for index in range(residual.shape[0]):
        residual[index] = rhs[index] - operator_buffer[index]
    residual_norm2 = _scaled_norm_impl(residual, solution, tolerance, rtol)
    if residual_norm2 <= typed_one:
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
        residual_norm2 = _scaled_norm_impl(
            residual, solution, tolerance, rtol
        )

        _log_krylov_iteration(
            instrumented=instrumented,
            stage_index=stage_index,
            index=iteration - 1,
            iterate=solution,
            residual=residual,
            squared_norm=residual_norm2,
            direction=direction,
            logging_iteration_guesses=logging_iteration_guesses,
            logging_residuals=logging_residuals,
            logging_squared_norms=logging_squared_norms,
            logging_preconditioned_vectors=logging_preconditioned_vectors,
        )

        if residual_norm2 <= typed_one:
            converged = True
            break

    return solution, converged, iteration


@njit(cache=True)
def _bicgstab_solve_dense_impl(
    rhs: Array,
    operator_matrix: Array,
    tolerance: np.floating,
    rtol: np.floating,
    max_iterations: int,
    initial_guess: Array,
    has_initial_guess: bool,
    neumann_order: int,
) -> tuple[Array, bool, int]:
    """Return the BiCGSTAB solution for a dense operator matrix.

    Mirrors the device solver's preconditioned recurrences: the
    search direction and intermediate residual are preconditioned
    before each operator application, and a vanished recurrence
    scalar (pivot, omega, or rho) exits unconverged as breakdown.
    """

    dtype = operator_matrix.dtype
    zero = dtype.type(0.0)
    solution = np.empty_like(rhs)
    if has_initial_guess:
        solution[:] = initial_guess
    else:
        for index in range(solution.shape[0]):
            solution[index] = zero

    typed_one = operator_matrix.dtype.type(1.0)

    operator_buffer = np.empty_like(rhs)
    residual = np.empty_like(rhs)
    _matrix_vector_product(operator_matrix, solution, operator_buffer)
    for index in range(residual.shape[0]):
        residual[index] = rhs[index] - operator_buffer[index]
    residual_norm2 = _scaled_norm_impl(residual, solution, tolerance, rtol)
    if residual_norm2 <= typed_one:
        return solution, True, 0

    preconditioner_matrix = _compute_neumann_preconditioner(
        operator_matrix,
        neumann_order,
    )

    witness = residual.copy()
    direction = residual.copy()
    direction_hat = np.empty_like(rhs)
    s_hat = np.empty_like(rhs)
    v_vector = np.empty_like(rhs)
    t_vector = np.empty_like(rhs)

    rho_prev = _dot_product_impl(witness, residual)
    converged = False
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        _matrix_vector_product(
            preconditioner_matrix, direction, direction_hat
        )
        _matrix_vector_product(
            operator_matrix, direction_hat, v_vector
        )
        pivot = _dot_product_impl(witness, v_vector)
        if pivot == zero:
            return solution, False, iteration
        alpha = dtype.type(rho_prev / pivot)

        for index in range(solution.shape[0]):
            solution[index] = (
                solution[index] + alpha * direction_hat[index]
            )
            residual[index] = residual[index] - alpha * v_vector[index]
        residual_norm2 = _scaled_norm_impl(
            residual, solution, tolerance, rtol
        )
        if residual_norm2 <= typed_one:
            converged = True
            break

        _matrix_vector_product(preconditioner_matrix, residual, s_hat)
        _matrix_vector_product(operator_matrix, s_hat, t_vector)
        t_squared = _dot_product_impl(t_vector, t_vector)
        if t_squared == zero:
            return solution, False, iteration
        omega = dtype.type(
            _dot_product_impl(t_vector, residual) / t_squared
        )

        for index in range(solution.shape[0]):
            solution[index] = solution[index] + omega * s_hat[index]
            residual[index] = residual[index] - omega * t_vector[index]
        residual_norm2 = _scaled_norm_impl(
            residual, solution, tolerance, rtol
        )
        if residual_norm2 <= typed_one:
            converged = True
            break

        rho_new = _dot_product_impl(witness, residual)
        if rho_new == zero or omega == zero:
            return solution, False, iteration
        beta = dtype.type((rho_new / rho_prev) * (alpha / omega))
        for index in range(solution.shape[0]):
            direction[index] = residual[index] + beta * (
                direction[index] - omega * v_vector[index]
            )
        rho_prev = rho_new

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
    newton_rtol: np.floating = 0.0,
    correction_norm: Optional[Callable[[Array, Array], np.floating]] = None,
    prev_theta_store: Optional[Array] = None,
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
    """Solve the nonlinear system on the CPU.

    Mirrors the device Newton solver: the update-error bound with
    warm-started contraction estimates decides convergence, and the
    optional line search falls back to residual descent.
    ``correction_norm`` computes the scaled update norm against the
    stage context; when absent the update is scaled against the
    iterate. ``prev_theta_store`` is a one-element array persisting
    the contraction estimate between solves.
    """

    dtype, scalar_type = resolve_precision_signature(precision)
    atol_value = scalar_type(newton_tol)
    rtol_value = scalar_type(newton_rtol)
    damping_value = scalar_type(newton_damping)
    typed_one = scalar_type(1.0)
    iteration_limit = int(newton_max_iters)
    backtrack_limit = int(newton_max_backtracks)

    state = np.asarray(initial_guess, dtype=dtype).copy()
    residual = np.zeros_like(state)
    direction = np.zeros_like(state)

    typed_zero = scalar_type(0.0)
    typed_tiny = scalar_type(np.finfo(dtype).tiny)
    typed_huge = scalar_type(np.finfo(dtype).max)
    kappa = scalar_type(0.01)
    first_iteration_bound = scalar_type(1.0e-5)
    theta_decay = scalar_type(0.3)
    theta_divergence_bound = scalar_type(2.0)
    stagnation_eps = scalar_type(100.0 * np.sqrt(np.finfo(dtype).eps))

    if correction_norm is None:
        def correction_norm(update, iterate):
            return _scaled_norm_impl(
                update, iterate, atol_value, rtol_value
            )

    # Warm-started contraction estimate; zero marks a fresh store or
    # a failed previous solve.
    stored_theta = typed_zero
    if prev_theta_store is not None:
        stored_theta = scalar_type(prev_theta_store[0])
    prev_theta = stored_theta if stored_theta > typed_zero else typed_one

    # RMS norm of the previous accepted full-step correction.
    ndz_prev = typed_zero

    if instrumented and newton_initial_guesses is not None:
        newton_initial_guesses[stage_index, :] = state

    log_index = 0
    if instrumented:
        residual = np.asarray(residual_fn(state), dtype=dtype)
        entry_norm2 = _scaled_norm_impl(
            residual, state, atol_value, rtol_value
        )
        _log_newton_iteration(
            instrumented=True,
            stage_index=stage_index,
            index=log_index,
            candidate=state,
            residual=residual,
            squared_norm=entry_norm2,
            logging_iteration_guesses=newton_iteration_guesses,
            logging_residuals=newton_residuals,
            logging_squared_norms=newton_squared_norms,
        )
        log_index += 1

    converged = False
    failed = False
    iterations_used = 0
    for iteration in range(iteration_limit):
        if converged or failed:
            break
        iterations_used = iteration + 1

        residual = np.asarray(residual_fn(state), dtype=dtype)
        norm2_residual = _scaled_norm_impl(
            residual, state, atol_value, rtol_value
        )
        jacobian = np.asarray(jacobian_fn(state), dtype=dtype)

        direction.fill(typed_zero)
        linear_kwargs: dict[str, Any] = {"initial_guess": direction}
        if instrumented:
            slot = stage_index * max(iteration_limit, 1) + iteration
            linear_kwargs.update(
                {
                    "stage_index": slot,
                    "instrumented": True,
                    "logging_initial_guess": linear_initial_guesses,
                    "logging_iteration_guesses": linear_iteration_guesses,
                    "logging_residuals": linear_residuals,
                    "logging_squared_norms": linear_squared_norms,
                    "logging_preconditioned_vectors": (
                        linear_preconditioned_vectors
                    ),
                }
            )

        direction, linear_converged, _ = linear_solver(
            jacobian,
            -residual,
            **linear_kwargs,
        )

        step = np.asarray(direction, dtype=dtype)

        norm2_dz = scalar_type(correction_norm(step, state))
        ndz = scalar_type(np.sqrt(norm2_dz))

        # A failed linear solve yields no usable correction: nothing
        # commits and no contraction evidence accrues.
        judged = bool(linear_converged)
        history = ndz_prev > typed_zero
        if history:
            theta = scalar_type(
                max(
                    theta_decay * prev_theta,
                    ndz / max(ndz_prev, typed_tiny),
                )
            )
        else:
            theta = prev_theta
        small_first_step = iteration == 0 and ndz < first_iteration_bound
        eta_accept = bool(
            theta < typed_one
            and theta * ndz < kappa * (typed_one - theta)
        )

        scale = typed_one
        if backtrack_limit == 0:
            nonfinite = not (norm2_dz <= typed_huge)
            stagnant = (
                judged
                and history
                and abs(theta - typed_one) <= stagnation_eps
            )
            diverging = judged and (
                (history and theta > theta_divergence_bound)
                or nonfinite
            )
            converged_stagnant = (
                stagnant and ndz <= typed_one and not diverging
            )
            failed_now = diverging or (stagnant and ndz > typed_one)
            failed = failed or failed_now

            commit = judged and not failed_now and not converged_stagnant
            if commit:
                state = np.asarray(state + step, dtype=dtype)
            converged = converged or converged_stagnant or (
                commit and (eta_accept or small_first_step)
            )
            ndz_prev = ndz if commit else typed_zero
            if judged and history:
                prev_theta = theta
            if instrumented and commit:
                residual = np.asarray(residual_fn(state), dtype=dtype)
                post_norm2 = _scaled_norm_impl(
                    residual, state, atol_value, rtol_value
                )
                _log_newton_iteration(
                    instrumented=True,
                    stage_index=stage_index,
                    index=log_index,
                    candidate=state,
                    residual=residual,
                    squared_norm=post_norm2,
                    logging_iteration_guesses=newton_iteration_guesses,
                    logging_residuals=newton_residuals,
                    logging_squared_norms=newton_squared_norms,
                )
                log_index += 1
        else:
            accept_update = judged and (eta_accept or small_first_step)
            if judged and history:
                prev_theta = theta
            ndz_next = typed_zero
            if accept_update:
                state = np.asarray(state + step, dtype=dtype)
                converged = True
                ndz_next = ndz
                if instrumented:
                    residual = np.asarray(residual_fn(state), dtype=dtype)
                    post_norm2 = _scaled_norm_impl(
                        residual, state, atol_value, rtol_value
                    )
                    _log_newton_iteration(
                        instrumented=True,
                        stage_index=stage_index,
                        index=log_index,
                        candidate=state,
                        residual=residual,
                        squared_norm=post_norm2,
                        logging_iteration_guesses=newton_iteration_guesses,
                        logging_residuals=newton_residuals,
                        logging_squared_norms=newton_squared_norms,
                    )
                    log_index += 1
            else:
                for _ in range(backtrack_limit + 1):
                    trial_state = np.asarray(
                        state + scale * step, dtype=dtype
                    )
                    trial_residual = np.asarray(
                        residual_fn(trial_state), dtype=dtype
                    )
                    trial_norm2 = _scaled_norm_impl(
                        trial_residual, trial_state, atol_value, rtol_value
                    )

                    _log_newton_iteration(
                        instrumented=instrumented,
                        stage_index=stage_index,
                        index=log_index,
                        candidate=trial_state,
                        residual=trial_residual,
                        squared_norm=trial_norm2,
                        logging_iteration_guesses=newton_iteration_guesses,
                        logging_residuals=newton_residuals,
                        logging_squared_norms=newton_squared_norms,
                    )
                    log_index += 1

                    if trial_norm2 < norm2_residual:
                        state = trial_state
                        residual = trial_residual
                        converged = trial_norm2 <= typed_one
                        if linear_converged and scale == typed_one:
                            ndz_next = ndz
                        break
                    scale = scalar_type(scale * damping_value)

            ndz_prev = ndz_next

        if (
            instrumented
            and newton_iteration_scale is not None
            and iteration < newton_iteration_scale.shape[1]
        ):
            newton_iteration_scale[stage_index, iteration] = scale

    # Persist contraction history for the next solve; a failed solve
    # resets it to the conservative estimate.
    if prev_theta_store is not None:
        prev_theta_store[0] = prev_theta if converged else typed_one

    return state, converged, iterations_used


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
        CUBIE_RESULT_CODES.SUCCESS
        if converged
        else CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
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
        """Return Horner evaluations and range flag for ``coefficients``.

        Busybody AI over-checking left in place."""

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
        scaled_floor = precision(math.floor(scaled))
        idx = np.int32(scaled_floor)

        if self.wrap:
            segment = idx % segment_count
            tau = scaled - scaled_floor
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
    rtol: np.floating = 0.0,
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
        Absolute tolerance of the scaled convergence norm.
    max_iterations
        Maximum iteration count for the descent loop. Zero is permitted.
    precision
        Floating-point precision to use for the iteration.
    rtol
        Relative tolerance of the scaled convergence norm, scaled by
        the solution iterate.
    neumann_order
        Order of the truncated Neumann-series left preconditioner.
    correction_type
        Linear solve to apply. ``"steepest_descent"``,
        ``"minimal_residual"``, or ``"bicgstab"``. The logging
        arrays are only populated for the descent variants.
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

    if correction_type not in (
        "steepest_descent",
        "minimal_residual",
        "bicgstab",
    ):
        raise ValueError(
            "Correction type must be 'steepest_descent', "
            "'minimal_residual', or 'bicgstab'."
        )

    dtype, scalar_type = resolve_precision_signature(precision)
    matrix = np.asarray(operator_matrix, dtype=dtype)
    vector = np.asarray(rhs, dtype=dtype)

    minimal_residual = correction_type == "minimal_residual"
    tol_value = scalar_type(tolerance)
    rtol_value = scalar_type(rtol)
    iteration_limit = int(max_iterations)
    order = int(neumann_order)

    if initial_guess is None:
        guess = np.zeros_like(vector)
    else:
        guess = np.asarray(initial_guess, dtype=dtype)

    initial_provided = initial_guess is not None
    if correction_type == "bicgstab":
        solution, converged, iteration = _bicgstab_solve_dense_impl(
            vector,
            matrix,
            tol_value,
            rtol_value,
            iteration_limit,
            guess,
            initial_provided,
            order,
        )
    else:
        solution, converged, iteration = _krylov_solve_dense_impl(
            vector,
            matrix,
            tol_value,
            rtol_value,
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
