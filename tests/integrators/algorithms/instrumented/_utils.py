"""Utilities shared across instrumentation-enabled integrator tests."""

import numpy as np
import attrs

@attrs.define(slots=True)
class InstrumentationHostBuffers:
    """Host-side buffers for instrumented integrator diagnostics.

    Attributes
    ----------
    When ``num_steps`` exceeds one, arrays include a leading dimension
    that stores successive step diagnostics.
    stage_count:
        Number of stages in the tableau being instrumented.
    residuals:
        Per-stage nonlinear residual vectors.
    jacobian_updates:
        Per-stage accumulated Jacobian corrections.
    stage_states:
        Proposed states for each stage.
    stage_derivatives:
        Derivative evaluations for each stage.
    stage_observables:
        Observable evaluations for each stage.
    stage_drivers:
        Driver samples recorded per stage.
    stage_increments:
        Stage-wise state increments.
    newton_initial_guesses:
        Initial Newton guesses prior to backtracking.
    newton_iteration_guesses:
        Newton iteration guesses including backtracking slots.
    newton_residuals:
        Residual vectors captured during Newton backtracking.
    newton_squared_norms:
        Residual norms accumulated during Newton backtracking.
    newton_iteration_scale:
        Applied damping factors for each Newton iteration.
    linear_initial_guesses:
        Initial guesses for linear Krylov solves.
    linear_iteration_guesses:
        Krylov iteration guesses per linear solve.
    linear_residuals:
        Residual vectors recorded during Krylov iterations.
    linear_squared_norms:
        Residual norms recorded during Krylov iterations.
    linear_preconditioned_vectors:
        Preconditioned vectors from Krylov iterations.
    """

    stage_count: int
    residuals: np.ndarray
    jacobian_updates: np.ndarray
    stage_states: np.ndarray
    stage_derivatives: np.ndarray
    stage_observables: np.ndarray
    stage_drivers: np.ndarray
    stage_increments: np.ndarray
    newton_initial_guesses: np.ndarray
    newton_iteration_guesses: np.ndarray
    newton_residuals: np.ndarray
    newton_squared_norms: np.ndarray
    newton_iteration_scale: np.ndarray
    linear_initial_guesses: np.ndarray
    linear_iteration_guesses: np.ndarray
    linear_residuals: np.ndarray
    linear_squared_norms: np.ndarray
    linear_preconditioned_vectors: np.ndarray


def create_instrumentation_host_buffers(
    *,
    precision: np.dtype,
    stage_count: int,
    num_steps: int = 1,
    state_size: int,
    observable_size: int,
    driver_size: int,
    newton_max_iters: int,
    newton_max_backtracks: int,
    linear_max_iters: int,
    flattened_solver: bool = False,
) -> InstrumentationHostBuffers:
    """Return zeroed buffers sized for instrumentation diagnostics.

    Parameters
    ----------
    precision:
        Floating point dtype used for integrator state arrays.
    stage_count:
        Number of stages requested by the tableau.
    num_steps:
        Number of successive steps captured by the instrumentation buffers.
    state_size:
        Dimension of the state vector.
    observable_size:
        Dimension of the observable vector.
    driver_size:
        Dimension of the driver vector.
    newton_max_iters:
        Maximum allowed Newton iterations per stage.
    newton_max_backtracks:
        Maximum number of backtracking attempts per Newton iteration.
    linear_max_iters:
        Maximum iterations permitted for Krylov solves.
    flattened_solver:
        When True, solver arrays use flattened stage_count * state_size dimension
        instead of separate stage arrays. Default False.

    Returns
    -------
    InstrumentationHostBuffers
        Buffer container with arrays ready for instrumentation writes.
    """

    resolved_stage_count = int(stage_count)
    step_slots = max(1, int(num_steps))
    state_dim = int(state_size)
    observable_dim = int(observable_size)
    driver_dim = int(driver_size)
    newton_iters = int(newton_max_iters)
    backtracks = int(newton_max_backtracks)
    newton_slots = newton_iters * (backtracks + 1) + 1
    linear_iters = int(linear_max_iters)
    linear_slots = resolved_stage_count * newton_iters
    dtype = np.dtype(precision)
    
    # When flattened_solver is True, solve for all stages simultaneously
    solver_dim = resolved_stage_count * state_dim if flattened_solver else state_dim
    solver_stage_count = 1 if flattened_solver else resolved_stage_count

    def _step_shape(*base: int) -> tuple[int, ...]:
        base_shape = tuple(int(value) for value in base)
        if step_slots == 1:
            return base_shape
        return (step_slots,) + base_shape

    residuals = np.zeros(_step_shape(resolved_stage_count, state_dim), dtype=dtype)
    jacobian_updates = np.zeros_like(residuals)
    stage_states = np.zeros_like(residuals)
    stage_derivatives = np.zeros_like(residuals)
    stage_observables = np.zeros(
        _step_shape(resolved_stage_count, observable_dim),
        dtype=dtype,
    )
    stage_drivers = np.zeros(
        _step_shape(resolved_stage_count, driver_dim),
        dtype=dtype,
    )
    stage_increments = np.zeros_like(residuals)
    newton_initial_guesses = np.zeros(
        _step_shape(solver_stage_count, solver_dim),
        dtype=dtype,
    )
    newton_iteration_guesses = np.zeros(
        _step_shape(solver_stage_count, newton_slots, solver_dim),
        dtype=dtype,
    )
    newton_residuals = np.zeros_like(newton_iteration_guesses)
    newton_squared_norms = np.zeros(
        _step_shape(solver_stage_count, newton_slots),
        dtype=dtype,
    )
    newton_iteration_scale = np.zeros(
        _step_shape(solver_stage_count, newton_iters),
        dtype=dtype,
    )
    linear_initial_guesses = np.zeros(
        _step_shape(linear_slots, solver_dim),
        dtype=dtype,
    )
    linear_iteration_guesses = np.zeros(
        _step_shape(linear_slots, linear_iters, solver_dim),
        dtype=dtype,
    )
    linear_residuals = np.zeros_like(linear_iteration_guesses)
    linear_squared_norms = np.zeros(
        _step_shape(linear_slots, linear_iters),
        dtype=dtype,
    )
    linear_preconditioned_vectors = np.zeros_like(linear_iteration_guesses)

    return InstrumentationHostBuffers(
        stage_count=resolved_stage_count,
        residuals=residuals,
        jacobian_updates=jacobian_updates,
        stage_states=stage_states,
        stage_derivatives=stage_derivatives,
        stage_observables=stage_observables,
        stage_drivers=stage_drivers,
        stage_increments=stage_increments,
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
    )
