"""CPU reference implementations for integrator step algorithms."""

from functools import partial
import math
from typing import Callable, Optional, Sequence, Union

import numpy as np

from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    CrankNicolsonStep,
    DIRKStep,
    ERKStep,
    ExplicitEulerStep,
    GenericRosenbrockWStep,
    resolve_alias,
    resolve_supplied_tableau,
)
from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRKTableau,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERKTableau,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    RosenbrockTableau,
)

from .cpu_ode_system import CPUODESystem
from .cpu_utils import (
    Array,
    DriverEvaluator,
    StepResult,
    _encode_solver_status,
)


def explicit_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: Optional[int] = None,
    time: float = 0.0,
) -> StepResult:
    """Explicit Euler integration step."""

    drivers_now = driver_evaluator(time)
    drivers_next = driver_evaluator(time + dt)

    observables_now = evaluator.observables(state, params, drivers_now, time)
    dxdt, _ = evaluator.rhs(
        state,
        params,
        drivers_now,
        observables_now,
        time,
    )
    new_state = state + dt * dxdt
    observables = evaluator.observables(
        new_state,
        params,
        drivers_next,
        time + dt,
    )
    error = np.zeros_like(state)
    status = _encode_solver_status(True, 0)
    return StepResult(new_state, observables, error, status, 0)


def _newton_solve(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    initial_guess: Array,
    precision: np.dtype,
    tol: float = 1e-10,
    max_iters: int = 25,
):
    """Solve ``residual(x) = 0`` using a dense Newton iteration."""

    state = initial_guess.astype(precision, copy=True)
    for iteration in range(max_iters):
        res = residual(state)
        res_norm = np.linalg.norm(res, ord=2)
        if res_norm < tol:
            return state, True, iteration + 1
        jac = jacobian(state)
        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac, -res, rcond=None)[0]
        state = state + delta.astype(precision)
    return state, False, max_iters


def backward_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    initial_guess: Optional[Array] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Backward Euler step solved via Newton iteration."""

    precision = evaluator.precision
    next_time = time + precision(dt)

    drivers_next = driver_evaluator(next_time)

    def residual(candidate: Array) -> Array:
        candidate_observables = evaluator.observables(
            candidate,
            params,
            drivers_next,
            next_time,
        )
        dxdt, _ = evaluator.rhs(
            candidate,
            params,
            drivers_next,
            candidate_observables,
            next_time,
        )
        return candidate - state - dt * dxdt

    def jacobian(candidate: Array) -> Array:
        candidate_observables = evaluator.observables(
            candidate,
            params,
            drivers_next,
            next_time,
        )
        jac = evaluator.jacobian(
            candidate,
            params,
            drivers_next,
            candidate_observables,
            next_time,
        )
        identity = np.eye(jac.shape[0], dtype=precision)
        return identity - dt * jac

    if initial_guess is None:
        guess = state.astype(precision, copy=True)
    else:
        guess = np.asarray(initial_guess, dtype=precision).astype(
            precision, copy=True
        )
    next_state, converged, niters = _newton_solve(
        residual,
        jacobian,
        guess,
        precision,
        tol,
        max_iters,
    )
    observables = evaluator.observables(
        next_state,
        params,
        drivers_next,
        time + dt,
    )
    error = np.zeros_like(next_state)
    status = _encode_solver_status(converged, niters)
    return StepResult(next_state, observables, error, status, niters)


def crank_nicolson_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Crank–Nicolson step with embedded backward Euler for error estimation."""

    precision = evaluator.precision
    current_time = precision(time)
    next_time = current_time + precision(dt)
    drivers_now = driver_evaluator(current_time)
    drivers_next = driver_evaluator(next_time)
    observables_now = evaluator.observables(
        state,
        params,
        drivers_now,
        current_time,
    )

    def residual(candidate: Array) -> Array:
        candidate_observables = evaluator.observables(
            candidate,
            params,
            drivers_next,
            next_time,
        )
        f_candidate, _ = evaluator.rhs(
            candidate,
            params,
            drivers_next,
            candidate_observables,
            next_time,
        )
        f_now, _ = evaluator.rhs(
            state,
            params,
            drivers_now,
            observables_now,
            current_time,
        )
        return candidate - state - precision(0.5) * dt * (f_now + f_candidate)

    def jacobian(candidate: Array) -> Array:
        candidate_observables = evaluator.observables(
            candidate,
            params,
            drivers_next,
            next_time,
        )
        jac = evaluator.jacobian(
            candidate,
            params,
            drivers_next,
            candidate_observables,
            next_time,
        )
        identity = np.eye(jac.shape[0], dtype=precision)
        return identity - precision(0.5) * dt * jac

    guess = state.astype(precision, copy=True)
    next_state, converged, niters = _newton_solve(
        residual,
        jacobian,
        guess,
        precision,
        tol,
        max_iters,
    )
    observables = evaluator.observables(
        next_state,
        params,
        drivers_next,
        next_time,
    )

    be_result = backward_euler_step(
        evaluator=evaluator,
        state=state,
        params=params,
        dt=dt,
        tol=tol,
        initial_guess=next_state,
        max_iters=max_iters,
        time=time,
        driver_evaluator=driver_evaluator,
    )
    error = next_state - be_result.state
    status = _encode_solver_status(converged, niters)
    return StepResult(next_state, observables, error, status, niters)


def _tableau_matrix(
    rows: Sequence[Sequence[float]],
    stage_count: int,
    dtype: np.dtype,
) -> Array:
    """Return a dense matrix created from ``rows`` padded with zeros."""

    matrix = np.zeros((stage_count, stage_count), dtype=dtype)
    for row_index, row in enumerate(rows[:stage_count]):
        if not row:
            continue
        padded = np.asarray(row, dtype=dtype)
        limit = min(stage_count, padded.shape[0])
        matrix[row_index, :limit] = padded[:limit]
    return matrix


def _tableau_vector(entries: Sequence[float], dtype: np.dtype) -> Array:
    """Return a one-dimensional array from ``entries`` using ``dtype``."""

    return np.asarray(entries, dtype=dtype)


def erk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: Optional[int] = None,
    time: float = 0.0,
    tableau: Optional[ERKTableau] = None,
) -> StepResult:
    """Execute a generic explicit Runge--Kutta step on the CPU."""

    if state is None or dt is None:
        raise ValueError("State and dt must be provided for ERK steppers.")

    precision = evaluator.precision
    dt_value = precision(dt)
    current_time = precision(time)

    if tableau is None:
        tableau_value = DEFAULT_ERK_TABLEAU
    else:
        tableau_value = tableau

    stage_count = tableau_value.stage_count
    a_matrix = _tableau_matrix(tableau_value.a, stage_count, precision)
    b_weights = _tableau_vector(tableau_value.b, precision)
    if tableau_value.d is None:
        error_weights = np.zeros(stage_count, dtype=precision)
    else:
        error_weights = _tableau_vector(tableau_value.d, precision)
    c_nodes = _tableau_vector(tableau_value.c, precision)

    if params is None:
        params_array = np.zeros(
            evaluator.system.sizes.parameters,
            dtype=precision,
        )
    else:
        params_array = np.asarray(params, dtype=precision)
    state_vector = state.astype(precision, copy=True)
    stage_derivatives = np.zeros(
        (stage_count, state_vector.shape[0]),
        dtype=precision,
    )

    for stage_index in range(stage_count):
        stage_state = state_vector.copy()
        for dependency in range(stage_index):
            stage_state = stage_state + (
                dt_value
                * a_matrix[stage_index, dependency]
                * stage_derivatives[dependency]
            )
        stage_time = current_time + c_nodes[stage_index] * dt_value
        drivers_stage = driver_evaluator(float(stage_time))
        observables_stage = evaluator.observables(
            stage_state,
            params_array,
            drivers_stage,
            stage_time,
        )
        derivative, _ = evaluator.rhs(
            stage_state,
            params_array,
            drivers_stage,
            observables_stage,
            stage_time,
        )
        stage_derivatives[stage_index, :] = derivative

    state_update = np.zeros_like(state_vector)
    for stage_index in range(stage_count):
        state_update = state_update + (
            b_weights[stage_index] * stage_derivatives[stage_index]
        )
    new_state = state_vector + dt_value * state_update

    error_update = np.zeros_like(state_vector)
    for stage_index in range(stage_count):
        error_update = error_update + (
            error_weights[stage_index] * stage_derivatives[stage_index]
        )
    error = dt_value * error_update

    end_time = current_time + dt_value
    drivers_next = driver_evaluator(float(end_time))
    observables = evaluator.observables(
        new_state,
        params_array,
        drivers_next,
        end_time,
    )
    status = _encode_solver_status(True, 0)
    return StepResult(new_state, observables, error, status, 0)


def dirk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Array,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: Optional[int] = None,
    time: float = 0.0,
    tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
) -> StepResult:
    """Execute a generic diagonally implicit Runge--Kutta step on the CPU."""

    precision = evaluator.precision
    dt_value = precision(dt)
    current_time = precision(time)

    tableau_value = tableau

    stage_count = tableau_value.stage_count
    a_matrix = _tableau_matrix(tableau_value.a, stage_count, precision)
    b_weights = _tableau_vector(tableau_value.b, precision)
    if tableau_value.d is None:
        error_weights = np.zeros(stage_count, dtype=precision)
    else:
        error_weights = _tableau_vector(tableau_value.d, precision)
    c_nodes = _tableau_vector(tableau_value.c, precision)

    if params is None:
        params_array = np.zeros(
            evaluator.system.sizes.parameters,
            dtype=precision,
        )
    else:
        params_array = np.asarray(params, dtype=precision)
    state_vector = state.astype(precision, copy=True)
    stage_derivatives = np.zeros(
        (stage_count, state_vector.shape[0]),
        dtype=precision,
    )

    tol_value = 1e-10 if tol is None else float(tol)
    max_iters_value = 25 if max_iters is None else int(max_iters)
    all_converged = True
    total_iters = 0

    for stage_index in range(stage_count):
        stage_state = state_vector.copy()
        for dependency in range(stage_index):
            stage_state = stage_state + (
                dt_value
                * a_matrix[stage_index, dependency]
                * stage_derivatives[dependency]
            )

        stage_time = current_time + c_nodes[stage_index] * dt_value
        drivers_stage = driver_evaluator(float(stage_time))
        diag_coeff = a_matrix[stage_index, stage_index]

        if math.isclose(float(diag_coeff), 0.0):
            stage_observables = evaluator.observables(
                stage_state,
                params_array,
                drivers_stage,
                stage_time,
            )
            derivative, _ = evaluator.rhs(
                stage_state,
                params_array,
                drivers_stage,
                stage_observables,
                stage_time,
            )
            stage_derivatives[stage_index, :] = derivative
            continue

        def residual(candidate: Array) -> Array:
            candidate_observables = evaluator.observables(
                candidate,
                params_array,
                drivers_stage,
                stage_time,
            )
            derivative_value, _ = evaluator.rhs(
                candidate,
                params_array,
                drivers_stage,
                candidate_observables,
                stage_time,
            )
            return (
                candidate
                - stage_state
                - dt_value * diag_coeff * derivative_value
            )

        def jacobian(candidate: Array) -> Array:
            candidate_observables = evaluator.observables(
                candidate,
                params_array,
                drivers_stage,
                stage_time,
            )
            jacobian_value = evaluator.jacobian(
                candidate,
                params_array,
                drivers_stage,
                candidate_observables,
                stage_time,
            )
            identity = np.eye(jacobian_value.shape[0], dtype=precision)
            return identity - dt_value * diag_coeff * jacobian_value

        guess = stage_state.copy()
        solved_state, converged, niters = _newton_solve(
            residual,
            jacobian,
            guess,
            precision,
            tol_value,
            max_iters_value,
        )
        all_converged = all_converged and converged
        total_iters += niters
        stage_observables = evaluator.observables(
            solved_state,
            params_array,
            drivers_stage,
            stage_time,
        )
        derivative, _ = evaluator.rhs(
            solved_state,
            params_array,
            drivers_stage,
            stage_observables,
            stage_time,
        )
        stage_derivatives[stage_index, :] = derivative

    state_update = np.zeros_like(state_vector)
    for stage_index in range(stage_count):
        state_update = state_update + (
            b_weights[stage_index] * stage_derivatives[stage_index]
        )
    new_state = state_vector + dt_value * state_update

    error_update = np.zeros_like(state_vector)
    for stage_index in range(stage_count):
        error_update = error_update + (
            error_weights[stage_index] * stage_derivatives[stage_index]
        )
    error = dt_value * error_update

    end_time = current_time + dt_value
    drivers_next = driver_evaluator(float(end_time))
    observables = evaluator.observables(
        new_state,
        params_array,
        drivers_next,
        end_time,
    )
    status = _encode_solver_status(all_converged, total_iters)
    return StepResult(new_state, observables, error, status, total_iters)


def rosenbrock_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: Optional[int] = None,
    time: float = 0.0,
    tableau: Optional[RosenbrockTableau] = None,
) -> StepResult:
    """Six-stage Rosenbrock-W method with an embedded error estimate."""

    precision = evaluator.precision
    dt_value = precision(dt)
    current_time = precision(time)
    end_time = current_time + dt_value

    tableau_value = tableau
    stage_count = tableau_value.stage_count

    a_matrix = _tableau_matrix(tableau_value.a, stage_count, precision)
    C_matrix = _tableau_matrix(tableau_value.C, stage_count, precision)
    b_weights = _tableau_vector(tableau_value.b, precision)
    error_weights = _tableau_vector(tableau_value.d, precision)
    c_nodes = _tableau_vector(tableau_value.c, precision)
    gamma = precision(tableau_value.gamma)
    zero = precision(0.0)

    drivers_now = driver_evaluator(float(current_time))
    observables_now = evaluator.observables(
        state,
        params,
        drivers_now,
        current_time,
    )
    f_now, _ = evaluator.rhs(
        state,
        params,
        drivers_now,
        observables_now,
        current_time,
    )
    jac = evaluator.jacobian(
        state,
        params,
        drivers_now,
        observables_now,
        current_time,
    )

    identity = np.eye(len(state), dtype=precision)
    lhs = identity - dt_value * gamma * jac

    stage_count = b_weights.size
    state_accum = np.zeros_like(state, dtype=precision)
    error_accum = np.zeros_like(state, dtype=precision)
    state_shifts = np.zeros((stage_count, len(state)), dtype=precision)
    jacobian_shifts = np.zeros_like(state_shifts)

    drivers_stage = drivers_now
    rhs = np.zeros_like(state, dtype=precision)

    for stage_index in range(stage_count):
        stage_time = current_time + c_nodes[stage_index] * dt_value
        if stage_index == 0:
            stage_state = state
            f_stage = f_now
        else:
            drivers_stage = driver_evaluator(float(stage_time))
            stage_state = state + state_shifts[stage_index]
            observables_stage = evaluator.observables(
                stage_state,
                params,
                drivers_stage,
                stage_time,
            )
            f_stage, _ = evaluator.rhs(
                stage_state,
                params,
                drivers_stage,
                observables_stage,
                stage_time,
            )

        rhs[:] = dt_value * f_stage
        if stage_index > 0 and np.any(C_matrix[stage_index, :stage_index]):
            jac_term = jac @ jacobian_shifts[stage_index]
            rhs[:] += dt_value * jac_term

        stage_increment = np.linalg.solve(lhs, rhs).astype(precision)
        state_accum += b_weights[stage_index] * stage_increment
        error_accum += error_weights[stage_index] * stage_increment

        for successor in range(stage_index + 1, stage_count):
            a_coeff = a_matrix[successor, stage_index]
            if a_coeff != zero:
                state_shifts[successor] += a_coeff * stage_increment
            C_coeff = C_matrix[successor, stage_index]
            if C_coeff != zero:
                jacobian_shifts[successor] += C_coeff * stage_increment

    new_state = state + state_accum
    drivers_end = driver_evaluator(float(end_time))
    observables = evaluator.observables(
        new_state,
        params,
        drivers_end,
        end_time,
    )
    status = _encode_solver_status(True, 0)
    return StepResult(new_state, observables, error_accum, status, 0)


def backward_euler_predict_correct_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Predict with explicit Euler and correct with backward Euler."""

    precision = evaluator.precision
    drivers_now = driver_evaluator(time)
    observables_now = evaluator.observables(
        state,
        params,
        drivers_now,
        time,
    )
    f_now, _ = evaluator.rhs(
        state,
        params,
        drivers_now,
        observables_now,
        time,
    )
    predictor = state.astype(precision, copy=True) + dt * f_now
    return backward_euler_step(
        evaluator=evaluator,
        state=state,
        params=params,
        dt=dt,
        tol=tol,
        initial_guess=predictor,
        max_iters=max_iters,
        time=time,
        driver_evaluator=driver_evaluator,
    )


def get_ref_step_function(
    algorithm: str,
    *,
    tableau: Optional[
        Union[ERKTableau, DIRKTableau, RosenbrockTableau]
    ] = None,
) -> Callable:
    """Return the CPU reference implementation for ``algorithm``."""

    key = algorithm.lower()
    constructor_to_cpu = {
        ExplicitEulerStep: explicit_euler_step,
        BackwardsEulerStep: backward_euler_step,
        BackwardsEulerPCStep: backward_euler_predict_correct_step,
        CrankNicolsonStep: crank_nicolson_step,
        ERKStep: erk_step,
        DIRKStep: dirk_step,
        GenericRosenbrockWStep: rosenbrock_step,
    }

    try:
        step_constructor, resolved_tableau = resolve_alias(key)
    except KeyError as exc:
        raise ValueError(f"Unknown stepper algorithm: {algorithm}") from exc

    stepper = constructor_to_cpu.get(step_constructor)
    if stepper is None:
        raise ValueError(
            f"No CPU reference implementation registered for {algorithm}."
        )

    tableau_value = resolved_tableau
    if isinstance(tableau, str):
        try:
            override_constructor, tableau_value = resolve_alias(tableau)
        except KeyError as exc:
            raise ValueError(
                f"Unknown {step_constructor.__name__} tableau '{tableau}'."
            ) from exc
        if tableau_value is None:
            raise ValueError(
                f"Alias '{tableau}' does not reference a tableau."
            )
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau alias does not match the requested algorithm type."
            )
    elif isinstance(tableau, ButcherTableau):
        override_constructor, override_tableau = resolve_supplied_tableau(tableau)
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau instance does not match the requested algorithm type."
            )
        tableau_value = override_tableau
    elif tableau is not None:
        raise TypeError(
            "Expected tableau alias or ButcherTableau instance, "
            f"received {type(tableau).__name__}."
        )

    if tableau_value is None:
        return stepper

    return partial(stepper, tableau=tableau_value)


__all__ = [
    "backward_euler_predict_correct_step",
    "backward_euler_step",
    "crank_nicolson_step",
    "dirk_step",
    "erk_step",
    "explicit_euler_step",
    "get_ref_step_function",
    "rosenbrock_step",
]
