"""CPU reference implementations for integrator step algorithms."""

from functools import partial
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
    DEFAULT_ROSENBROCK_TABLEAU,
    RosenbrockTableau,
)

from .cpu_ode_system import CPUODESystem
from .cpu_utils import (
    Array,
    DriverEvaluator,
    StepResult,
    _encode_solver_status,
)

StepCallable = Callable[..., StepResult]


class CPUStep:
    """Base class for CPU reference integrator steps."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
    ) -> None:
        self.evaluator = evaluator
        self.driver_evaluator = driver_evaluator
        self.precision = evaluator.precision
        self._state_size = evaluator.system.sizes.states
        self._identity = np.eye(self._state_size, dtype=self.precision)

    def __call__(self, **kwargs: object) -> StepResult:
        return self.step(**kwargs)

    def step(self, **_: object) -> StepResult:
        """Execute a single integration step."""

        raise NotImplementedError

    def ensure_array(
        self,
        vector: Optional[Sequence[float] | Array],
        *,
        copy: bool = False,
    ) -> Array:
        if vector is None:
            return np.zeros(0, dtype=self.precision)
        array = np.asarray(vector, dtype=self.precision)
        if array.ndim != 1:
            raise ValueError("Expected a one-dimensional array.")
        if copy:
            return array.copy()
        return array

    def drivers(self, time: np.floating) -> Array:
        return self.driver_evaluator(self.precision(time))

    def observables(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time: np.floating,
    ) -> Array:
        return self.evaluator.observables(state, params, drivers, time)

    def rhs(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time: np.floating,
    ) -> Array:
        derivative, _ = self.evaluator.rhs(
            state,
            params,
            drivers,
            observables,
            time,
        )
        return derivative

    def observables_and_jac(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time: np.floating,
    ) -> tuple[Array, Array]:
        observables = self.evaluator.observables(
            state,
            params,
            drivers,
            time,
        )
        jacobian = self.evaluator.jacobian(
            state,
            params,
            drivers,
            observables,
            time,
        )
        return observables, jacobian

    def newton_solve(
        self,
        initial_guess: Array,
        tol: Optional[float],
        max_iters: Optional[int],
    ) -> tuple[Array, bool, int]:
        tol_value = (
            self.precision(1e-10) if tol is None else self.precision(tol)
        )
        max_iters_value = 25 if max_iters is None else int(max_iters)
        state = self.ensure_array(initial_guess, copy=True)
        for iteration in range(max_iters_value):
            residual = self.residual(state)
            norm = np.linalg.norm(residual, ord=2)
            if norm < tol_value:
                return state, True, iteration + 1
            jacobian = self.jacobian(state)
            try:
                delta = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            state = state + np.asarray(delta, dtype=self.precision)
        return state, False, max_iters_value

    def residual(self, candidate: Array) -> Array:
        raise NotImplementedError

    def jacobian(self, candidate: Array) -> Array:
        raise NotImplementedError

    def _status(self, converged: bool, niters: int) -> int:
        return _encode_solver_status(converged, niters)


class CPUExplicitEulerStep(CPUStep):
    """Explicit Euler step implementation."""

    def residual(self, candidate: Array) -> Array:
        raise NotImplementedError

    def jacobian(self, candidate: Array) -> Array:
        raise NotImplementedError

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        new_state = state_vector + dt_value * derivative

        next_time = current_time + dt_value
        drivers_next = self.drivers(next_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            next_time,
        )
        error = np.zeros_like(state_vector, dtype=self.precision)
        status = self._status(True, 0)
        return StepResult(new_state, observables, error, status, 0)


class CPUBackwardEulerStep(CPUStep):
    """Backward Euler step solved with Newton iterations."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
    ) -> None:
        super().__init__(evaluator, driver_evaluator)
        self._be_state = np.zeros(self._state_size, dtype=self.precision)
        self._be_params = np.zeros(0, dtype=self.precision)
        self._be_drivers = np.zeros(0, dtype=self.precision)
        self._be_time = self.precision(0.0)
        self._be_dt = self.precision(0.0)

    def residual(self, candidate: Array) -> Array:
        observables = self.observables(
            candidate,
            self._be_params,
            self._be_drivers,
            self._be_time,
        )
        derivative = self.rhs(
            candidate,
            self._be_params,
            self._be_drivers,
            observables,
            self._be_time,
        )
        return candidate - self._be_state - self._be_dt * derivative

    def jacobian(self, candidate: Array) -> Array:
        observables, jacobian = self.observables_and_jac(
            candidate,
            self._be_params,
            self._be_drivers,
            self._be_time,
        )
        return self._identity - self._be_dt * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        initial_guess: Optional[Array] = None,
        max_iters: int = 25,
        time: float = 0.0,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        next_time = current_time + dt_value

        drivers_next = self.drivers(next_time)
        self._be_state = state_vector
        self._be_params = params_array
        self._be_drivers = drivers_next
        self._be_time = next_time
        self._be_dt = dt_value

        if initial_guess is None:
            guess = state_vector.copy()
        else:
            guess = self.ensure_array(initial_guess, copy=True)

        next_state, converged, niters = self.newton_solve(
            guess,
            tol,
            max_iters,
        )
        observables = self.observables(
            next_state,
            params_array,
            drivers_next,
            next_time,
        )
        error = np.zeros_like(next_state, dtype=self.precision)
        status = self._status(converged, niters)
        return StepResult(next_state, observables, error, status, niters)


class CPUCrankNicolsonStep(CPUStep):
    """Crank--Nicolson step with backward Euler error estimate."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        backward_step: Optional[CPUBackwardEulerStep] = None,
    ) -> None:
        super().__init__(evaluator, driver_evaluator)
        self._cn_previous_state = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._cn_previous_derivative = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._cn_params = np.zeros(0, dtype=self.precision)
        self._cn_drivers_next = np.zeros(0, dtype=self.precision)
        self._cn_next_time = self.precision(0.0)
        self._cn_half_dt = self.precision(0.0)
        if backward_step is None:
            backward_step = CPUBackwardEulerStep(evaluator, driver_evaluator)
        self._backward = backward_step

    def residual(self, candidate: Array) -> Array:
        observables = self.observables(
            candidate,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        derivative = self.rhs(
            candidate,
            self._cn_params,
            self._cn_drivers_next,
            observables,
            self._cn_next_time,
        )
        return (
            candidate
            - self._cn_previous_state
            - self._cn_half_dt
            * (self._cn_previous_derivative + derivative)
        )

    def jacobian(self, candidate: Array) -> Array:
        observables, jacobian = self.observables_and_jac(
            candidate,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        return self._identity - self._cn_half_dt * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: int = 25,
        time: float = 0.0,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        next_time = current_time + dt_value

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative_now = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )

        drivers_next = self.drivers(next_time)
        self._cn_previous_state = state_vector
        self._cn_previous_derivative = derivative_now
        self._cn_params = params_array
        self._cn_drivers_next = drivers_next
        self._cn_next_time = next_time
        self._cn_half_dt = self.precision(0.5) * dt_value

        guess = state_vector.copy()
        next_state, converged, niters = self.newton_solve(
            guess,
            tol,
            max_iters,
        )
        observables = self.observables(
            next_state,
            params_array,
            drivers_next,
            next_time,
        )
        backward_result = self._backward(
            state=state_vector,
            params=params_array,
            dt=dt_value,
            tol=tol,
            initial_guess=next_state,
            max_iters=max_iters,
            time=current_time,
        )
        error = next_state - backward_result.state
        status = self._status(converged, niters)
        return StepResult(next_state, observables, error, status, niters)


class CPUERKStep(CPUStep):
    """Explicit Runge--Kutta step implementation."""

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
        tableau: Optional[ERKTableau] = None,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        tableau_value = DEFAULT_ERK_TABLEAU if tableau is None else tableau
        stage_count = tableau_value.stage_count
        a_matrix = _tableau_matrix(
            tableau_value.a,
            stage_count,
            self.precision,
        )
        b_weights = _tableau_vector(tableau_value.b, self.precision)
        if tableau_value.d is None:
            error_weights = np.zeros(stage_count, dtype=self.precision)
        else:
            error_weights = _tableau_vector(tableau_value.d, self.precision)
        c_nodes = _tableau_vector(tableau_value.c, self.precision)

        stage_derivatives = np.zeros(
            (stage_count, state_vector.shape[0]),
            dtype=self.precision,
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
            drivers_stage = self.drivers(stage_time)
            observables_stage = self.observables(
                stage_state,
                params_array,
                drivers_stage,
                stage_time,
            )
            derivative = self.rhs(
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
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )
        status = self._status(True, 0)
        return StepResult(new_state, observables, error, status, 0)


class CPUDIRKStep(CPUStep):
    """Diagonally implicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
    ) -> None:
        super().__init__(evaluator, driver_evaluator)
        self._dirk_reference = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._dirk_params = np.zeros(0, dtype=self.precision)
        self._dirk_drivers = np.zeros(0, dtype=self.precision)
        self._dirk_time = self.precision(0.0)
        self._dirk_dt_coeff = self.precision(0.0)

    def residual(self, candidate: Array) -> Array:
        observables = self.observables(
            candidate,
            self._dirk_params,
            self._dirk_drivers,
            self._dirk_time,
        )
        derivative = self.rhs(
            candidate,
            self._dirk_params,
            self._dirk_drivers,
            observables,
            self._dirk_time,
        )
        return candidate - self._dirk_reference - self._dirk_dt_coeff * derivative

    def jacobian(self, candidate: Array) -> Array:
        observables, jacobian = self.observables_and_jac(
            candidate,
            self._dirk_params,
            self._dirk_drivers,
            self._dirk_time,
        )
        return self._identity - self._dirk_dt_coeff * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        tableau_value = tableau
        stage_count = tableau_value.stage_count
        a_matrix = _tableau_matrix(
            tableau_value.a,
            stage_count,
            self.precision,
        )
        b_weights = _tableau_vector(tableau_value.b, self.precision)
        if tableau_value.d is None:
            error_weights = np.zeros(stage_count, dtype=self.precision)
        else:
            error_weights = _tableau_vector(tableau_value.d, self.precision)
        c_nodes = _tableau_vector(tableau_value.c, self.precision)

        stage_derivatives = np.zeros(
            (stage_count, state_vector.shape[0]),
            dtype=self.precision,
        )

        tol_value = (
            self.precision(1e-10)
            if tol is None
            else self.precision(tol)
        )
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
            drivers_stage = self.drivers(stage_time)
            diag_coeff = a_matrix[stage_index, stage_index]

            if np.isclose(diag_coeff, self.precision(0.0)):
                observables_stage = self.observables(
                    stage_state,
                    params_array,
                    drivers_stage,
                    stage_time,
                )
                derivative = self.rhs(
                    stage_state,
                    params_array,
                    drivers_stage,
                    observables_stage,
                    stage_time,
                )
                stage_derivatives[stage_index, :] = derivative
                continue

            self._dirk_reference = stage_state
            self._dirk_params = params_array
            self._dirk_drivers = drivers_stage
            self._dirk_time = stage_time
            self._dirk_dt_coeff = dt_value * diag_coeff

            guess = stage_state.copy()
            solved_state, converged, niters = self.newton_solve(
                guess,
                tol_value,
                max_iters_value,
            )
            all_converged = all_converged and converged
            total_iters += niters
            observables_stage = self.observables(
                solved_state,
                params_array,
                drivers_stage,
                stage_time,
            )
            derivative = self.rhs(
                solved_state,
                params_array,
                drivers_stage,
                observables_stage,
                stage_time,
            )
            stage_derivatives[stage_index, :] = derivative

        state_accum = np.zeros_like(state_vector)
        error_accum = np.zeros_like(state_vector)
        for stage_index in range(stage_count):
            state_accum = state_accum + (
                b_weights[stage_index] * stage_derivatives[stage_index]
            )
            error_accum = error_accum + (
                error_weights[stage_index] * stage_derivatives[stage_index]
            )

        new_state = state_vector + dt_value * state_accum
        end_time = current_time + dt_value
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )
        status = self._status(all_converged, total_iters)
        return StepResult(
            new_state,
            observables,
            error_accum,
            status,
            total_iters,
        )


class CPURosenbrockWStep(CPUStep):
    """Rosenbrock--W step implementation."""

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
        tableau: Optional[RosenbrockTableau] = None,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        end_time = current_time + dt_value

        tableau_value = (
            DEFAULT_ROSENBROCK_TABLEAU if tableau is None else tableau
        )
        stage_count = tableau_value.stage_count
        a_matrix = _tableau_matrix(
            tableau_value.a,
            stage_count,
            self.precision,
        )
        C_matrix = _tableau_matrix(
            tableau_value.C,
            stage_count,
            self.precision,
        )
        b_weights = _tableau_vector(tableau_value.b, self.precision)
        error_weights = _tableau_vector(tableau_value.d, self.precision)
        c_nodes = _tableau_vector(tableau_value.c, self.precision)
        gamma = self.precision(tableau_value.gamma)
        zero = self.precision(0.0)

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative_now = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        jacobian_now = self.evaluator.jacobian(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )

        lhs = self._identity - dt_value * gamma * jacobian_now

        state_accum = np.zeros_like(state_vector)
        error_accum = np.zeros_like(state_vector)
        state_shifts = np.zeros(
            (stage_count, len(state_vector)),
            dtype=self.precision,
        )
        jacobian_shifts = np.zeros_like(state_shifts)
        rhs = np.zeros_like(state_vector)

        drivers_stage = drivers_now
        for stage_index in range(stage_count):
            stage_time = current_time + c_nodes[stage_index] * dt_value
            if stage_index == 0:
                stage_state = state_vector
                derivative_stage = derivative_now
            else:
                drivers_stage = self.drivers(stage_time)
                stage_state = state_vector + state_shifts[stage_index]
                observables_stage = self.observables(
                    stage_state,
                    params_array,
                    drivers_stage,
                    stage_time,
                )
                derivative_stage = self.rhs(
                    stage_state,
                    params_array,
                    drivers_stage,
                    observables_stage,
                    stage_time,
                )

            rhs[:] = dt_value * derivative_stage
            if stage_index > 0 and np.any(C_matrix[stage_index, :stage_index]):
                jac_term = jacobian_now @ jacobian_shifts[stage_index]
                rhs[:] = rhs + dt_value * jac_term

            stage_increment = np.linalg.solve(lhs, rhs)
            stage_increment = np.asarray(stage_increment, dtype=self.precision)
            state_accum = state_accum + (
                b_weights[stage_index] * stage_increment
            )
            error_accum = error_accum + (
                error_weights[stage_index] * stage_increment
            )

            for successor in range(stage_index + 1, stage_count):
                a_coeff = a_matrix[successor, stage_index]
                if a_coeff != zero:
                    state_shifts[successor] = (
                        state_shifts[successor] + a_coeff * stage_increment
                    )
                C_coeff = C_matrix[successor, stage_index]
                if C_coeff != zero:
                    jacobian_shifts[successor] = (
                        jacobian_shifts[successor] + C_coeff * stage_increment
                    )

        new_state = state_vector + state_accum
        drivers_end = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_end,
            end_time,
        )
        status = self._status(True, 0)
        return StepResult(new_state, observables, error_accum, status, 0)


class CPUBackwardEulerPCStep(CPUStep):
    """Backward Euler predictor--corrector step."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        corrector: Optional[CPUBackwardEulerStep] = None,
    ) -> None:
        super().__init__(evaluator, driver_evaluator)
        if corrector is None:
            corrector = CPUBackwardEulerStep(evaluator, driver_evaluator)
        self._corrector = corrector

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: int = 25,
        time: float = 0.0,
    ) -> StepResult:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative_now = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        predictor = state_vector + dt_value * derivative_now
        return self._corrector(
            state=state_vector,
            params=params_array,
            dt=dt_value,
            tol=tol,
            initial_guess=predictor,
            max_iters=max_iters,
            time=current_time,
        )


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


def explicit_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single explicit Euler step."""

    stepper = CPUExplicitEulerStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def backward_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single backward Euler step."""

    stepper = CPUBackwardEulerStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def crank_nicolson_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single Crank--Nicolson step."""

    stepper = CPUCrankNicolsonStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def erk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single explicit Runge--Kutta step."""

    stepper = CPUERKStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def dirk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single DIRK step."""

    stepper = CPUDIRKStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def rosenbrock_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a single Rosenbrock--W step."""

    stepper = CPURosenbrockWStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


def backward_euler_predict_correct_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    **kwargs: object,
) -> StepResult:
    """Execute a backward Euler predictor--corrector step."""

    stepper = CPUBackwardEulerPCStep(evaluator, driver_evaluator)
    return stepper(**kwargs)


_STEP_CONSTRUCTOR_TO_CLASS = {
    ExplicitEulerStep: CPUExplicitEulerStep,
    BackwardsEulerStep: CPUBackwardEulerStep,
    BackwardsEulerPCStep: CPUBackwardEulerPCStep,
    CrankNicolsonStep: CPUCrankNicolsonStep,
    ERKStep: CPUERKStep,
    DIRKStep: CPUDIRKStep,
    GenericRosenbrockWStep: CPURosenbrockWStep,
}


def _resolve_step_configuration(
    algorithm: str,
    tableau: Optional[Union[str, ButcherTableau]],
) -> tuple[type, Optional[ButcherTableau]]:
    """Resolve the step constructor and tableau for ``algorithm``."""

    try:
        step_constructor, resolved_tableau = resolve_alias(algorithm.lower())
    except KeyError as exc:
        raise ValueError(f"Unknown stepper algorithm: {algorithm}") from exc

    tableau_value = resolved_tableau
    if isinstance(tableau, str):
        try:
            override_constructor, tableau_override = resolve_alias(tableau)
        except KeyError as exc:
            raise ValueError(
                f"Unknown {step_constructor.__name__} tableau '{tableau}'."
            ) from exc
        if tableau_override is None:
            raise ValueError(
                f"Alias '{tableau}' does not reference a tableau."
            )
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau alias does not match the requested algorithm type."
            )
        tableau_value = tableau_override
    elif isinstance(tableau, ButcherTableau):
        override_constructor, override_tableau = resolve_supplied_tableau(
            tableau
        )
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

    return step_constructor, tableau_value


def get_ref_step_factory(
    algorithm: str,
    *,
    tableau: Optional[Union[str, ButcherTableau]] = None,
) -> Callable[[CPUODESystem, DriverEvaluator], StepCallable]:
    """Return a factory binding the CPU stepper for ``algorithm``."""

    step_constructor, tableau_value = _resolve_step_configuration(
        algorithm,
        tableau,
    )
    try:
        step_class = _STEP_CONSTRUCTOR_TO_CLASS[step_constructor]
    except KeyError as exc:
        raise ValueError(
            f"No CPU reference implementation registered for {algorithm}."
        ) from exc

    def factory(
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
    ) -> StepCallable:
        stepper = step_class(evaluator, driver_evaluator)
        if tableau_value is not None:
            return partial(stepper, tableau=tableau_value)

        return stepper

    return factory


def get_ref_stepper(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    algorithm: str,
    *,
    tableau: Optional[Union[str, ButcherTableau]] = None,
) -> StepCallable:
    """Return a configured CPU reference stepper for ``algorithm``."""

    factory = get_ref_step_factory(algorithm, tableau=tableau)
    return factory(evaluator, driver_evaluator)


def get_ref_step_function(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    step_constructor,
    *,
    tableau: Optional[ButcherTableau] = None,
) -> StepCallable:
    """Return the CPU stepper matching ``step_constructor``."""

    try:
        step_class = _STEP_CONSTRUCTOR_TO_CLASS[step_constructor]
    except KeyError as exc:
        raise ValueError(
            "Requested step constructor has no CPU reference implementation."
        ) from exc
    stepper = step_class(evaluator, driver_evaluator)
    if tableau is not None:
        return partial(stepper, tableau=tableau)
    return stepper
