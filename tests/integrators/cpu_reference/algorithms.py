"""CPU reference implementations for integrator step algorithms."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from cubie.integrators import IntegratorReturnCodes
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
    DIRKTableau,
    DEFAULT_DIRK_TABLEAU,
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
    StepResultLike,
    euclidean_norm,
    make_step_result,
    _encode_solver_status,
    krylov_solve,
)

StepCallable = Callable[..., StepResultLike]


@dataclass(slots=True)
class LoggingBuffers:
    """Collection of zero-initialised arrays used for instrumentation."""

    stage_count: int
    residuals: Array
    jacobian_updates: Array
    stage_states: Array
    stage_derivatives: Array
    stage_observables: Array
    stage_drivers: Array
    stage_increments: Array
    solver_initial_guesses: Array
    solver_solutions: Array
    solver_iterations: NDArray[np.int_]
    solver_status: NDArray[np.int_]
    newton_initial_guesses: Array
    newton_iteration_guesses: Array
    newton_residuals: Array
    newton_squared_norms: Array
    linear_initial_guesses: Array
    linear_iteration_guesses: Array
    linear_residuals: Array
    linear_squared_norms: Array
    linear_preconditioned_vectors: Array


class CPUStep:
    """Base class for CPU reference integrator steps."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        instrument: bool = False,
        tableau: Optional[ButcherTableau] = None,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        self.evaluator = evaluator
        self.driver_evaluator = driver_evaluator
        self.precision = evaluator.precision
        self._state_size = evaluator.system.sizes.states
        self._identity = np.eye(self._state_size, dtype=self.precision)
        self._newton_tol = self.precision(newton_tol)
        self._newton_max_iters = int(newton_max_iters)
        self._linear_tol = self.precision(linear_tol)
        self._linear_max_iters = int(linear_max_iters)
        correction = str(linear_correction_type)
        if correction not in {
            "steepest_descent",
            "minimal_residual",
        }:
            raise ValueError(
                "Correction type must be 'steepest_descent' or 'minimal_residual'."
            )
        self._linear_correction_type = correction
        self._preconditioner_order = int(preconditioner_order)
        if self._preconditioner_order < 0:
            raise ValueError("Preconditioner order must be non-negative.")
        self._newton_damping = self.precision(newton_damping)
        self._newton_max_backtracks = int(newton_max_backtracks)
        self.tableau = tableau
        self.instrument = instrument

    def _tableau_rows(
        self, values: Optional[Sequence[Sequence[float]]]
    ) -> Optional[Array]:
        """Return ``values`` typed to the step precision when available."""

        if values is None or self.tableau is None:
            return None
        typed_rows = self.tableau.typed_rows(values, self.precision)
        return np.asarray(typed_rows, dtype=self.precision)

    def _tableau_vector(
        self, values: Optional[Sequence[float]]
    ) -> Optional[Array]:
        """Return ``values`` typed to the step precision when available."""

        if values is None or self.tableau is None:
            return None
        typed_values = self.tableau.typed_vector(values, self.precision)
        return np.asarray(typed_values, dtype=self.precision)

    @property
    def stage_count(self) -> Optional[int]:
        """Return the number of stages described by the tableau."""

        if self.tableau is None:
            return None
        return getattr(self.tableau, "stage_count", None)

    @property
    def first_same_as_last(self) -> Optional[bool]:
        """Return whether the tableau shares its first and last stage."""

        if self.tableau is None:
            return None
        return getattr(self.tableau, "first_same_as_last", None)

    @property
    def a_matrix(self) -> Optional[Array]:
        """Return the stage coupling coefficients."""

        if self.tableau is None:
            return None
        return self._tableau_rows(getattr(self.tableau, "a", None))

    @property
    def b_weights(self) -> Optional[Array]:
        """Return the weights for combining stage derivatives."""

        if self.tableau is None:
            return None
        return self._tableau_vector(getattr(self.tableau, "b", None))

    @property
    def c_nodes(self) -> Optional[Array]:
        """Return the substage time nodes."""

        if self.tableau is None:
            return None
        return self._tableau_vector(getattr(self.tableau, "c", None))

    @property
    def error_weights(self) -> Optional[Array]:
        """Return embedded error weights when available."""

        if self.tableau is None:
            return None
        weights = self.tableau.error_weights(self.precision)
        if weights is None:
            return None
        return np.asarray(weights, dtype=self.precision)

    def _create_logging_buffers(self, stage_count: int) -> LoggingBuffers:
        """Return logging buffers sized for ``stage_count`` stages."""

        resolved_count = max(stage_count, 1)
        state_dim = self._state_size
        observable_dim = self.evaluator.system.sizes.observables
        driver_dim = self.evaluator.system.sizes.drivers
        newton_slots = (
            self._newton_max_iters * (self._newton_max_backtracks + 1)
            + 1
        )
        linear_slots = resolved_count * max(self._newton_max_iters, 1)
        linear_iters = max(self._linear_max_iters, 1)

        residuals = np.zeros((resolved_count, state_dim), dtype=self.precision)
        jacobian_updates = np.zeros_like(residuals)
        stage_states = np.zeros_like(residuals)
        stage_derivatives = np.zeros_like(residuals)
        stage_observables = np.zeros(
            (resolved_count, observable_dim),
            dtype=self.precision,
        )
        stage_drivers = np.zeros((resolved_count, driver_dim), dtype=self.precision)
        stage_increments = np.zeros_like(residuals)
        solver_initial_guesses = np.zeros_like(residuals)
        solver_solutions = np.zeros_like(residuals)
        solver_iterations = np.zeros(resolved_count, dtype=np.int64)
        solver_status = np.zeros(resolved_count, dtype=np.int64)
        newton_initial_guesses = np.zeros_like(residuals)
        newton_iteration_guesses = np.zeros(
            (resolved_count, newton_slots, state_dim),
            dtype=self.precision,
        )
        newton_residuals = np.zeros_like(newton_iteration_guesses)
        newton_squared_norms = np.zeros(
            (resolved_count, newton_slots),
            dtype=self.precision,
        )
        linear_initial_guesses = np.zeros(
            (linear_slots, state_dim),
            dtype=self.precision,
        )
        linear_iteration_guesses = np.zeros(
            (linear_slots, linear_iters, state_dim),
            dtype=self.precision,
        )
        linear_residuals = np.zeros_like(linear_iteration_guesses)
        linear_squared_norms = np.zeros(
            (linear_slots, linear_iters),
            dtype=self.precision,
        )
        linear_preconditioned_vectors = np.zeros_like(linear_iteration_guesses)

        return LoggingBuffers(
            stage_count=resolved_count,
            residuals=residuals,
            jacobian_updates=jacobian_updates,
            stage_states=stage_states,
            stage_derivatives=stage_derivatives,
            stage_observables=stage_observables,
            stage_drivers=stage_drivers,
            stage_increments=stage_increments,
            solver_initial_guesses=solver_initial_guesses,
            solver_solutions=solver_solutions,
            solver_iterations=solver_iterations,
            solver_status=solver_status,
            newton_initial_guesses=newton_initial_guesses,
            newton_iteration_guesses=newton_iteration_guesses,
            newton_residuals=newton_residuals,
            newton_squared_norms=newton_squared_norms,
            linear_initial_guesses=linear_initial_guesses,
            linear_iteration_guesses=linear_iteration_guesses,
            linear_residuals=linear_residuals,
            linear_squared_norms=linear_squared_norms,
            linear_preconditioned_vectors=linear_preconditioned_vectors,
        )

    def _build_newton_logging_kwargs(
        self,
        *,
        stage_index: int,
        logging: Optional[LoggingBuffers],
    ) -> dict[str, object]:
        """Return keyword arguments enabling Newton logging when requested."""

        if logging is None:
            return {"instrumented": False}
        return {
            "stage_index": stage_index,
            "instrumented": True,
            "newton_initial_guesses": logging.newton_initial_guesses,
            "newton_iteration_guesses": logging.newton_iteration_guesses,
            "newton_residuals": logging.newton_residuals,
            "newton_squared_norms": logging.newton_squared_norms,
            "linear_initial_guesses": logging.linear_initial_guesses,
            "linear_iteration_guesses": logging.linear_iteration_guesses,
            "linear_residuals": logging.linear_residuals,
            "linear_squared_norms": logging.linear_squared_norms,
            "linear_preconditioned_vectors": (
                logging.linear_preconditioned_vectors
            ),
        }

    def _logging_result_kwargs(
        self,
        logging: Optional[LoggingBuffers],
    ) -> dict[str, object]:
        """Return keyword arguments for :func:`make_step_result`."""

        if logging is None:
            return {}
        return {
            "stage_count": logging.stage_count,
            "residuals": logging.residuals,
            "jacobian_updates": logging.jacobian_updates,
            "stage_states": logging.stage_states,
            "stage_derivatives": logging.stage_derivatives,
            "stage_observables": logging.stage_observables,
            "stage_drivers": logging.stage_drivers,
            "stage_increments": logging.stage_increments,
            "solver_initial_guesses": logging.solver_initial_guesses,
            "solver_solutions": logging.solver_solutions,
            "solver_iterations": logging.solver_iterations,
            "solver_status": logging.solver_status,
            "newton_initial_guesses": logging.newton_initial_guesses,
            "newton_iteration_guesses": logging.newton_iteration_guesses,
            "newton_residuals": logging.newton_residuals,
            "newton_squared_norms": logging.newton_squared_norms,
            "linear_initial_guesses": logging.linear_initial_guesses,
            "linear_iteration_guesses": logging.linear_iteration_guesses,
            "linear_residuals": logging.linear_residuals,
            "linear_squared_norms": logging.linear_squared_norms,
            "linear_preconditioned_vectors": (
                logging.linear_preconditioned_vectors
            ),
        }

    def __call__(self, **kwargs: object) -> StepResultLike:
        return self.step(**kwargs)

    def step(self, **_: object) -> StepResultLike:
        """Execute a single integration step."""

        raise NotImplementedError

    def _make_result(
        self,
        *,
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
        solver_initial_guesses: Optional[Array] = None,
        solver_solutions: Optional[Array] = None,
        solver_iterations: Optional[Sequence[int] | np.ndarray] = None,
        solver_status: Optional[Sequence[int] | np.ndarray] = None,
        newton_initial_guesses: Optional[Array] = None,
        newton_iteration_guesses: Optional[Array] = None,
        newton_residuals: Optional[Array] = None,
        newton_squared_norms: Optional[Array] = None,
        linear_initial_guesses: Optional[Array] = None,
        linear_iteration_guesses: Optional[Array] = None,
        linear_residuals: Optional[Array] = None,
        linear_squared_norms: Optional[Array] = None,
        linear_preconditioned_vectors: Optional[Array] = None,
        extra_vectors: Optional[dict[str, Array]] = None,
    ) -> StepResultLike:
        """Return a step result honoring the instrumentation setting."""

        return make_step_result(
            instrument=self.instrument,
            state=state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
            stage_count=stage_count,
            residuals=residuals,
            jacobian_updates=jacobian_updates,
            stage_states=stage_states,
            stage_derivatives=stage_derivatives,
            stage_observables=stage_observables,
            stage_drivers=stage_drivers,
            stage_increments=stage_increments,
            solver_initial_guesses=solver_initial_guesses,
            solver_solutions=solver_solutions,
            solver_iterations=solver_iterations,
            solver_status=solver_status,
            newton_initial_guesses=newton_initial_guesses,
            newton_iteration_guesses=newton_iteration_guesses,
            newton_residuals=newton_residuals,
            newton_squared_norms=newton_squared_norms,
            linear_initial_guesses=linear_initial_guesses,
            linear_iteration_guesses=linear_iteration_guesses,
            linear_residuals=linear_residuals,
            linear_squared_norms=linear_squared_norms,
            linear_preconditioned_vectors=linear_preconditioned_vectors,
            extra_vectors=extra_vectors,
        )

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

    def mass_matrix_apply(self, vector: Array) -> Array:
        return vector

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
        *,
        stage_index: int = 0,
        instrumented: bool = False,
        newton_initial_guesses: Optional[Array] = None,
        newton_iteration_guesses: Optional[Array] = None,
        newton_residuals: Optional[Array] = None,
        newton_squared_norms: Optional[Array] = None,
        linear_initial_guesses: Optional[Array] = None,
        linear_iteration_guesses: Optional[Array] = None,
        linear_residuals: Optional[Array] = None,
        linear_squared_norms: Optional[Array] = None,
        linear_preconditioned_vectors: Optional[Array] = None,
    ) -> tuple[Array, bool, int]:
        state = self.ensure_array(initial_guess, copy=True)
        residual = self.residual(state)
        norm = euclidean_norm(residual, precision=self.precision)
        norm_squared = norm * norm
        if instrumented and newton_initial_guesses is not None:
            newton_initial_guesses[stage_index, :] = state

        newton_log_index = 0

        def _log_newton(
            index: int,
            candidate: Array,
            residual_vec: Array,
            squared_norm: np.floating,
        ) -> None:
            if not instrumented or index < 0:
                return
            if (
                newton_iteration_guesses is not None
                and index < newton_iteration_guesses.shape[1]
            ):
                newton_iteration_guesses[stage_index, index, :] = candidate
            if newton_residuals is not None and index < newton_residuals.shape[1]:
                newton_residuals[stage_index, index, :] = residual_vec
            if (
                newton_squared_norms is not None
                and index < newton_squared_norms.shape[1]
            ):
                newton_squared_norms[stage_index, index] = squared_norm

        _log_newton(newton_log_index, state, residual, norm_squared)
        newton_log_index += 1

        if norm <= self._newton_tol:
            return state, True, 0

        for iteration in range(self._newton_max_iters):
            jacobian = self.jacobian(state)
            linear_kwargs = {}
            if instrumented:
                slot = stage_index * max(self._newton_max_iters, 1) + iteration
                linear_kwargs = {
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

            direction, converged, _ = self.linear_solve(
                jacobian,
                -residual,
                **linear_kwargs,
            )
            if not converged:
                return state, False, iteration + 1

            step = np.asarray(direction, dtype=self.precision)
            scale = self.precision(1.0)
            accepted = False

            for _ in range(self._newton_max_backtracks + 1):
                trial_state = state + scale * step
                trial_residual = self.residual(trial_state)
                trial_norm = euclidean_norm(
                    trial_residual,
                    precision=self.precision,
                )
                trial_squared = trial_norm * trial_norm
                _log_newton(newton_log_index, trial_state, trial_residual, trial_squared)
                newton_log_index += 1
                if trial_norm <= self._newton_tol:
                    return trial_state, True, iteration + 1
                if trial_norm < norm:
                    state = trial_state
                    residual = trial_residual
                    norm = trial_norm
                    norm_squared = trial_squared
                    accepted = True
                    break
                scale = scale * self._newton_damping

            if not accepted:
                return state, False, iteration + 1

        return state, False, self._newton_max_iters

    def linear_solve(
        self,
        matrix: Array,
        rhs: Array,
        *,
        initial_guess: Optional[Array] = None,
        stage_index: int = 0,
        instrumented: bool = False,
        logging_initial_guess: Optional[Array] = None,
        logging_iteration_guesses: Optional[Array] = None,
        logging_residuals: Optional[Array] = None,
        logging_squared_norms: Optional[Array] = None,
        logging_preconditioned_vectors: Optional[Array] = None,
    ) -> tuple[Array, bool, int]:
        coefficients = np.asarray(matrix, dtype=self.precision)
        vector = np.asarray(rhs, dtype=self.precision)

        def operator_apply(candidate: Array) -> Array:
            return coefficients @ candidate

        solution, converged, niters = krylov_solve(
            vector,
            operator_apply=operator_apply,
            tolerance=self._linear_tol,
            max_iterations=self._linear_max_iters,
            precision=self.precision,
            neumann_order=self._preconditioner_order,
            correction_type=self._linear_correction_type,
            initial_guess=initial_guess,
            instrumented=instrumented,
            logging_initial_guess=logging_initial_guess,
            logging_iteration_guesses=logging_iteration_guesses,
            logging_residuals=logging_residuals,
            logging_squared_norms=logging_squared_norms,
            logging_preconditioned_vectors=logging_preconditioned_vectors,
            stage_index=stage_index,
        )
        return np.asarray(solution, dtype=self.precision), converged, niters

    @property
    def linear_correction_type(self) -> str:
        """Return the Krylov correction strategy used by the step."""

        return self._linear_correction_type

    @property
    def preconditioner_order(self) -> int:
        """Return the Neumann-series preconditioner order."""

        return self._preconditioner_order

    @property
    def newton_tolerance(self) -> np.floating:
        """Return the Newton iteration convergence tolerance."""

        return self._newton_tol

    @property
    def max_newton_iterations(self) -> int:
        """Return the Newton iteration cap."""

        return self._newton_max_iters

    @property
    def linear_tolerance(self) -> np.floating:
        """Return the Krylov solver convergence tolerance."""

        return self._linear_tol

    @property
    def max_linear_iterations(self) -> int:
        """Return the Krylov iteration cap."""

        return self._linear_max_iters

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
        time: float = 0.0,
    ) -> StepResultLike:
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
        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=1)
            logging.stage_states[0, :] = state_vector
            logging.stage_derivatives[0, :] = derivative
            logging.stage_observables[0, :] = observables_now
            logging.solver_solutions[0, :] = state_vector
            logging.solver_status[0] = int(IntegratorReturnCodes.SUCCESS)
            logging.stage_drivers[0, :] = drivers_now
            logging.stage_increments[0, :] = dt_value * derivative
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = (
            logging.stage_derivatives if logging else None
        )
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error,
            status=status,
            niters=0,
            **result_kwargs,
        )


class CPUBackwardEulerStep(CPUStep):
    """Backward Euler step solved with Newton iterations."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        *,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        self._be_state = np.zeros(self._state_size, dtype=self.precision)
        self._be_params = np.zeros(0, dtype=self.precision)
        self._be_drivers = np.zeros(0, dtype=self.precision)
        self._be_time = self.precision(0.0)
        self._be_dt = self.precision(0.0)
        self._be_increment = np.zeros(self._state_size, dtype=self.precision)

    def residual(self, candidate: Array) -> Array:
        base_state = self._be_state
        increment = candidate
        stage_state = base_state + increment
        observables = self.observables(
            stage_state,
            self._be_params,
            self._be_drivers,
            self._be_time,
        )
        derivative = self.rhs(
            stage_state,
            self._be_params,
            self._be_drivers,
            observables,
            self._be_time,
        )
        beta = self.precision(1.0)
        gamma = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        return beta * mass_term - gamma * self._be_dt * derivative

    def jacobian(self, candidate: Array) -> Array:
        stage_state = self._be_state + candidate
        _, jacobian = self.observables_and_jac(
            stage_state,
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
        initial_guess: Optional[Array] = None,
        time: float = 0.0,
    ) -> StepResultLike:
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
            guess = self._be_increment
        else:
            guess = self.ensure_array(initial_guess, copy=True)

        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=1)
        newton_kwargs = self._build_newton_logging_kwargs(
            stage_index=0,
            logging=logging,
        )
        increment, converged, niters = self.newton_solve(guess, **newton_kwargs)
        next_state = state_vector + increment

        observables = self.observables(
            next_state,
            params_array,
            drivers_next,
            next_time,
        )
        error = np.zeros_like(next_state, dtype=self.precision)
        status = self._status(converged, niters)
        self._be_increment = increment
        if logging:
            residual_vector = self.residual(increment)
            logging.residuals[0, :] = residual_vector
            logging.stage_states[0, :] = next_state
            logging.stage_derivatives[0, :] = self.rhs(
                next_state,
                params_array,
                drivers_next,
                observables,
                next_time,
            )
            logging.stage_observables[0, :] = observables
            logging.solver_initial_guesses[0, :] = self.ensure_array(guess)
            logging.solver_solutions[0, :] = increment
            logging.solver_iterations[0] = niters
            logging.solver_status[0] = int(
                IntegratorReturnCodes.SUCCESS
                if converged
                else IntegratorReturnCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
            )
            logging.stage_drivers[0, :] = drivers_next
            logging.stage_increments[0, :] = increment
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = (
            logging.stage_derivatives if logging else None
        )
        return self._make_result(
            state=next_state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
            **result_kwargs,
        )


class CPUCrankNicolsonStep(CPUStep):
    """Crank--Nicolson step with backward Euler error estimate."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        backward_step: Optional[CPUBackwardEulerStep] = None,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        self._cn_previous_state = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._cn_previous_derivative = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._cn_params = np.zeros(0, dtype=self.precision)
        self._cn_drivers_next = np.zeros(0, dtype=self.precision)
        self._cn_next_time = self.precision(0.0)
        self._cn_dt = self.precision(0.0)
        self._cn_stage_coefficient = self.precision(0.0)
        self._cn_base_state = np.zeros(self._state_size, dtype=self.precision)
        self._cn_increment = np.zeros(self._state_size, dtype=self.precision)
        if backward_step is None:
            backward_step = CPUBackwardEulerStep(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                instrument=instrument,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks,
            )
            self._backward = backward_step

    def residual(self, candidate: Array) -> Array:
        increment = candidate
        base_increment = self._cn_dt * self._cn_stage_coefficient
        stage_state = (
            self._cn_base_state
            + increment
            - base_increment * self._cn_previous_derivative
        )
        observables = self.observables(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        derivative = self.rhs(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            observables,
            self._cn_next_time,
        )
        beta = self.precision(1.0)
        gamma = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        trapezoid_rhs = self._cn_previous_derivative + derivative
        scale = self._cn_dt * self._cn_stage_coefficient
        return beta * mass_term - gamma * scale * trapezoid_rhs

    def jacobian(self, candidate: Array) -> Array:
        base_increment = self._cn_dt * self._cn_stage_coefficient
        stage_state = (
            self._cn_base_state
            + candidate
            - base_increment * self._cn_previous_derivative
        )
        _, jacobian = self.observables_and_jac(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        scale = self._cn_dt * self._cn_stage_coefficient
        return self._identity - scale * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ) -> StepResultLike:
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
        self._cn_dt = dt_value
        self._cn_stage_coefficient = self.precision(0.5)
        base_increment = self._cn_dt * self._cn_stage_coefficient
        self._cn_base_state = state_vector + base_increment * derivative_now

        guess = self._cn_increment
        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=1)
        newton_kwargs = self._build_newton_logging_kwargs(
            stage_index=0,
            logging=logging,
        )

        increment, converged, niters = self.newton_solve(guess, **newton_kwargs)
        next_state = self._cn_previous_state + increment
        final_increment = next_state - self._cn_previous_state

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
            initial_guess=final_increment,
            time=current_time,
        )
        error = next_state - backward_result.state
        status = self._status(converged, niters)
        self._cn_increment = final_increment
        if logging:
            residual_vector = self.residual(increment)
            logging.residuals[0, :] = residual_vector
            logging.stage_states[0, :] = next_state
            logging.stage_derivatives[0, :] = self.rhs(
                next_state,
                params_array,
                drivers_next,
                observables,
                next_time,
            )
            logging.stage_observables[0, :] = observables
            logging.solver_initial_guesses[0, :] = np.asarray(
                guess,
                dtype=self.precision,
            )
            logging.solver_solutions[0, :] = final_increment
            logging.solver_iterations[0] = niters
            logging.solver_status[0] = int(
                IntegratorReturnCodes.SUCCESS
                if converged
                else IntegratorReturnCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
            )
            logging.stage_drivers[0, :] = drivers_next
            logging.stage_increments[0, :] = final_increment
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = (
            logging.stage_derivatives if logging else None
        )
        return self._make_result(
            state=next_state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
            **result_kwargs,
        )


class CPUERKStep(CPUStep):
    """Explicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        tableau: Optional[ERKTableau] = None,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        resolved = DEFAULT_ERK_TABLEAU if tableau is None else tableau
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            tableau=resolved,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        self._erk_cached_slope = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._erk_has_cached_slope = False

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights


        state_dim = state_vector.shape[0]
        stage_derivatives = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=stage_count)
            logging.solver_status.fill(
                int(IntegratorReturnCodes.SUCCESS)
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
            if logging:
                logging.stage_drivers[stage_index, :] = drivers_stage
                logging.stage_states[stage_index, :] = stage_state
            if (
                stage_index == 0
                and self.tableau.first_same_as_last
                and self._erk_has_cached_slope
            ):
                stage_derivatives[stage_index, :] = self._erk_cached_slope
                if logging:
                    logging.stage_observables[stage_index, :] = (
                        self.observables(
                            stage_state,
                            params_array,
                            drivers_stage,
                            stage_time,
                        )
                    )
                    logging.solver_initial_guesses[stage_index, :] = stage_state
                    logging.solver_solutions[stage_index, :] = stage_state
                continue
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
            if logging:
                logging.stage_observables[stage_index, :] = observables_stage
                logging.solver_initial_guesses[stage_index, :] = stage_state
                logging.solver_solutions[stage_index, :] = stage_state
                logging.stage_increments[stage_index, :] = dt_value * derivative

        state_update = np.zeros_like(state_vector)
        for stage_index in range(stage_count):
            state_update = state_update + (
                b_weights[stage_index] * stage_derivatives[stage_index]
            )
        new_state = state_vector + dt_value * state_update

        error = np.zeros_like(state_vector)
        if error_weights is not None:
            error_update = np.zeros_like(state_vector)
            for stage_index in range(stage_count):
                error_update = error_update + (
                    error_weights[stage_index]
                    * stage_derivatives[stage_index]
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
        if self.tableau.first_same_as_last:
            self._erk_cached_slope = stage_derivatives[-1, :].copy()
            self._erk_has_cached_slope = True
        else:
            self._erk_has_cached_slope = False
        status = self._status(True, 0)
        stage_derivative_output = stage_derivatives if logging else None
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = stage_derivative_output
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error,
            status=status,
            niters=0,
            **result_kwargs,
        )


class CPUDIRKStep(CPUStep):
    """Diagonally implicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        tableau: Optional[DIRKTableau] = None,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        resolved = DEFAULT_DIRK_TABLEAU if tableau is None else tableau
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            tableau=resolved,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        self._dirk_reference = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._dirk_params = np.zeros(0, dtype=self.precision)
        self._dirk_drivers = np.zeros(0, dtype=self.precision)
        self._dirk_time = self.precision(0.0)
        self._dirk_dt_coeff = self.precision(0.0)
        self._dirk_slope = np.zeros(self._state_size, dtype=self.precision)
        self._dirk_increment = np.zeros(self._state_size, dtype=self.precision)
        self._dirk_has_slope = False
        self._dirk_has_increment = False

    def residual(self, candidate: Array) -> Array:
        base_state = self._dirk_reference
        increment = candidate
        stage_state = base_state + increment
        observables = self.observables(
            stage_state,
            self._dirk_params,
            self._dirk_drivers,
            self._dirk_time,
        )
        derivative = self.rhs(
            stage_state,
            self._dirk_params,
            self._dirk_drivers,
            observables,
            self._dirk_time,
        )
        beta = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        return beta * mass_term - self._dirk_dt_coeff * derivative

    def jacobian(self, candidate: Array) -> Array:
        stage_state = self._dirk_reference + candidate
        _, jacobian = self.observables_and_jac(
            stage_state,
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
        time: float = 0.0,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights

        state_dim = state_vector.shape[0]
        stage_derivatives = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=stage_count)
            logging.solver_status.fill(
                int(IntegratorReturnCodes.SUCCESS)
            )

        all_converged = True
        total_iters = 0

        drivers_now = self.drivers(current_time)
        observables_now = None
        derivative_now = None

        for stage_index in range(stage_count):
            if stage_index == 0:
                if (
                    self.tableau.first_same_as_last
                    and self._dirk_has_slope
                ):
                    stage_derivatives[stage_index, :] = self._dirk_slope
                    if logging:
                        logging.stage_states[stage_index, :] = state_vector
                        logging.stage_observables[stage_index, :] = (
                            self.observables(
                                state_vector,
                                params_array,
                                drivers_now,
                                current_time,
                            )
                        )
                        logging.stage_drivers[stage_index, :] = drivers_now
                        logging.solver_initial_guesses[stage_index, :] = (
                            state_vector
                        )
                        logging.solver_solutions[stage_index, :] = state_vector
                    continue
                if self.tableau.can_reuse_accepted_start:
                    if observables_now is None:
                        observables_now = self.observables(
                            state_vector,
                            params_array,
                            drivers_now,
                            current_time,
                        )
                    if derivative_now is None:
                        derivative_now = self.rhs(
                            state_vector,
                            params_array,
                            drivers_now,
                            observables_now,
                            current_time,
                        )
                    stage_derivatives[stage_index, :] = derivative_now
                    if logging:
                        logging.stage_states[stage_index, :] = state_vector
                        logging.stage_observables[stage_index, :] = (
                            observables_now
                        )
                        logging.stage_drivers[stage_index, :] = drivers_now
                        logging.solver_initial_guesses[stage_index, :] = (
                            state_vector
                        )
                        logging.solver_solutions[stage_index, :] = state_vector
                    continue

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
                if logging:
                    logging.stage_states[stage_index, :] = stage_state
                    logging.stage_observables[stage_index, :] = (
                        observables_stage
                    )
                    logging.stage_drivers[stage_index, :] = drivers_stage
                    logging.solver_initial_guesses[stage_index, :] = stage_state
                    logging.solver_solutions[stage_index, :] = stage_state
                if stage_index == stage_count - 1:
                    self._dirk_has_increment = False
                continue

            base_state = stage_state.copy()
            self._dirk_reference = base_state
            self._dirk_params = params_array
            self._dirk_drivers = drivers_stage
            self._dirk_time = stage_time
            self._dirk_dt_coeff = dt_value * diag_coeff

            guess = np.zeros_like(base_state)
            if (
                stage_index == 0
                and not self.tableau.first_same_as_last
                and not self.tableau.can_reuse_accepted_start
                and self._dirk_has_increment
            ):
                guess = self._dirk_increment
            newton_kwargs = self._build_newton_logging_kwargs(
                stage_index=stage_index,
                logging=logging,
            )
            increment, converged, niters = self.newton_solve(
                guess,
                **newton_kwargs,
            )
            solved_state = base_state + increment
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
            if logging:
                logging.stage_states[stage_index, :] = solved_state
                logging.residuals[stage_index, :] = self.residual(increment)
                logging.stage_increments[stage_index, :] = increment
                logging.stage_observables[stage_index, :] = observables_stage
                logging.stage_drivers[stage_index, :] = drivers_stage
                logging.solver_initial_guesses[stage_index, :] = guess
                logging.solver_solutions[stage_index, :] = increment
                logging.solver_iterations[stage_index] = niters
                logging.solver_status[stage_index] = int(
                    IntegratorReturnCodes.SUCCESS
                    if converged
                    else IntegratorReturnCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
                )
            if (
                stage_index == stage_count - 1
                and not self.tableau.can_reuse_accepted_start
            ):
                self._dirk_increment = increment
                self._dirk_has_increment = True

        state_accum = np.zeros_like(state_vector)
        error_accum = np.zeros_like(state_vector)
        for stage_index in range(stage_count):
            state_accum = state_accum + (
                b_weights[stage_index] * stage_derivatives[stage_index]
            )
            if error_weights is not None:
                error_accum = error_accum + (
                    error_weights[stage_index]
                    * stage_derivatives[stage_index]
                ) * dt_value

        new_state = state_vector + dt_value * state_accum
        end_time = current_time + dt_value
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )
        if self.tableau.first_same_as_last:
            self._dirk_slope = stage_derivatives[-1, :].copy()
            self._dirk_has_slope = True
        else:
            self._dirk_has_slope = False
        if self.tableau.can_reuse_accepted_start:
            self._dirk_has_increment = False
        status = self._status(all_converged, total_iters)
        stage_derivative_output = stage_derivatives if logging else None
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = stage_derivative_output
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error_accum,
            status=status,
            niters=total_iters,
            **result_kwargs,
        )


class CPURosenbrockWStep(CPUStep):
    """Rosenbrock--W step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        tableau: Optional[RosenbrockTableau] = None,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        resolved = (
            DEFAULT_ROSENBROCK_TABLEAU if tableau is None else tableau
        )
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            tableau=resolved,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        self._rosenbrock_cached_slope = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._rosenbrock_has_cached_slope = False
    @property
    def C_matrix(self) -> Optional[Array]:
        """Return the Jacobian update coefficients."""

        tableau = self.tableau
        if tableau is None or not hasattr(tableau, "C"):
            return None
        C_values = getattr(tableau, "C")
        if C_values is None:
            return None
        typed_rows = tableau.typed_rows(C_values, self.precision)
        return np.asarray(typed_rows, dtype=self.precision)

    @property
    def gamma(self) -> Optional[np.floating[Any]]:
        """Return the stage Jacobian shift."""

        tableau = self.tableau
        if tableau is None or not hasattr(tableau, "gamma"):
            return None
        return self.precision(getattr(tableau, "gamma"))

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        end_time = current_time + dt_value

        zero = self.precision(0.0)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights
        C_matrix = self.C_matrix
        gamma = self.gamma

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        if self._rosenbrock_has_cached_slope:
            derivative_now = self._rosenbrock_cached_slope.copy()
        else:
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
        state_dim = state_vector.shape[0]
        state_shifts = np.zeros(
            (stage_count, len(state_vector)),
            dtype=self.precision,
        )
        jacobian_shifts = np.zeros_like(state_shifts)
        rhs = np.zeros_like(state_vector)
        logging = None
        if self.instrument:
            logging = self._create_logging_buffers(stage_count=stage_count)
            logging.solver_status.fill(
                int(IntegratorReturnCodes.SUCCESS)
            )

        drivers_stage = drivers_now
        last_stage_derivative = derivative_now
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
                if logging:
                    logging.stage_observables[stage_index, :] = (
                        observables_stage
                    )
            last_stage_derivative = derivative_stage
            if logging:
                logging.stage_states[stage_index, :] = stage_state
                logging.stage_derivatives[stage_index, :] = derivative_stage
                logging.stage_drivers[stage_index, :] = drivers_stage
                if stage_index == 0:
                    logging.stage_observables[stage_index, :] = observables_now

            rhs[:] = dt_value * derivative_stage
            if stage_index > 0 and np.any(
                C_matrix[stage_index, :stage_index]
            ):
                jac_term = jacobian_now @ jacobian_shifts[stage_index]
                rhs[:] = rhs + dt_value * jac_term
            if logging:
                logging.residuals[stage_index, :] = rhs

            linear_kwargs = {}
            if logging:
                linear_kwargs = {
                    "stage_index": stage_index,
                    "instrumented": True,
                    "logging_initial_guess": logging.linear_initial_guesses,
                    "logging_iteration_guesses": (
                        logging.linear_iteration_guesses
                    ),
                    "logging_residuals": logging.linear_residuals,
                    "logging_squared_norms": logging.linear_squared_norms,
                    "logging_preconditioned_vectors": (
                        logging.linear_preconditioned_vectors
                    ),
                }
            stage_increment, linear_converged, linear_iters = self.linear_solve(
                lhs,
                rhs,
                initial_guess=None,
                **linear_kwargs,
            )
            state_accum = state_accum + (
                b_weights[stage_index] * stage_increment
            )
            if error_weights is not None:
                error_accum = error_accum + (
                    error_weights[stage_index] * stage_increment
                )
            if logging:
                logging.jacobian_updates[stage_index, :] = stage_increment
                logging.stage_increments[stage_index, :] = stage_increment
                logging.solver_solutions[stage_index, :] = stage_increment
                logging.solver_iterations[stage_index] = linear_iters
                logging.solver_status[stage_index] = int(
                    IntegratorReturnCodes.SUCCESS
                    if linear_converged
                    else IntegratorReturnCodes.MAX_LINEAR_ITERATIONS_EXCEEDED
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
        self._rosenbrock_cached_slope = last_stage_derivative.copy()
        self._rosenbrock_has_cached_slope = True
        status = self._status(True, 0)
        stage_derivative_output = (
            logging.stage_derivatives if logging else None
        )
        result_kwargs = self._logging_result_kwargs(logging)
        result_kwargs["stage_derivatives"] = stage_derivative_output
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error_accum,
            status=status,
            niters=0,
            **result_kwargs,
        )


class CPUBackwardEulerPCStep(CPUStep):
    """Backward Euler predictor--corrector step."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        corrector: Optional[CPUBackwardEulerStep] = None,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )
        if corrector is None:
            corrector = CPUBackwardEulerStep(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                instrument=instrument,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks,
            )
        self._corrector = corrector

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
    ) -> StepResultLike:
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
        predictor = dt_value * derivative_now
        return self._corrector(
            state=state_vector,
            params=params_array,
            dt=dt_value,
            initial_guess=predictor,
            time=current_time,
        )

def explicit_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    *,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single explicit Euler step."""

    stepper = CPUExplicitEulerStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def backward_euler_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single backward Euler step."""

    stepper = CPUBackwardEulerStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def crank_nicolson_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single Crank--Nicolson step."""

    stepper = CPUCrankNicolsonStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def erk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single explicit Runge--Kutta step."""

    stepper = CPUERKStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def dirk_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single DIRK step."""

    stepper = CPUDIRKStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def rosenbrock_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a single Rosenbrock--W step."""

    stepper = CPURosenbrockWStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
    return stepper(**kwargs)


def backward_euler_predict_correct_step(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    instrument: bool = False,
    **kwargs: object,
) -> StepResultLike:
    """Execute a backward Euler predictor--corrector step."""

    stepper = CPUBackwardEulerPCStep(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
    )
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

_STEP_CONSTRUCTOR_TO_FUNCTION = {
    ExplicitEulerStep: explicit_euler_step,
    BackwardsEulerStep: backward_euler_step,
    BackwardsEulerPCStep: backward_euler_predict_correct_step,
    CrankNicolsonStep: crank_nicolson_step,
    ERKStep: erk_step,
    DIRKStep: dirk_step,
    GenericRosenbrockWStep: rosenbrock_step,
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
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        instrument: bool = False,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
    ) -> Callable:
        if tableau_value is None:
            return step_class(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                instrument=instrument,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks,
            )
        return step_class(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            instrument=instrument,
            tableau=tableau_value,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
        )

    return factory


def get_ref_stepper(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    algorithm: str,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    tableau: Optional[Union[str, ButcherTableau]] = None,
    instrument: bool = False,
    newton_damping: float = 0.5,
    newton_max_backtracks: int = 8,
) -> StepCallable:
    """Return a configured CPU reference stepper for ``algorithm``."""

    factory = get_ref_step_factory(algorithm, tableau=tableau)
    return factory(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        instrument=instrument,
        newton_damping=newton_damping,
        newton_max_backtracks=newton_max_backtracks,
    )
