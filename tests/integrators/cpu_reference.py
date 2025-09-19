"""Reference CPU implementations for integrator algorithms and loops.

This module provides small, pure-Python utilities that mirror the behaviour
of the CUDA loop implementations closely enough for testing.  The helpers are
designed around the fixtures defined in :mod:`tests.conftest` so they can be
used from multiple test modules without re-implementing integration logic.

The two main entry points are :func:`run_reference_stepper`, which executes a
single explicit or implicit step, and :func:`run_reference_loop`, which mimics
the outer GPU loop including save/summarise cadence and optional adaptive
stepping.  Both return NumPy arrays that can be compared directly with device
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.jacobian import generate_jacobian
from cubie.odesystems.symbolic.sym_utils import topological_sort

from tests._utils import calculate_expected_summaries


Array = NDArray[np.floating]


def _ensure_array(vector: Sequence[float] | Array, dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.asarray(vector, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


class SystemEvaluator:
    """Light-weight wrapper exposing Python-side evaluations for a system."""

    def __init__(self, system: BaseODE) -> None:
        self.system = system
        self.precision = system.precision
        self.n_states = system.sizes.states
        self.n_observables = system.sizes.observables
        self._state_template = np.zeros(self.n_states, dtype=self.precision)
        self._observable_template = np.zeros(
            self.n_observables, dtype=self.precision
        )

    # ------------------------------------------------------------------
    # API expected by the steppers
    # ------------------------------------------------------------------
    def rhs(self, state: Array, params: Array, drivers: Array) -> tuple[Array, Array]:
        """Return state derivatives and observables at ``state``."""

        raise NotImplementedError

    def jacobian(self, state: Array, params: Array, drivers: Array) -> Array:
        """Return the Jacobian of the state derivative."""

        raise NotImplementedError


class PythonSystemEvaluator(SystemEvaluator):
    """Evaluator that relies on :meth:`BaseODE.correct_answer_python`."""

    def __init__(self, system: BaseODE) -> None:
        super().__init__(system)
        machine_eps = np.finfo(self.precision).eps
        self._finite_difference_epsilon = float(
            np.cbrt(machine_eps).astype(self.precision)
        )

    def rhs(self, state: Array, params: Array, drivers: Array) -> tuple[Array, Array]:
        derivatives, observables = self.system.correct_answer_python(
            state, params, drivers
        )
        return (
            np.asarray(derivatives, dtype=self.precision),
            np.asarray(observables, dtype=self.precision),
        )

    def jacobian(self, state: Array, params: Array, drivers: Array) -> Array:
        base_dxdt, _ = self.rhs(state, params, drivers)
        n = base_dxdt.size
        jac = np.zeros((n, n), dtype=self.precision)
        eps = self._finite_difference_epsilon
        for col in range(n):
            perturbed = state.astype(self.precision, copy=True)
            perturbed[col] += eps
            dxdt, _ = self.rhs(perturbed, params, drivers)
            jac[:, col] = (dxdt - base_dxdt) / eps
        return jac


class SymbolicSystemEvaluator(SystemEvaluator):
    """Evaluator for symbolic systems using SymPy expressions directly."""

    def __init__(self, system: BaseODE) -> None:
        super().__init__(system)
        indexed = system.indices
        self._state_index: Mapping[sp.Symbol, int] = indexed.states.index_map
        self._parameter_index: Mapping[sp.Symbol, int] = indexed.parameters.index_map
        self._constant_index: Mapping[sp.Symbol, int] = indexed.constants.index_map
        self._driver_index: Mapping[sp.Symbol, int] = indexed.drivers.index_map
        self._observable_index: Mapping[sp.Symbol, int] = (
            indexed.observables.index_map
        )
        self._dx_index: Mapping[sp.Symbol, int] = indexed.dxdt.index_map
        ordered_equations = topological_sort(system.equations)
        self._equations: list[tuple[sp.Symbol, sp.Expr]] = ordered_equations
        self._jacobian_expr = generate_jacobian(
            ordered_equations,
            self._state_index,
            self._dx_index,
        )

    def _subs_dict(
        self, state: Array, params: Array, drivers: Array
    ) -> Dict[sp.Symbol, float]:
        values: Dict[sp.Symbol, float] = {}
        for symbol, index in self._state_index.items():
            values[symbol] = float(state[index])
        for symbol, index in self._parameter_index.items():
            values[symbol] = float(params[index])
        for symbol, index in self._constant_index.items():
            values[symbol] = float(self.system.constants.values_array[index])
        for symbol, index in self._driver_index.items():
            values[symbol] = float(drivers[index])
        return values

    def rhs(self, state: Array, params: Array, drivers: Array) -> tuple[Array, Array]:
        subs = self._subs_dict(state, params, drivers)
        dxdt = self._state_template.copy()
        observables = self._observable_template.copy()
        for lhs, rhs in self._equations:
            value = float(rhs.evalf(subs=subs))
            subs[lhs] = value
            if lhs in self._dx_index:
                dxdt[self._dx_index[lhs]] = value
            elif lhs in self._observable_index:
                observables[self._observable_index[lhs]] = value
            else:  # auxiliary expression
                continue
        return dxdt, observables

    def jacobian(self, state: Array, params: Array, drivers: Array) -> Array:
        subs = self._subs_dict(state, params, drivers)
        jacobian = np.array(self._jacobian_expr.evalf(subs=subs), dtype=self.precision)
        return jacobian


def build_system_evaluator(system: BaseODE) -> SystemEvaluator:
    """Return a system evaluator suitable for ``system``."""

    if "correct_answer_python" in system.__class__.__dict__:
        return PythonSystemEvaluator(system)
    return SymbolicSystemEvaluator(system)


@dataclass
class StepResult:
    """Container describing the outcome of a single integration step."""

    state: Array
    observables: Array
    error: Array
    converged: bool = True


def explicit_euler_step(
    evaluator: SystemEvaluator,
    state: Array,
    params: Array,
    drivers: Array,
    dt: float,
) -> StepResult:
    """Explicit Euler integration step."""

    dxdt, observables = evaluator.rhs(state, params, drivers)
    new_state = state + dt * dxdt
    error = np.zeros_like(state)
    return StepResult(new_state, observables, error, True)


def _newton_solve(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    initial_guess: Array,
    precision: np.dtype,
    tol: float = 1e-10,
    max_iters: int = 25,
) -> tuple[Array, bool]:
    """Solve ``residual(x) = 0`` using a dense Newton iteration."""

    state = initial_guess.astype(precision, copy=True)
    for _ in range(max_iters):
        res = residual(state)
        res_norm = np.linalg.norm(res, ord=np.inf)
        if res_norm < tol:
            return state, True
        jac = jacobian(state)
        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac, -res, rcond=None)[0]
        state = state + delta.astype(precision)
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return state, True
    return state, False


def backward_euler_step(
    evaluator: SystemEvaluator,
    state: Array,
    params: Array,
    drivers: Array,
    dt: float,
    tol: float = 1e-10,
    initial_guess: Array | None = None,
) -> StepResult:
    """Backward Euler step solved via Newton iteration."""

    precision = evaluator.precision

    def residual(candidate: Array) -> Array:
        dxdt, _ = evaluator.rhs(candidate, params, drivers)
        return candidate - state - dt * dxdt

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers)
        identity = np.eye(jac.shape[0], dtype=precision)
        return identity - dt * jac

    if initial_guess is None:
        guess = state.astype(precision, copy=True)
    else:
        guess = np.asarray(initial_guess, dtype=precision).astype(
            precision, copy=True
        )
    next_state, converged = _newton_solve(residual, jacobian, guess, precision, tol)
    dxdt, observables = evaluator.rhs(next_state, params, drivers)
    error = np.zeros_like(next_state)
    return StepResult(next_state, observables, error, converged)


def crank_nicolson_step(
    evaluator: SystemEvaluator,
    state: Array,
    params: Array,
    drivers_now: Array,
    drivers_next: Array,
    dt: float,
    tol: float = 1e-10,
) -> StepResult:
    """Crank–Nicolson step with embedded backward Euler for error estimation."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now)

    def residual(candidate: Array) -> Array:
        f_candidate, _ = evaluator.rhs(candidate, params, drivers_next)
        return candidate - state - 0.5 * dt * (f_now + f_candidate)

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers_next)
        identity = np.eye(jac.shape[0], dtype=precision)
        return identity - 0.5 * dt * jac

    guess = state.astype(precision, copy=True)
    next_state, converged = _newton_solve(residual, jacobian, guess, precision, tol)
    _, observables = evaluator.rhs(next_state, params, drivers_next)

    # Embedded backward Euler step for the error estimate
    be_result = backward_euler_step(
        evaluator, state, params, drivers_next, dt, tol
    )
    error = next_state - be_result.state
    return StepResult(next_state, observables, error, converged)


def backward_euler_predict_correct_step(
    evaluator: SystemEvaluator,
    state: Array,
    params: Array,
    drivers_now: Array,
    drivers_next: Array,
    dt: float,
    tol: float = 1e-10,
) -> StepResult:
    """Predict with explicit Euler and correct with backward Euler."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now)
    predictor = state.astype(precision, copy=True) + dt * f_now
    return backward_euler_step(
        evaluator,
        state,
        params,
        drivers_next,
        dt,
        tol=tol,
        initial_guess=predictor,
    )


class DriverSampler:
    """Utility translating forcing vectors to values at arbitrary times."""

    def __init__(self, drivers: Array, base_dt: float, precision: np.dtype) -> None:
        if drivers.size == 0:
            self._drivers = np.zeros((0, 1), dtype=precision)
        else:
            if drivers.ndim != 2:
                raise ValueError("forcing_vectors must have shape (n_drivers, n_samples)")
            self._drivers = drivers.astype(precision)
        self._base_dt = float(base_dt)
        self._samples = self._drivers.shape[1]

    def sample(self, time: float) -> Array:
        if self._drivers.size == 0:
            return np.zeros(0, dtype=self._drivers.dtype)
        index = int(np.floor(time / self._base_dt)) % self._samples
        return self._drivers[:, index]


class AdaptiveController:
    """Simple adaptive step controller mirroring GPU heuristics."""

    def __init__(
        self,
        *,
        kind: str,
        dt_min: float,
        dt_max: float,
        atol: float,
        rtol: float,
        order: int,
        precision: np.dtype,
    ) -> None:
        self.kind = kind
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt = float(dt_min)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.order = max(int(order), 1)
        self.precision = precision
        self.safety = 0.9
        self.min_gain = 0.2
        self.max_gain = 5.0
        self._history: list[float] = []

    @property
    def is_adaptive(self) -> bool:
        return self.kind != "fixed"

    def error_norm(self, state_prev: Array, state_new: Array, error: Array) -> float:
        scale = self.atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.sqrt(np.mean((error / scale) ** 2))
        if not np.isfinite(norm):
            norm = np.inf
        return float(norm)

    def propose_dt(self, error_estimate: float, accept: bool) -> None:
        if not self.is_adaptive:
            return
        gain = self._gain(error_estimate)
        if accept:
            self._history.append(max(error_estimate, 1e-16))
            if len(self._history) > 3:
                self._history.pop(0)
            self.dt = min(self.dt_max, max(self.dt_min, self.dt * gain))
        else:
            self.dt = max(self.dt_min, self.dt * min(gain, 1.0))

    def _gain(self, error_estimate: float) -> float:
        if error_estimate <= 0.0:
            return self.max_gain
        if self.kind == "I":
            exponent = 1.0 / (self.order + 1.0)
            gain = self.safety * error_estimate ** (-exponent)
        elif self.kind == "PI":
            exponent = 0.7 / (self.order + 1.0)
            prev = self._history[-1] if self._history else error_estimate
            gain = self.safety * error_estimate ** (-exponent) * prev ** (
                0.4 / (self.order + 1.0)
            )
        elif self.kind == "PID":
            prev = self._history[-1] if self._history else error_estimate
            prev_prev = self._history[-2] if len(self._history) > 1 else prev
            k1 = 0.7 / (self.order + 1.0)
            k2 = 0.4 / (self.order + 1.0)
            k3 = 0.1 / (self.order + 1.0)
            gain = (
                self.safety
                * error_estimate ** (-k1)
                * prev ** k2
                * prev_prev ** (-k3)
            )
        elif self.kind == "gustafsson":
            prev = self._history[-1] if self._history else error_estimate
            ratio = prev / max(error_estimate, 1e-16)
            exponent = 1.0 / (self.order + 1.0)
            gain = self.safety * error_estimate ** (-exponent) * ratio ** 0.1
        else:
            gain = 1.0
        gain = min(self.max_gain, max(self.min_gain, gain))
        return gain


def run_reference_stepper(
    stepper: str,
    evaluator: SystemEvaluator,
    state: Array,
    params: Array,
    drivers_now: Array,
    drivers_next: Array,
    dt: float,
) -> StepResult:
    """Execute a single step using the requested stepper."""

    stepper_name = stepper.lower()
    if stepper_name == "explicit_euler":
        return explicit_euler_step(
            evaluator, state, params, drivers_now, dt
        )
    if stepper_name == "backward_euler":
        return backward_euler_step(
            evaluator, state, params, drivers_next, dt
        )
    if stepper_name == "backward_euler_predict_correct":
        return backward_euler_predict_correct_step(
            evaluator,
            state,
            params,
            drivers_now,
            drivers_next,
            dt,
        )
    if stepper_name == "crank_nicolson":
        return crank_nicolson_step(
            evaluator, state, params, drivers_now, drivers_next, dt
        )
    raise ValueError(f"Unknown stepper '{stepper}'.")


def _collect_saved_outputs(
    save_history: list[Array],
    indices: Sequence[int],
    width: int,
    dtype: np.dtype,
) -> Array:
    if width == 0:
        return np.zeros((len(save_history), 0), dtype=dtype)
    data = np.zeros((len(save_history), width), dtype=dtype)
    for row, snapshot in enumerate(save_history):
        data[row, :] = snapshot[indices]
    return data


def run_reference_loop(
    *,
    system: BaseODE,
    inputs: Mapping[str, Array],
    solver_settings: Mapping[str, float],
    loop_compile_settings: Mapping[str, Iterable[int] | float | Sequence[str]],
    output_functions,
    stepper: str,
    step_controller_settings: Mapping[str, float | str] | None = None,
) -> dict[str, Array]:
    """Execute a CPU loop mirroring :class:`IVPLoop` behaviour."""

    evaluator = build_system_evaluator(system)
    precision = system.precision
    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    forcing_vectors = inputs["forcing_vectors"].astype(precision, copy=False)
    dt_min = float(solver_settings["dt_min"])
    dt_max = float(solver_settings.get("dt_max", dt_min))
    duration = float(solver_settings["duration"])
    warmup = float(solver_settings.get("warmup", 0.0))
    dt_save = float(solver_settings["dt_save"])
    dt_summarise = float(solver_settings.get("dt_summarise", dt_save))
    controller_settings = step_controller_settings or {"kind": "fixed"}

    controller = AdaptiveController(
        kind=str(controller_settings.get("kind", "fixed")).lower(),
        dt_min=controller_settings.get("dt", dt_min),
        dt_max=controller_settings.get("dt_max", dt_max),
        atol=solver_settings.get("atol", 1e-6),
        rtol=solver_settings.get("rtol", 1e-6),
        order=controller_settings.get("order", 1),
        precision=precision,
    )
    controller.dt = controller_settings.get("dt", dt_min)

    sampler = DriverSampler(forcing_vectors, dt_min, precision)
    saved_state_indices = _ensure_array(
        loop_compile_settings.get("saved_state_indices", []), np.int32
    )
    saved_observable_indices = _ensure_array(
        loop_compile_settings.get("saved_observable_indices", []), np.int32
    )
    summarised_state_indices = _ensure_array(
        loop_compile_settings.get("summarised_state_indices", []), np.int32
    )
    summarised_observable_indices = _ensure_array(
        loop_compile_settings.get("summarised_observable_indices", []),
        np.int32,
    )
    save_time = "time" in loop_compile_settings.get("output_functions", [])
    max_save_samples = int(duration / dt_save) if dt_save > 0 else 0
    state_history: list[Array] = []
    observable_history: list[Array] = []
    time_history: list[float] = []
    state = initial_state.copy()
    state_history.append(initial_state.copy())
    observable_history.append(np.zeros(len(saved_observable_indices),
                                       dtype=precision))
    t = 0.0
    end_time = warmup + duration
    next_save_time = warmup + dt_save if dt_save > 0 else end_time + 1.0

    while t < end_time - 1e-12:
        dt = min(controller.dt, end_time - t)
        drivers_now = sampler.sample(t)
        drivers_next = sampler.sample(t + dt)
        result = run_reference_stepper(
            stepper,
            evaluator,
            state,
            params,
            drivers_now,
            drivers_next,
            dt,
        )
        error_norm = controller.error_norm(state, result.state, result.error)
        if controller.is_adaptive and error_norm > 1.0:
            controller.propose_dt(error_norm, accept=False)
            continue
        t += dt
        controller.propose_dt(error_norm, accept=True)
        state = result.state
        if t >= warmup and len(state_history) < max_save_samples:
            if t + 1e-12 >= next_save_time:
                state_history.append(state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(next_save_time - warmup)
                next_save_time += dt_save
    # ------------------------------------------------------------------
    # Convert histories to arrays matching device layout
    # ------------------------------------------------------------------
    state_output = _collect_saved_outputs(
        state_history,
        saved_state_indices,
        saved_state_indices.size,
        precision,
    )
    if save_time and state_output.size > 0:
        times = np.asarray(time_history, dtype=precision)[:, np.newaxis]
        state_output = np.hstack((state_output, times))
    observables_output = _collect_saved_outputs(
        observable_history,
        saved_observable_indices,
        saved_observable_indices.size,
        precision,
    )
    if state_output.shape[0] == 0:
        width = saved_state_indices.size + int(save_time)
        state_output = np.zeros((max_save_samples or 1, max(1, width)), dtype=precision)
    if observables_output.shape[0] == 0:
        width = saved_observable_indices.size
        observables_output = np.zeros(
            (max_save_samples or 1, max(1, width)),
            dtype=precision,
        )

    if state_history:
        summarise_every = (
            max(int(round(dt_summarise / dt_save)), 1) if dt_save > 0 else 1
        )
        state_summary_source = _collect_saved_outputs(
            state_history,
            summarised_state_indices,
            summarised_state_indices.size,
            precision,
        )
        observable_summary_source = _collect_saved_outputs(
            observable_history,
            summarised_observable_indices,
            summarised_observable_indices.size,
            precision,
        )
        if state_summary_source.size == 0:
            state_summary_source = np.zeros(
                (len(state_history), 1), dtype=precision
            )
        if observable_summary_source.size == 0:
            observable_summary_source = np.zeros(
                (len(observable_history), 1), dtype=precision
            )
        state_summary, observable_summary = calculate_expected_summaries(
            state_summary_source,
            observable_summary_source,
            summarise_every,
            output_functions.compile_settings.output_types,
            output_functions.summaries_output_height_per_var,
            precision,
        )
    else:
        state_summary = np.zeros((1, 1), dtype=precision)
        observable_summary = np.zeros((1, 1), dtype=precision)

    return {
        "state": state_output,
        "observables": observables_output,
        "state_summaries": state_summary,
        "observable_summaries": observable_summary,
    }

