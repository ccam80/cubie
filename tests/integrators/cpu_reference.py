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
from typing import Callable, Dict, Mapping, Sequence, Optional
import numpy as np
import sympy as sp
from numpy.typing import NDArray

from cubie import SymbolicODE, IntegratorReturnCodes
from cubie.odesystems.symbolic.jacobian import generate_jacobian
from cubie.odesystems.symbolic.sym_utils import topological_sort

from tests._utils import calculate_expected_summaries

TIME_SYMBOL = sp.Symbol("t", real=True)


Array = NDArray[np.floating]
STATUS_MASK = 0xFFFF


def _ensure_array(vector: Sequence[float] | Array, dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.atleast_1d(vector).astype(dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


class CPUODESystem():
    """Evaluator for symbolic systems using compiled numerical functions."""

    def __init__(self, system: SymbolicODE) -> None:
        self.system = system
        self.precision = system.precision
        self.n_states = system.sizes.states
        self.n_observables = system.sizes.observables
        self._state_template = np.zeros(self.n_states, dtype=self.precision)
        self._observable_template = np.zeros(
                self.n_observables, dtype=self.precision
        )

        indexed = system.indices
        self._state_index = indexed.states.index_map
        self._parameter_index = indexed.parameters.index_map
        self._constant_index = indexed.constants.index_map
        self._driver_index = indexed.drivers.index_map
        self._observable_index = indexed.observables.index_map

        self._dx_index = indexed.dxdt.index_map
        ordered_equations = topological_sort(system.equations)
        self._equations = ordered_equations
        self._jacobian_expr = generate_jacobian(
            ordered_equations,
            self._state_index,
            self._dx_index,
        )

        # Precompile expressions for fast numerical evaluation
        self._compile_expressions()

    def _compile_expressions(self) -> None:
        """Compile symbolic expressions into fast numerical functions using lambdify."""

        self._compiled_equations = {}
        self._equation_symbols = {}  # Track which symbols each equation needs

        for lhs, rhs in self._equations:
            free_vars = list(rhs.free_symbols)
            if free_vars:
                compiled_fn = sp.lambdify(free_vars, rhs, modules=['numpy'])
            else:
                compiled_fn = self.precision(rhs)
            self._equation_symbols[lhs] = free_vars
            self._compiled_equations[lhs] = compiled_fn


        # Compile jacobian if it exists
        if self._jacobian_expr.shape[0] > 0:
            jacobian_entries = []
            jacobian_symbols = []
            jac_rows, jac_cols = self._jacobian_expr.shape

            for i in range(jac_rows):
                row_entries = []
                row_symbols = []
                for j in range(jac_cols):
                    expr = self._jacobian_expr[i, j]
                    expr_syms = list(expr.free_symbols)
                    if expr_syms:
                        compiled_entry = sp.lambdify(expr_syms, expr, modules=['numpy'])
                    else:
                        compiled_entry = self.precision(expr)

                    row_entries.append(compiled_entry)
                    row_symbols.append(expr_syms)
                jacobian_entries.append(row_entries)
                jacobian_symbols.append(row_symbols)

            self._compiled_jacobian = jacobian_entries
            self._jacobian_symbols = jacobian_symbols
        else:
            self._compiled_jacobian = []
            self._jacobian_symbols = []

    def _get_symbol_values(
            self,
            state: Array,
            params: Array,
            drivers: Array,
            time_scalar: float,
            observables: Optional[Array] = None,

    ) -> dict:
        """Get current values for all symbols."""
        # drivers = _ensure_array(drivers, self.precision)
        precision = self.precision
        values = {}
        values.update(
            {
                **{
                    sym: precision(state[index])
                    for sym, index in self._state_index.items()
                },
                **{
                    sym: precision(params[index])
                    for sym, index in self._parameter_index.items()
                },
                **{
                    sym: precision(self.system.constants.values_dict[str(sym)])
                    for sym in self._constant_index.keys()
                },
                **{
                    sym: precision(drivers[index])
                    for sym, index in self._driver_index.items()
                }
            }
        )

        if observables is not None:
            values.update({
                    sym: precision(observables[index])
                    for sym, index in self._observable_index.items()
                },
        )

        # Provide the current simulation time symbol value
        values[TIME_SYMBOL] = self.precision(time_scalar)
        return values

    def rhs(self, state: Array, params: Array, drivers: Array,
            time_scalar: float,
            ) -> tuple[Array, Array]:
        dxdt = self._state_template.copy()
        observables = self._observable_template.copy()
        symbol_values = self._get_symbol_values(state, params, drivers,
                                                time_scalar)

        # Evaluate each compiled equation
        # Try er thrice for good measure - for out-of-order evaluation? no,
        # args will fail if there's unidentified symbols.
        for lhs, argsymbols in self._equation_symbols.items():
            if argsymbols:
                args = tuple(symbol_values[sym] for sym in argsymbols)
                value = self.precision(self._compiled_equations[lhs](*args))
            else:
                value = self._compiled_equations[lhs]

            symbol_values.update({lhs: value})

            if lhs in self._dx_index:
                dxdt[self._dx_index[lhs]] = value
            elif lhs in self._observable_index:
                observables[self._observable_index[lhs]] = value
            # auxiliary expressions are stored in computed_values for dependent equations

        return dxdt, observables

    def jacobian(
            self,
            state: Array,
            params: Array,
            drivers: Array,
            time_scalar: float,
        ) -> Array:
        if not self._compiled_jacobian:
            return np.zeros((self.n_states, self.n_states), dtype=self.precision)

        _, observables = self.rhs(state, params, drivers, time_scalar)

        symbol_values = self._get_symbol_values(state, params, drivers,
                                                time_scalar, observables)
        jac_rows = len(self._compiled_jacobian)
        jac_cols = len(self._compiled_jacobian[0]) if jac_rows > 0 else 0
        jacobian = np.zeros((jac_rows, jac_cols), dtype=self.precision)

        for i, (row, row_symbols) in enumerate(zip(self._compiled_jacobian, self._jacobian_symbols)):
            for j, (compiled_entry, expr_symbols) in enumerate(zip(row, row_symbols)):
                if expr_symbols:
                    # Refresh symbol values only if needed (observables already reflect state)
                    args = tuple(symbol_values[sym] for sym in expr_symbols)
                    jacobian[i, j] = self.precision(compiled_entry(*args))
                else:
                    jacobian[i, j] = compiled_entry
        return jacobian

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

def explicit_euler_step(
    evaluator: CPUODESystem,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    drivers_now: Optional[Array] = None,
    drivers_next: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: Optional[int] = None,
    time: float = 0.0,
) -> StepResult:
    """Explicit Euler integration step."""

    dxdt, _ = evaluator.rhs(state, params, drivers_now, time)
    new_state = state + dt * dxdt
    _, observables = evaluator.rhs(new_state, params, drivers_next, time+dt)
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
        res_norm = np.linalg.norm(res, ord=np.inf)
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
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    drivers_now: Optional[Array] = None,
    drivers_next: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    initial_guess: Optional[Array] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Backward Euler step solved via Newton iteration."""

    precision = evaluator.precision

    def residual(candidate: Array) -> Array:
        dxdt, _ = evaluator.rhs(candidate, params, drivers_next, time)
        return candidate - state - dt * dxdt

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers_next, time)
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
    _, observables = evaluator.rhs(next_state, params, drivers_next, time)
    error = np.zeros_like(next_state)
    status = _encode_solver_status(converged, niters)
    return StepResult(next_state, observables, error, status, niters)


def crank_nicolson_step(
    evaluator: CPUODESystem,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    drivers_now: Optional[Array] = None,
    drivers_next: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Crank–Nicolson step with embedded backward Euler for error estimation."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now, time)

    def residual(candidate: Array) -> Array:
        f_candidate, _ = evaluator.rhs(candidate, params, drivers_next, time)
        return candidate - state - precision(0.5) * dt * (f_now + f_candidate)

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers_next, time)
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
    _, observables = evaluator.rhs(next_state, params, drivers_next, time + dt)

    # Embedded backward Euler step for the error estimate
    be_result = backward_euler_step(
        evaluator=evaluator,
        state=state,
        params=params,
        drivers_next=drivers_next,
        dt=dt,
        tol=tol,
        initial_guess=next_state,
        max_iters=max_iters,
        time=time,
    )
    error = next_state - be_result.state
    status = _encode_solver_status(converged, niters)
    return StepResult(next_state, observables, error, status, niters)

def backward_euler_predict_correct_step(
    evaluator: CPUODESystem,
    state: Optional[Array] = None,
    params: Optional[Array] = None,
    drivers_now: Optional[Array] = None,
    drivers_next: Optional[Array] = None,
    dt: Optional[float] = None,
    tol: Optional[float] = None,
    max_iters: int = 25,
    time: float = 0.0,
) -> StepResult:
    """Predict with explicit Euler and correct with backward Euler."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now, time)
    predictor = state.astype(precision, copy=True) + dt * f_now
    return backward_euler_step(
        evaluator=evaluator,
        state=state,
        params=params,
        drivers_next=drivers_next,
        dt=dt,
        tol=tol,
        initial_guess=predictor,
        max_iters=max_iters,
        time=time,
    )


class DriverSampler:
    """Utility mapping loop save indices to driver samples."""

    def __init__(self, drivers: Array, base_dt: float, precision: np.dtype) -> None:
        if drivers.size == 0:
            self._drivers = np.zeros((0, 0), dtype=precision)
        else:
            if drivers.ndim != 2:
                raise ValueError("forcing_vectors must have shape (n_samples, n_drivers)")
            self._drivers = drivers.astype(precision, copy=True)
        self._samples = self._drivers.shape[0]

    def sample(self, save_index: int) -> Array:
        """Return the driver values associated with ``save_index``."""

        if self._drivers.size == 0:
            return np.zeros(0, dtype=self._drivers.dtype)

        index = int(save_index)
        if self._samples > 0:
            index %= self._samples
        return self._drivers[index, :]

class CPUAdaptiveController:
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
        precision: type,
        kp: float = 1/18,
        ki: float = 1/9,
        kd: float = 1/18,
        gamma: float = 0.9,
        safety: float = 0.9,
        min_gain: float = 0.2,
        max_gain: float = 2.0,
        max_newton_iters: int = 0,
    ) -> None:

        self.kind = kind.lower()
        self.dt_min = precision(dt_min)
        self.dt_max = precision(dt_max)
        if kind == "fixed":
            self.dt0 = precision(dt_min)
        else:
            self.dt0 = precision((dt_min + dt_max) / 2)
        self.dt = self.dt_min
        self.atol = precision(atol)
        self.rtol = precision(rtol)
        self.order = order
        self.precision = precision
        self.safety = precision(safety)
        self.min_gain = precision(min_gain)
        self.max_gain = precision(max_gain)
        self.kp = precision(kp)
        self.ki = precision(ki)
        self.kd = precision(kd)
        self.gamma = precision(gamma)
        self.max_newton_iters = int(max_newton_iters)
        zero = precision(0.0)
        self._history: list = [zero, zero]
        self._step_count = 0
        self._convergence_failed = False
        self._rejections_at_dt_min = 0
        self._prev_nrm2 = zero
        self._prev_prev_nrm2 = zero
        self._prev_dt = zero

    @property
    def is_adaptive(self) -> bool:
        return self.kind != "fixed"

    def error_norm(self, state_prev: Array, state_new: Array, error: Array) -> float:
        error = np.maximum(error, 1e-30)
        scale = self.atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new))
        nrm2 = np.sum((scale * scale) / (error * error))
        norm = nrm2 / len(error)
        return self.precision(norm)

    def propose_dt(self,
                   error_vector: Array,
                   prev_state: Array,
                   new_state: Array,
                   niters: int = 0) -> bool:
        self._step_count += 1
        if not self.is_adaptive:
            return True
        errornorm = self.error_norm(
            state_prev=prev_state,
            state_new=new_state,
            error=error_vector)

        accept = errornorm > 1.0

        # Save current dt before computing new gain (needed for gustafsson)
        current_dt = self.dt
        gain = self._gain(
            errornorm=errornorm,
            accept=accept,
            niters=niters,
            current_dt=current_dt)

        unclamped_dt = self.precision(current_dt * gain)
        new_dt = min(self.dt_max, max(self.dt_min, unclamped_dt))
        self.dt = new_dt
        self._prev_dt = current_dt
        if self.kind == "pid":
            self._prev_prev_nrm2 = self._prev_nrm2
        self._prev_nrm2 = errornorm

        if new_dt < self.dt_min:
            raise ValueError(f"dt < dt_min: {new_dt} < {self.dt_min}"
                             f"exceeded")

        return accept

    def _gain(self,
              errornorm: float,
              accept: bool,
              niters: int,
              current_dt: float) -> float:
        precision = self.precision
        expo_fraction = precision( precision(1.0) / (precision(2) * (precision(self.order) + precision(1))))
        kp_exp = precision(self.kp * expo_fraction)
        ki_exp = precision(self.ki * expo_fraction)
        kd_exp = precision(self.kd * expo_fraction)

        if self.kind == "i":
            exponent = expo_fraction
            gain = self.safety * precision(errornorm ** exponent)

        elif self.kind == "pi":

            prev = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            gain = (
                self.safety
                * precision(errornorm ** kp_exp)
                * precision(prev ** (-ki_exp))
            )

        elif self.kind == "pid":

            prev_nrm2 = (
                self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            )
            prev_prev = (
                self._prev_prev_nrm2
                if self._prev_prev_nrm2 > 0.0
                else prev_nrm2
            )
            gain = (
                self.safety
                * precision(errornorm ** kp_exp)
                * precision(prev_nrm2 ** (-ki_exp))
                * precision(prev_prev ** (-kd_exp))
            )

        elif self.kind == "gustafsson":
            one = precision(1.0)
            two = precision(2.0)
            M = self.max_newton_iters
            fac = min(self.gamma, ((one + two * M) * self.gamma) / (
                    niters + two * M))
            gain_basic = self.safety * fac * (errornorm ** expo_fraction)

            use_gus = (accept
                       and (self._prev_dt > 0.0)
                       and (self._prev_nrm2 > 0.0))
            if use_gus:
                ratio = (errornorm * errornorm) / self._prev_nrm2
                gain_gus = (
                    self.safety
                    * (current_dt / self._prev_dt)
                    * precision(ratio ** expo_fraction)
                    * self.gamma
                )
                gain = gain_gus if gain_gus < gain_basic else gain_basic
            else:
                gain = gain_basic
        else:
            gain = precision(1.0)

        gain = min(self.max_gain, max(self.min_gain, gain))
        return precision(gain)


def get_ref_step_fn(
         algorithm: str,
    ) -> Callable:

    if algorithm.lower() == "euler":
        return explicit_euler_step
    elif algorithm.lower() == "backwards_euler":
        return backward_euler_step
    elif algorithm.lower() == "backwards_euler_pc":
        return backward_euler_predict_correct_step
    elif algorithm.lower() == "crank_nicolson":
        return crank_nicolson_step
    else:
        raise ValueError(f"Unknown stepper algorithm: {algorithm}")


def _collect_saved_outputs(
    save_history: list[Array],
    indices: Sequence[int],
    dtype: np.dtype,
) -> Array:
    width = len(indices)
    if width == 0:
        return np.zeros((len(save_history), 0), dtype=dtype)
    data = np.zeros((len(save_history), width), dtype=dtype)
    for row, sample in enumerate(save_history):
        data[row, :] = sample[indices]
    return data


def run_reference_loop(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    solver_settings,
    implicit_step_settings,
    output_functions,
    controller,
    step_controller_settings: Optional[Dict] = None,
) -> dict[str, Array]:
    """Execute a CPU loop mirroring :class:`IVPLoop` behaviour."""

    precision = evaluator.precision

    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    forcing_vectors = inputs["drivers"].astype(precision, copy=True)
    duration = precision(solver_settings["duration"])
    warmup = precision(solver_settings["warmup"])
    dt_save = precision(solver_settings["dt_save"])
    dt_summarise = precision(solver_settings["dt_summarise"])
    # controller.dt = controller.dt0
    status_flags = 0
    zero = precision(0.0)

    step_fn = get_ref_step_fn(solver_settings["algorithm"])

    sampler = DriverSampler(forcing_vectors, dt_save, precision)


    saved_state_indices = _ensure_array(
        solver_settings["saved_state_indices"], np.int32
    )
    saved_observable_indices = _ensure_array(
        solver_settings["saved_observable_indices"], np.int32)
    summarised_state_indices = _ensure_array(
        solver_settings["summarised_state_indices"], np.int32
    )
    summarised_observable_indices = _ensure_array(
        solver_settings["summarised_observable_indices"], np.int32
    )

    save_time = output_functions.save_time
    max_save_samples = int(np.ceil(duration / dt_save))

    state = initial_state.copy()
    state_history = [state.copy()]
    observable_history = []
    t = precision(0.0)

    if warmup > zero:
        next_save_time = warmup
        save_index = 0
    else:
        next_save_time = dt_save
        save_index = 1
        state_history = [state.copy()]
        _, observables = evaluator.rhs(state, params, sampler.sample(0), t)
        observable_history.append(observables.copy())
        time_history: list[float] = [t]

    end_time = precision(warmup + duration)
    max_iters = implicit_step_settings['max_newton_iters']
    fixed_steps_per_save = int(np.ceil(dt_save / controller.dt_min))
    fixed_step_count = 0

    while t < end_time - precision(1e-12):
        dt = precision(min(controller.dt, end_time - t))
        do_save=False
        if controller.is_adaptive:
            if t + dt + precision(1e-10) >= next_save_time:
                dt = precision(next_save_time - t)
                do_save = True
        else:
            if (fixed_step_count+1) % fixed_steps_per_save == 0:
                do_save = True
                fixed_step_count = 0
            else:
                fixed_step_count += 1


        driver_sample = sampler.sample(save_index)

        #Drivers handling matches devices (poor) approach
        drivers_now = driver_sample
        drivers_next = driver_sample

        result = step_fn(
                evaluator,
                state=state,
                params=params,
                drivers_now=drivers_now,
                drivers_next=drivers_next,
                dt=dt,
                tol=implicit_step_settings['nonlinear_tolerance'],
                max_iters=max_iters,
                time=float(t),
            )

        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        accept = controller.propose_dt(
            error_vector=result.error,
            prev_state=state,
            new_state=result.state,
            niters=result.niters)
        if not accept:
            continue

        state = result.state.copy()
        t += precision(dt)

        if do_save:
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(precision(next_save_time - warmup))
            next_save_time += dt_save
            save_index += 1


    state_output = _collect_saved_outputs(
        state_history,
        saved_state_indices,
        precision,
    )

    observables_output = _collect_saved_outputs(
        observable_history,
        saved_observable_indices,
        precision,
    )
    if save_time:
        state_output = np.column_stack((state_output, np.asarray(time_history)))

    summarise_every = int(np.ceil(dt_summarise / dt_save))

    state_summary, observable_summary = calculate_expected_summaries(
            state_output,
            observables_output,
            summarised_state_indices,
            summarised_observable_indices,
            summarise_every,
            output_functions.compile_settings.output_types,
            output_functions.summaries_output_height_per_var,
            precision,
        )
    final_status = (status_flags & STATUS_MASK)

    return {
        "state": state_output,
        "observables": observables_output,
        "state_summaries": state_summary,
        "observable_summaries": observable_summary,
        "status": final_status,
    }
