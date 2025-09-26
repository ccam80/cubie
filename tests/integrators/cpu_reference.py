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
        # Create symbol-to-value mapping for easy lookup
        self._symbol_to_index = {}

        # Add state variables
        for symbol, index in self._state_index.items():
            self._symbol_to_index[symbol] = ('state', index)

        # Add parameters
        for symbol, index in self._parameter_index.items():
            self._symbol_to_index[symbol] = ('param', index)

        # Add constants
        for symbol, index in self._constant_index.items():
            self._symbol_to_index[symbol] = ('const', index)

        # Add drivers
        for symbol, index in self._driver_index.items():
            self._symbol_to_index[symbol] = ('driver', index)

        # Compile each equation's RHS with only the symbols it depends on
        self._compiled_equations = []
        self._equation_targets = []
        self._equation_symbols = []  # Track which symbols each equation needs

        for lhs, rhs in self._equations:
            # Find which symbols this expression actually uses
            expr_symbols = list(rhs.free_symbols)

            # Compile the RHS expression with only the symbols it needs
            if expr_symbols:
                compiled_fn = sp.lambdify(expr_symbols, rhs, modules=['numpy'])
            else:
                # Handle constant expressions
                compiled_fn = lambda: self.precision(rhs)

            self._compiled_equations.append(compiled_fn)
            self._equation_targets.append(lhs)
            self._equation_symbols.append(expr_symbols)

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
            observables: Array,
            computed_values: dict
        ) -> dict:
        """Get current values for all symbols.
        """
        drivers = _ensure_array(drivers, self.precision)

        values = {}

        # Add state values
        for symbol, index in self._state_index.items():
            values[symbol] = self.precision(state[index])

        # Add parameter values
        for symbol, index in self._parameter_index.items():
            values[symbol] = self.precision(params[index])

        # Add constant values
        for symbol, index in self._constant_index.items():
            values[symbol] = self.precision(self.system.constants.values_array[index])

        # Add driver values
        for symbol, index in self._driver_index.items():
            values[symbol] = self.precision(drivers[index])

        # Add already-computed observables (may be partially filled during RHS sweep)
        if observables is not None and len(self._observable_index) > 0:
            for symbol, index in self._observable_index.items():
                # Only insert if we have a slot (array sized to n_observables)
                if index < len(observables):
                    values[symbol] = self.precision(observables[index])

        # Add computed intermediate values (auxiliaries / previously evaluated equations)
        values.update(computed_values)

        return values

    def _prepare_args(
            self,
            symbols: list,
            symbol_values: dict
        ) -> tuple:
        """Prepare arguments for a compiled function with specific symbols."""
        args = []
        for symbol in symbols:
            args.append(symbol_values[symbol])
        return tuple(args)

    def rhs(
            self,
            state: Array,
            params: Array,
            drivers: Array
        ) -> tuple[Array, Array]:
        dxdt = self._state_template.copy()
        observables = self._observable_template.copy()

        # Track computed intermediate values
        computed_values = {}

        # Evaluate each compiled equation
        for i, (compiled_fn, lhs, expr_symbols) in enumerate(zip(
            self._compiled_equations, self._equation_targets, self._equation_symbols
        )):
            # Get current symbol values including any computed intermediates & observables so far
            symbol_values = self._get_symbol_values(state, params, drivers, observables, computed_values)

            # Prepare arguments for this specific function
            if expr_symbols:
                args = self._prepare_args(expr_symbols, symbol_values)
                value = self.precision(compiled_fn(*args))
            else:
                # Constant expression
                value = self.precision(compiled_fn())

            computed_values[lhs] = value

            if lhs in self._dx_index:
                dxdt[self._dx_index[lhs]] = value
            elif lhs in self._observable_index:
                observables[self._observable_index[lhs]] = value
            # auxiliary expressions are stored in computed_values for dependent equations

        return dxdt, observables

    def jacobian(self, state: Array, params: Array, drivers: Array) -> Array:
        if not self._compiled_jacobian:
            return np.zeros((self.n_states, self.n_states), dtype=self.precision)

        # Precompute observables (and auxiliary expressions) so that any
        # derivative expressions depending on them receive correct numeric values.
        # We discard dxdt here; only need observables array.
        _, observables = self.rhs(state, params, drivers)

        symbol_values = self._get_symbol_values(state, params, drivers, observables, {})
        jac_rows = len(self._compiled_jacobian)
        jac_cols = len(self._compiled_jacobian[0]) if jac_rows > 0 else 0
        jacobian = np.zeros((jac_rows, jac_cols), dtype=self.precision)

        for i, (row, row_symbols) in enumerate(zip(self._compiled_jacobian, self._jacobian_symbols)):
            for j, (compiled_entry, expr_symbols) in enumerate(zip(row, row_symbols)):
                if expr_symbols:
                    # Refresh symbol values only if needed (observables already reflect state)
                    args = self._prepare_args(expr_symbols, symbol_values)
                    jacobian[i, j] = self.precision((compiled_entry(*args)))
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
) -> StepResult:
    """Explicit Euler integration step."""

    dxdt, observables = evaluator.rhs(state, params, drivers_now)
    new_state = state + dt * dxdt
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
) -> StepResult:
    """Backward Euler step solved via Newton iteration."""

    precision = evaluator.precision

    def residual(candidate: Array) -> Array:
        dxdt, _ = evaluator.rhs(candidate, params, drivers_next)
        return candidate - state - dt * dxdt

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers_next)
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
    dxdt, observables = evaluator.rhs(next_state, params, drivers_next)
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
) -> StepResult:
    """Crank–Nicolson step with embedded backward Euler for error estimation."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now)

    def residual(candidate: Array) -> Array:
        f_candidate, _ = evaluator.rhs(candidate, params, drivers_next)
        return candidate - state - precision(0.5) * dt * (f_now + f_candidate)

    def jacobian(candidate: Array) -> Array:
        jac = evaluator.jacobian(candidate, params, drivers_next)
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
    _, observables = evaluator.rhs(next_state, params, drivers_next)

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
) -> StepResult:
    """Predict with explicit Euler and correct with backward Euler."""

    precision = evaluator.precision
    f_now, _ = evaluator.rhs(state, params, drivers_now)
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
        precision: np.dtype,
        kp: float = 0.7,
        ki: float = 0.4,
        kd: float = 0.1,
        gamma: float = 0.9,
        max_newton_iters: int = 4,
    ) -> None:

        self.kind = kind.lower()
        self.dt_min = precision(dt_min)
        self.dt_max = precision(dt_max)
        self.dt = precision(dt_min)
        self.atol = precision(atol)
        self.rtol = precision(rtol)
        self.order = max(int(order), 1)
        self.precision = precision
        self.safety = precision(0.9)
        self.min_gain = precision(0.2)
        self.max_gain = precision(2.0)
        self.kp = precision(kp)
        self.ki = precision(ki)
        self.kd = precision(kd)
        self.gamma = precision(gamma)
        self.max_newton_iters = int(max_newton_iters)
        zero = precision(0.0)
        self._history: list[precision] = [zero, zero]
        self._dt_history: list[precision] = [zero, zero]
        self._step_count = 0
        self._convergence_failed = False
        self._rejections_at_dt_min = 0
        self._prev_nrm2 = zero
        self._prev_inv_nrm2 = zero
        self._prev_dt = zero
        self._dt_floor = precision(1e-30)
        self._gain_cap = precision(1e4)

    @property
    def is_adaptive(self) -> bool:
        return self.kind != "fixed"

    def error_norm(self, state_prev: Array, state_new: Array, error: Array) -> float:
        scale = self.atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            nrm2 = np.sum((scale * scale) / (error * error))
            norm = nrm2 / len(error)
        if not np.isfinite(norm):
            norm = 1e5 #Arbitrarily big
        return self.precision(norm)

    def propose_dt(self, error_estimate: float, accept: bool, niters: int = 0) -> None:
        self._step_count += 1

        if not self.is_adaptive:
            # print(f"[PID] Step #{self._step_count}: Fixed step size dt={self.dt:.6e}, error_norm={error_estimate:.2e}, ACCEPT")
            return

        # Save current dt before computing new gain (needed for gustafsson)
        current_dt = self.dt
        gain = self._gain(error_estimate, accept, niters, current_dt)
        unclamped_dt = self.precision(current_dt * gain)

        # Clamp
        new_dt = min(self.dt_max, max(self.dt_min, unclamped_dt))
        self.dt = new_dt

        # Track dt floor rejections
        if (not accept) and (new_dt <= self.dt_min + self._dt_floor):
            self._rejections_at_dt_min += 1
            if self._rejections_at_dt_min >= 8:
                self._convergence_failed = True
        elif accept:
            self._rejections_at_dt_min = 0

        # Update histories (store nrm2 capped like device code)
        zero = self.precision(0.0)
        capped = (
            min(error_estimate, self._gain_cap) if error_estimate > zero else zero
        )
        self._prev_nrm2 = capped
        self._prev_dt = current_dt  # previous accepted/attempted dt used for next gustafsson

    def _gain(self, error_estimate: float, accept: bool, niters: int, current_dt: float) -> float:
        precision = self.precision
        one = precision(1.0)
        two = precision(2.0)

        if error_estimate <= 0.0:
            return self.max_gain

        if self.kind == "i":
            # Device: safety * (nrm2 ** (1.0 / (2 * (order + 1))))
            exponent = one / (two * (self.order + one))
            gain = self.safety * (error_estimate ** exponent)

        elif self.kind == "pi":
            kp_exp = (self.kp / (self.order + one)) / two
            ki_exp = (self.ki / (self.order + one)) / two
            prev = self._prev_nrm2 if self._prev_nrm2 > 0.0 else error_estimate
            gain = self.safety * (error_estimate ** kp_exp) * (prev ** ki_exp)

        elif self.kind == "pid":
            # Device-aligned: gain = safety * (nrm2**kp) * (prev_nrm2**ki) * ((nrm2/prev_nrm2)**kd)
            prev_nrm2 = self._prev_nrm2 if self._prev_nrm2 > 0.0 else error_estimate
            prev_inv = (
                self._prev_inv_nrm2
                if self._prev_inv_nrm2 > 0.0
                else (one / error_estimate)
            )
            kp_exp = self.kp / (two * (self.order + one))
            ki_exp = self.ki / (two * (self.order + one))
            kd_exp = self.kd / (two * (self.order + one))
            ratio_term = error_estimate * prev_inv  # nrm2 / prev_nrm2
            gain = (
                self.safety
                * (error_estimate ** kp_exp)
                * (prev_nrm2 ** ki_exp)
                * (ratio_term ** kd_exp)
            )

        elif self.kind == "gustafsson":
            expo = one / (two * (self.order + one))
            M = self.max_newton_iters
            nit = max(1, niters)  # ensure positive
            fac = min(self.gamma, ((one + two * M) * self.gamma) / (
                    nit + two * M))
            gain_basic = self.safety * fac * (error_estimate ** expo)
            use_gus = accept and (self._prev_dt > 0.0) and (self._prev_nrm2 > 0.0)
            if use_gus:
                ratio = (error_estimate * error_estimate) / self._prev_nrm2
                gain_gus = (
                    self.safety
                    * (current_dt / self._prev_dt)
                    * (ratio ** expo)
                    * self.gamma
                )
                gain = gain_gus if gain_gus < gain_basic else gain_basic
            else:
                gain = gain_basic
        else:
            gain = one

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
    forcing_vectors = inputs["drivers"].astype(precision, copy=False)

    dt_min = precision(solver_settings["dt_min"])
    dt_max = precision(solver_settings["dt_max"])
    duration = precision(solver_settings["duration"])
    warmup = precision(solver_settings["warmup"])
    dt_save = precision(solver_settings["dt_save"])
    dt_summarise = precision(solver_settings["dt_summarise"])
    controller_settings = step_controller_settings or {"kind": "fixed"}
    status_flags = 0
    last_step_status = 0
    zero = precision(0.0)

    step_fn = get_ref_step_fn(solver_settings["algorithm"])

    # Store the tolerance for explicit passing to step functions
    # nonlinear_tol = implicit_step_settings["nonlinear_tolerance"]

    controller.dt = precision(controller_settings.get("dt", dt_min))

    sampler = DriverSampler(forcing_vectors, dt_save, precision)
    save_index = 0
    if warmup <= zero and dt_save > zero:
        save_index = 1

    saved_state_indices = _ensure_array(
        solver_settings["saved_state_indices"], np.int32)
    saved_observable_indices = _ensure_array(
        solver_settings["saved_observable_indices"], np.int32)
    summarised_state_indices = _ensure_array(
        solver_settings["summarised_state_indices"], np.int32
    )
    summarised_observable_indices = _ensure_array(
        solver_settings["summarised_observable_indices"], np.int32)

    save_time = output_functions.save_time

    max_save_samples = int(np.ceil(duration / dt_save))

    state_history = [initial_state.copy()]
    observable_history = [
        np.zeros(len(saved_observable_indices), dtype=precision)
    ]
    time_history: list[float] = [precision(0.0)]
    state = initial_state.copy()
    t = precision(0.0)
    end_time = precision(warmup + duration)
    next_save_time = (
        warmup + dt_save if dt_save > zero else end_time + precision(1.0)
    )
    max_iters = int(implicit_step_settings.get('max_iterations', 25))

    while t < end_time - precision(1e-12):
        dt = min(controller.dt, end_time - t)
        dt = min(dt, next_save_time - t)

        driver_sample = sampler.sample(save_index)
        # Mirror the device loop where the shared driver buffer holds the
        # values associated with the upcoming save index for the entire step.
        drivers_now = driver_sample
        drivers_next = driver_sample

        # Pass max_iters from implicit_step_settings to step functions

        result = step_fn(
                evaluator,
                state=state,
                params=params,
                drivers_now=drivers_now,
                drivers_next=drivers_next,
                dt=dt,
                tol=implicit_step_settings['nonlinear_tolerance'],
                max_iters=max_iters
            )
        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        
        error_norm = controller.error_norm(state, result.state, result.error)

        if controller.is_adaptive and error_norm < 1.0:
            controller.propose_dt(error_norm, accept=False, niters=result.niters)
            if hasattr(controller, '_convergence_failed') and controller._convergence_failed:
                print(f"[LOOP] Integration failed: unclamped dt < dt_min at t={t:.6e}")
                status_flags |= int(IntegratorReturnCodes.STEP_TOO_SMALL)

                break
            continue

        t += dt
        controller.propose_dt(error_norm, accept=True, niters=result.niters)
        state = result.state
        last_step_status = step_status

        if t >= warmup and len(state_history) < max_save_samples:
            if t + precision(1e-12) >= next_save_time:
                state_history.append(state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(precision(next_save_time - warmup))
                next_save_time += dt_save
                save_index += 1

    # ------------------------------------------------------------------
    # Convert histories to arrays matching device layout
    # ------------------------------------------------------------------

    state_output = _collect_saved_outputs(
        state_history,
        saved_state_indices,
        precision,
    )
    if save_time:
        state_output = np.column_stack((state_output, np.asarray(time_history)))
    assert controller._convergence_failed is False, \
        ("Integration failed: CPU solver did not converge, try decreasing "
         "dt_min")


    observables_output = _collect_saved_outputs(
        observable_history,
        saved_observable_indices,
        precision,
    )
    summarise_every = int(np.ceil(dt_summarise / dt_save))

    if saved_state_indices.shape[0] > 0:
        state_summary_source = _collect_saved_outputs(
            state_history,
            summarised_state_indices,
            precision,
        )
    else:
        state_summary_source = np.zeros(
                (len(state_history), 1), dtype=precision
        )
    if saved_observable_indices.shape[0] > 0:
        observable_summary_source = _collect_saved_outputs(
            observable_history, summarised_observable_indices, precision
        )
    else:
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
    final_status = (status_flags & STATUS_MASK)

    return {
        "state": state_output,
        "observables": observables_output,
        "state_summaries": state_summary,
        "observable_summaries": observable_summary,
        "status": final_status,
    }
