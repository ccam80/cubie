from __future__ import annotations
import os
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "0")

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.step_control import (
    AdaptivePIDController,
    FixedStepController,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@dataclass
class StepAttempt:
    """Record of a single attempted step from the perspective of the loop."""

    index: int
    t_start: float
    dt_used: float
    dt_after: float
    accepted: bool
    step_status: int
    controller_status: int
    error_norm: float


@dataclass
class RunResult:
    """Container for the time history produced by one execution path."""

    times: List[float]
    states: List[np.ndarray]
    errors: List[np.ndarray]
    attempts: List[StepAttempt]


@dataclass
class AlgorithmComparison:
    """CPU/GPU comparison for a single algorithm on one problem."""

    cpu: RunResult
    gpu: RunResult


@dataclass
class ProblemDefinition:
    """Definition of a simple test problem."""

    name: str
    system: object
    analytic: Callable[[np.ndarray, float, Dict[str, float]], np.ndarray]
    derivative: Callable[[float, Dict[str, float]], float]
    derivative_prime: Callable[[float, Dict[str, float]], float]
    initial_state: np.ndarray
    parameters: np.ndarray
    parameter_names: Tuple[str, ...]
    duration: float
    dt_min: float
    dt_max: float
    dt_initial: float
    atol: np.ndarray
    rtol: np.ndarray
    precision: np.dtype

    @property
    def parameter_dict(self) -> Dict[str, float]:
        """Return parameter values keyed by name."""

        return {
            name: float(self.parameters[idx])
            for idx, name in enumerate(self.parameter_names)
        }


def linear_derivative(state: float, params: Dict[str, float]) -> float:
    """Return derivative for the linear decay test."""

    lam = params["lam"]
    return -lam * state


def linear_derivative_prime(state: float, params: Dict[str, float]) -> float:
    """Jacobian of the linear decay derivative."""

    del state
    return -params["lam"]


def linear_analytic(times: np.ndarray,
                    initial: float,
                    params: Dict[str, float]) -> np.ndarray:
    """Analytical solution for a linear decay."""

    lam = params["lam"]
    return initial * np.exp(-lam * times)


def logistic_derivative(state: float, params: Dict[str, float]) -> float:
    """Derivative for a logistic growth law."""

    r = params["r"]
    kappa = params["kappa"]
    return r * state * (1.0 - state / kappa)


def logistic_derivative_prime(state: float, params: Dict[str, float]) -> float:
    """Jacobian of the logistic growth derivative."""

    r = params["r"]
    kappa = params["kappa"]
    return r * (1.0 - (2.0 * state) / kappa)


def logistic_analytic(times: np.ndarray,
                      initial: float,
                      params: Dict[str, float]) -> np.ndarray:
    """Analytical solution of the logistic growth equation."""

    r = params["r"]
    kappa = params["kappa"]
    ratio = (kappa / initial) - 1.0
    return kappa / (1.0 + ratio * np.exp(-r * times))


def build_problem_definitions(precision: np.dtype) -> List[ProblemDefinition]:
    """Construct linear and nonlinear problems for comparison."""

    problems: List[ProblemDefinition] = []

    linear_system = create_ODE_system(
        dxdt=["dx0 = -lam * x0", "o0 = x0"],
        states={"x0": 1.0},
        parameters={"lam": 1.0},
        constants={},
        drivers=[],
        observables=["o0"],
        precision=precision,
        strict=True,
    )
    linear_system.build()
    lam_params = linear_system.parameters.values_array.astype(precision)
    problems.append(
        ProblemDefinition(
            name="linear_decay",
            system=linear_system,
            analytic=linear_analytic,
            derivative=linear_derivative,
            derivative_prime=linear_derivative_prime,
            initial_state=linear_system.initial_values.values_array.astype(
                precision
            ),
            parameters=lam_params,
            parameter_names=tuple(linear_system.parameters.names),
            duration=2.0,
            dt_min=0.01,
            dt_max=0.25,
            dt_initial=0.08,
            atol=np.full(linear_system.sizes.states, 1e-6, dtype=precision),
            rtol=np.full(linear_system.sizes.states, 1e-6, dtype=precision),
            precision=precision,
        )
    )

    logistic_system = create_ODE_system(
        dxdt=["dx0 = r * x0 * (1 - x0 / kappa)", "o0 = x0"],
        states={"x0": 0.25},
        parameters={"r": 1.5, "kappa": 1.0},
        constants={},
        drivers=[],
        observables=["o0"],
        precision=precision,
        strict=True,
    )
    logistic_system.build()
    log_params = logistic_system.parameters.values_array.astype(precision)
    problems.append(
        ProblemDefinition(
            name="logistic_growth",
            system=logistic_system,
            analytic=logistic_analytic,
            derivative=logistic_derivative,
            derivative_prime=logistic_derivative_prime,
            initial_state=logistic_system.initial_values.values_array.astype(
                precision
            ),
            parameters=log_params,
            parameter_names=tuple(logistic_system.parameters.names),
            duration=4.0,
            dt_min=0.01,
            dt_max=0.25,
            dt_initial=0.08,
            atol=np.full(logistic_system.sizes.states, 1e-6, dtype=precision),
            rtol=np.full(logistic_system.sizes.states, 1e-6, dtype=precision),
            precision=precision,
        )
    )

    return problems


def compute_error_norm(state_prev: np.ndarray,
                       state_new: np.ndarray,
                       error: np.ndarray,
                       atol: np.ndarray,
                       rtol: np.ndarray) -> float:
    """Compute the inverted error norm used by the controllers."""

    if error.size == 0:
        return math.inf

    accum = 0.0
    count = error.size
    for idx in range(count):
        tolerance = float(
            atol[idx] + rtol[idx] *
            max(abs(state_prev[idx]), abs(state_new[idx]))
        )
        denominator = float(error[idx])
        if denominator == 0.0:
            ratio_sq = math.inf
        else:
            ratio = tolerance / abs(denominator)
            ratio_sq = ratio * ratio
        accum += ratio_sq
    norm = accum / count
    if not math.isfinite(norm):
        return math.inf
    return norm


def newton_solve(residual: Callable[[float], float],
                 derivative: Callable[[float], float],
                 guess: float,
                 tol: float = 1e-12,
                 max_iters: int = 64) -> float:
    """Simple scalar Newton iteration."""

    x_val = guess
    for _ in range(max_iters):
        resid = residual(x_val)
        if abs(resid) < tol:
            return x_val
        deriv = derivative(x_val)
        if deriv == 0.0:
            break
        x_val -= resid / deriv
    return x_val


def cpu_step(problem: ProblemDefinition,
             algorithm: str,
             state_val: float,
             dt_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """Execute one CPU reference step."""

    params = problem.parameter_dict
    precision = problem.precision

    if algorithm == "euler":
        derivative = problem.derivative(state_val, params)
        proposed = state_val + dt_val * derivative
        error_vec = np.zeros(1, dtype=precision)
    elif algorithm == "backwards_euler":
        def residual(x_val: float) -> float:
            return x_val - state_val - dt_val * problem.derivative(x_val, params)

        def deriv(x_val: float) -> float:
            return 1.0 - dt_val * problem.derivative_prime(x_val, params)

        proposed = newton_solve(residual, deriv, state_val)
        error_vec = np.zeros(1, dtype=precision)
    elif algorithm == "backwards_euler_pc":
        predictor = state_val + dt_val * problem.derivative(state_val, params)

        def residual(x_val: float) -> float:
            return x_val - state_val - dt_val * problem.derivative(x_val, params)

        def deriv(x_val: float) -> float:
            return 1.0 - dt_val * problem.derivative_prime(x_val, params)

        proposed = newton_solve(residual, deriv, predictor)
        error_vec = np.zeros(1, dtype=precision)
    elif algorithm == "crank_nicolson":
        f_now = problem.derivative(state_val, params)

        def residual(x_val: float) -> float:
            return (
                x_val
                - state_val
                - 0.5 * dt_val * (f_now + problem.derivative(x_val, params))
            )

        def deriv(x_val: float) -> float:
            return 1.0 - 0.5 * dt_val * problem.derivative_prime(x_val, params)

        cn_state = newton_solve(residual, deriv, state_val)

        def residual_be(x_val: float) -> float:
            return x_val - state_val - dt_val * problem.derivative(x_val, params)

        def deriv_be(x_val: float) -> float:
            return 1.0 - dt_val * problem.derivative_prime(x_val, params)

        be_state = newton_solve(residual_be, deriv_be, cn_state)
        proposed = cn_state
        error_vec = np.array([cn_state - be_state], dtype=precision)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported algorithm {algorithm}")

    proposed_arr = np.array([proposed], dtype=precision)
    return proposed_arr, error_vec


class CPUAdaptivePID:
    """Scalar implementation of the GPU PID controller."""

    def __init__(self,
                 dt_min: float,
                 dt_max: float,
                 dt_initial: float,
                 atol: np.ndarray,
                 rtol: np.ndarray,
                 order: int,
                 kp: float = 0.7,
                 ki: float = 0.4,
                 kd: float = 0.2,
                 safety: float = 0.9,
                 min_gain: float = 0.2,
                 max_gain: float = 5.0) -> None:
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt = dt_initial
        self.atol = atol
        self.rtol = rtol
        self.order = order
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.safety = safety
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.prev_norm = 0.0
        self.prev_inv = 0.0

    def propose(self,
                dt_used: float,
                state_prev: np.ndarray,
                state_new: np.ndarray,
                error: np.ndarray) -> Tuple[bool, float, float, int]:
        """Return acceptance decision and next step size."""

        norm = compute_error_norm(
            state_prev, state_new, error, self.atol, self.rtol)
        accept = norm >= 1.0

        prev_norm = self.prev_norm if self.prev_norm > 0.0 else norm
        prev_inv = self.prev_inv if self.prev_inv > 0.0 else (
            1.0 / norm if norm > 0 else 0.0)
        expo = 2.0 * (self.order + 1.0)
        gain = self.safety
        gain *= norm ** (self.kp / expo)
        gain *= prev_norm ** (self.ki / expo)
        ratio = norm * prev_inv
        gain *= ratio ** (self.kd / expo)
        gain = min(self.max_gain, max(self.min_gain, gain))

        raw_next = dt_used * gain
        next_dt = min(self.dt_max, max(self.dt_min, raw_next))
        floor_reached = next_dt <= self.dt_min * (1.0 + 1e-12)
        status = 1 if (not accept and floor_reached) else 0
        if status == 1:
            accept = True
        self.prev_norm = norm if math.isfinite(norm) and norm > 0 else 0.0
        self.prev_inv = (
            1.0 / norm) if math.isfinite(norm) and norm > 0 else 0.0
        self.dt = next_dt
        return accept, next_dt, norm, status


def run_cpu_algorithm(problem: ProblemDefinition,
                      algorithm: str) -> RunResult:
    """Integrate a problem using the CPU reference logic."""

    times: List[float] = [0.0]
    states: List[np.ndarray] = [problem.initial_state.copy()]
    errors: List[np.ndarray] = []
    attempts: List[StepAttempt] = []

    state = problem.initial_state.astype(problem.precision)
    time_now = 0.0
    dt_current = problem.dt_initial
    pid_controller = CPUAdaptivePID(
        dt_min=problem.dt_min,
        dt_max=problem.dt_max,
        dt_initial=problem.dt_initial,
        atol=problem.atol,
        rtol=problem.rtol,
        order=2,
    )
    attempt_index = 0

    while time_now < problem.duration - 1e-12:
        dt_eff = min(dt_current, problem.duration - time_now)
        proposed_state, error_vec = cpu_step(
            problem, algorithm, float(state[0]), dt_eff
        )

        if algorithm == "crank_nicolson":
            accept, dt_next, norm, status = pid_controller.propose(
                dt_eff, state, proposed_state, error_vec
            )
        else:
            accept = True
            dt_next = dt_eff
            norm = compute_error_norm(
                state, proposed_state, error_vec, problem.atol, problem.rtol
            )
            status = 0

        attempts.append(
            StepAttempt(
                index=attempt_index,
                t_start=time_now,
                dt_used=dt_eff,
                dt_after=dt_next,
                accepted=accept,
                step_status=0,
                controller_status=status,
                error_norm=norm,
            )
        )
        attempt_index += 1

        dt_current = dt_next
        if accept:
            time_now += dt_eff
            state = proposed_state
            times.append(time_now)
            states.append(state.copy())
            errors.append(error_vec.copy())

    return RunResult(times=times, states=states, errors=errors, attempts=attempts)


def build_step_kernel(step_fn: Callable,
                      numba_precision,
                      shared_elems: int,
                      persistent_len: int):
    """Compile a CUDA kernel that executes a single step."""

    shared_len = max(shared_elems, 0)
    persistent_size = max(persistent_len, 1)

    @cuda.jit(debug=True)
    def kernel(state,
               proposed,
               work,
               parameters,
               drivers,
               observables,
               error,
               status,
               dt_scalar):
        idx = cuda.grid(1)
        if idx > 0:
            return
        if shared_len > 0:
            shared = cuda.shared.array(shared_len, dtype=numba_precision)
        else:
            shared = cuda.shared.array(1, dtype=numba_precision)
        persistent = cuda.local.array(persistent_size, dtype=numba_precision)
        status[0] = step_fn(
            state,
            proposed,
            work,
            parameters,
            drivers,
            observables,
            error,
            dt_scalar,
            shared,
            persistent,
        )

    return kernel


def build_controller_kernel(controller_fn: Callable):
    """Compile a CUDA kernel that runs a controller exactly once."""

    @cuda.jit(debug=True)
    def kernel(dt_array,
               state,
               proposed,
               error,
               accept,
               local_temp,
               status):
        idx = cuda.grid(1)
        if idx > 0:
            return
        status[0] = controller_fn(
            dt_array,
            state,
            proposed,
            error,
            accept,
            local_temp,
        )

    return kernel


def run_gpu_algorithm(problem: ProblemDefinition,
                      algorithm: str,
                      implicit_settings: Dict[str, float]) -> RunResult:
    """Integrate a problem by repeatedly launching the GPU kernels."""

    precision = problem.precision
    system = problem.system
    step_kwargs: Dict[str, object]

    if algorithm == "euler":
        step_kwargs = {
            "dt": problem.dt_initial,
            "precision": precision,
            "n": system.sizes.states,
            "dxdt_function": system.dxdt_function,
        }
    else:
        step_kwargs = {
            "precision": precision,
            "n": system.sizes.states,
            "dxdt_function": system.dxdt_function,
            "get_solver_helper_fn": system.get_solver_helper,
            "preconditioner_order": implicit_settings["preconditioner_order"],
            "linsolve_tolerance": implicit_settings["linear_tolerance"],
            "max_linear_iters": implicit_settings["max_linear_iters"],
            "linear_correction_type": implicit_settings["correction_type"],
            "nonlinear_tolerance": implicit_settings["nonlinear_tolerance"],
            "max_newton_iters": implicit_settings["max_newton_iters"],
            "newton_damping": implicit_settings["newton_damping"],
            "newton_max_backtracks": implicit_settings["newton_max_backtracks"],
        }

    step_obj = get_algorithm_step(algorithm, **step_kwargs)

    if algorithm == "crank_nicolson":
        controller = AdaptivePIDController(
            precision=precision,
            dt_min=problem.dt_min,
            dt_max=problem.dt_max,
            atol=problem.atol,
            rtol=problem.rtol,
            algorithm_order=2,
            n=system.sizes.states,
            kp=0.7,
            ki=0.4,
            kd=0.2,
        )
    else:
        controller = FixedStepController(
            precision=precision, dt=problem.dt_initial)

    step_fn = step_obj.step_function
    controller_fn = controller.device_function
    numba_precision = step_obj.compile_settings.numba_precision

    step_kernel = build_step_kernel(
        step_fn,
        numba_precision,
        step_obj.shared_memory_required,
        step_obj.persistent_local_required,
    )
    controller_kernel = build_controller_kernel(controller_fn)

    times: List[float] = [0.0]
    states: List[np.ndarray] = [problem.initial_state.copy()]
    errors: List[np.ndarray] = []
    attempts: List[StepAttempt] = []

    state = problem.initial_state.astype(precision)
    params = problem.parameters.astype(precision)
    drivers = np.zeros(system.sizes.drivers, dtype=precision)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    time_now = 0.0
    dt_current = np.array([problem.dt_initial], dtype=precision)
    attempt_index = 0

    while time_now < problem.duration - 1e-12:
        dt_eff = min(float(dt_current[0]), problem.duration - time_now)
        dt_scalar = precision(dt_eff)

        d_state = cuda.to_device(state)
        d_proposed = cuda.to_device(state.copy())
        d_work = cuda.to_device(np.zeros(system.sizes.states, dtype=precision))
        d_params = cuda.to_device(params)
        d_drivers = cuda.to_device(drivers)
        d_observables = cuda.to_device(observables.copy())
        d_error = cuda.to_device(
            np.zeros(system.sizes.states, dtype=precision))
        d_status = cuda.to_device(np.zeros(1, dtype=np.int32))

        step_kernel[1, 1](
            d_state,
            d_proposed,
            d_work,
            d_params,
            d_drivers,
            d_observables,
            d_error,
            d_status,
            dt_scalar,
        )
        cuda.synchronize()

        step_status = int(d_status.copy_to_host()[0])
        proposed_state = d_proposed.copy_to_host()
        error_vec = d_error.copy_to_host()

        if algorithm == "crank_nicolson":
            d_dt = cuda.to_device(np.array([dt_scalar], dtype=precision))
            d_accept = cuda.to_device(np.zeros(1, dtype=np.int32))
            d_local = cuda.to_device(
                np.zeros(controller.local_memory_elements or 1, dtype=precision)
            )
            d_ctrl_status = cuda.to_device(np.zeros(1, dtype=np.int32))
            controller_kernel[1, 1](
                d_dt,
                d_state,
                cuda.to_device(proposed_state),
                cuda.to_device(error_vec),
                d_accept,
                d_local,
                d_ctrl_status,
            )
            cuda.synchronize()
            dt_current = d_dt.copy_to_host()
            accept_flag = bool(d_accept.copy_to_host()[0])
            controller_status = int(d_ctrl_status.copy_to_host()[0])
        else:
            accept_flag = True
            controller_status = 0
            dt_current = np.array([dt_scalar], dtype=precision)

        norm = compute_error_norm(state, proposed_state, error_vec,
                                  problem.atol, problem.rtol)
        attempts.append(
            StepAttempt(
                index=attempt_index,
                t_start=time_now,
                dt_used=dt_eff,
                dt_after=float(dt_current[0]),
                accepted=accept_flag,
                step_status=step_status,
                controller_status=controller_status,
                error_norm=norm,
            )
        )
        attempt_index += 1

        if accept_flag:
            time_now += dt_eff
            state = proposed_state
            times.append(time_now)
            states.append(proposed_state.copy())
            errors.append(error_vec.copy())

    return RunResult(times=times, states=states, errors=errors, attempts=attempts)


def plot_solution_grid(problem: ProblemDefinition,
                       results: Dict[str, AlgorithmComparison],
                       output_dir: Path) -> None:
    """Plot CPU vs GPU solutions for all algorithms on one figure."""

    algorithms = ["euler", "backwards_euler",
                  "backwards_euler_pc", "crank_nicolson"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    time_dense = np.linspace(0.0, problem.duration, 400)
    analytic = problem.analytic(time_dense, float(problem.initial_state[0]),
                                problem.parameter_dict)

    for ax, algo in zip(axes.flatten(), algorithms):
        comparison = results[algo]
        cpu_vals = [state[0] for state in comparison.cpu.states]
        gpu_vals = [state[0] for state in comparison.gpu.states]

        ax.plot(time_dense, analytic, color="black",
                linestyle="-", label="Analytical")
        ax.plot(comparison.cpu.times, cpu_vals, marker="o", linestyle="--",
                label="CPU")
        ax.plot(comparison.gpu.times, gpu_vals, marker="s", linestyle=":",
                label="GPU")
        ax.set_title(algo.replace("_", " ").title())
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"Algorithm comparison on {problem.name}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / f"{problem.name}_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cn_history(problem: ProblemDefinition,
                    comparison: AlgorithmComparison,
                    output_dir: Path) -> None:
    """Plot Crank–Nicolson step-size history for CPU and GPU."""

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, label, run in (
        (axes[0], "GPU", comparison.gpu),
        (axes[1], "CPU", comparison.cpu),
    ):
        indices = [attempt.index for attempt in run.attempts]
        dt_used = [attempt.dt_used for attempt in run.attempts]
        dt_next = [attempt.dt_after for attempt in run.attempts]
        accepted_idx = [
            attempt.index for attempt in run.attempts if attempt.accepted]
        accepted_dt = [
            attempt.dt_used for attempt in run.attempts if attempt.accepted]
        rejected_idx = [
            attempt.index for attempt in run.attempts if not attempt.accepted]
        rejected_dt = [
            attempt.dt_used for attempt in run.attempts if not attempt.accepted]

        axis.plot(indices, dt_used, marker="o", label="dt used")
        axis.plot(indices, dt_next, marker="s",
                  linestyle="--", label="next dt")
        if accepted_idx:
            axis.scatter(accepted_idx, accepted_dt, color="C2", marker="o",
                         label="accepted")
        if rejected_idx:
            axis.scatter(rejected_idx, rejected_dt, color="C3", marker="x",
                         label="rejected")
        axis.set_ylabel("dt")
        axis.set_title(f"{label} Crank–Nicolson step history")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Attempt index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    output_path = output_dir / f"{problem.name}_crank_nicolson_history.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Entry point for the diagnostic script."""

    precision = np.float64
    problems = build_problem_definitions(precision)
    implicit_settings = {
        "preconditioner_order": 1,
        "linear_tolerance": 1e-10,
        "max_linear_iters": 128,
        "correction_type": "minimal_residual",
        "nonlinear_tolerance": 1e-10,
        "max_newton_iters": 64,
        "newton_damping": 0.5,
        "newton_max_backtracks": 8,
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["euler", "backwards_euler",
                  "backwards_euler_pc", "crank_nicolson"]

    for problem in problems:
        results: Dict[str, AlgorithmComparison] = {}
        for algo in algorithms:
            cpu_res = run_cpu_algorithm(problem, algo)
            gpu_res = run_gpu_algorithm(problem, algo, implicit_settings)
            results[algo] = AlgorithmComparison(cpu=cpu_res, gpu=gpu_res)

        plot_solution_grid(problem, results, output_dir)
        plot_cn_history(problem, results["crank_nicolson"], output_dir)
        print(f"Saved diagnostic plots for {problem.name} to {output_dir}")


if __name__ == "__main__":
    main()
