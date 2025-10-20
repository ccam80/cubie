import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    ButcherTableau,
    CrankNicolsonStep,
    DIRKStep,
    ERKStep,
    ExplicitEulerStep,
    GenericRosenbrockWStep,
    resolve_alias,
    resolve_supplied_tableau,
)

from .backwards_euler import (
    BackwardsEulerStep as InstrumentedBackwardsEulerStep,
)
from .backwards_euler_predict_correct import (
    BackwardsEulerPCStep as InstrumentedBackwardsEulerPCStep,
)
from .crank_nicolson import (
    CrankNicolsonStep as InstrumentedCrankNicolsonStep,
)
from .explicit_euler import (
    ExplicitEulerStep as InstrumentedExplicitEulerStep,
)
from .generic_dirk import DIRKStep as InstrumentedDIRKStep
from tests.integrators.cpu_reference import (
    CPUODESystem,
    InstrumentedStepResult,
    STATUS_MASK,
    get_ref_stepper,
)
from .generic_erk import ERKStep as InstrumentedERKStep
from .generic_rosenbrock_w import (
    GenericRosenbrockWStep as InstrumentedRosenbrockStep,
)

STEP_CONSTRUCTOR_TO_INSTRUMENTED: Dict[type, Callable[..., object]] = {
    ExplicitEulerStep: InstrumentedExplicitEulerStep,
    BackwardsEulerStep: InstrumentedBackwardsEulerStep,
    BackwardsEulerPCStep: InstrumentedBackwardsEulerPCStep,
    CrankNicolsonStep: InstrumentedCrankNicolsonStep,
    ERKStep: InstrumentedERKStep,
    DIRKStep: InstrumentedDIRKStep,
    GenericRosenbrockWStep: InstrumentedRosenbrockStep,
}


@dataclass
class InstrumentedKernel:
    """CUDA kernel wrapper and launch metadata for instrumented steps."""

    function: Callable
    shared_bytes: int
    persistent_len: int
    numba_precision: type

@pytest.fixture(scope="session")
def step_inputs(
    system,
    precision,
    initial_state,
    solver_settings,
    cpu_driver_evaluator,
) -> dict:
    """State, parameters, and drivers for a single step execution."""
    width = system.num_drivers
    driver_coefficients = np.array(
        cpu_driver_evaluator.coefficients, dtype=precision, copy=True
    )
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(precision),
        "drivers": np.zeros(width, dtype=precision),
        "driver_coefficients": driver_coefficients,
    }

@dataclass
class DeviceInstrumentedResult:
    """Host-side copies of the arrays produced by the CUDA step."""

    state: np.ndarray
    observables: np.ndarray
    error: np.ndarray
    residuals: np.ndarray
    jacobian_updates: np.ndarray
    stage_states: np.ndarray
    stage_derivatives: np.ndarray
    stage_observables: np.ndarray
    stage_drivers: np.ndarray
    stage_increments: np.ndarray
    solver_initial_guesses: np.ndarray
    solver_solutions: np.ndarray
    solver_iteration_guesses: np.ndarray
    solver_residuals: np.ndarray
    solver_residual_norms: np.ndarray
    solver_operator_outputs: np.ndarray
    solver_preconditioned_vectors: np.ndarray
    solver_iteration_end_x: np.ndarray
    solver_iteration_end_rhs: np.ndarray
    solver_iteration_scale: np.ndarray
    solver_iterations: np.ndarray
    solver_status: np.ndarray
    newton_initial_guesses: Optional[np.ndarray] = None
    newton_iteration_guesses: Optional[np.ndarray] = None
    newton_residuals: Optional[np.ndarray] = None
    newton_squared_norms: Optional[np.ndarray] = None
    linear_initial_guesses: Optional[np.ndarray] = None
    linear_iteration_guesses: Optional[np.ndarray] = None
    linear_residuals: Optional[np.ndarray] = None
    linear_squared_norms: Optional[np.ndarray] = None
    linear_preconditioned_vectors: Optional[np.ndarray] = None
    status: Optional[int] = None
    niters: Optional[int] = None
    extra_vectors: Dict[str, np.ndarray] = field(default_factory=dict)

@dataclass(frozen=True)
class ResolvedInstrumentedStep:
    """Instrumented step constructor and tableau resolved from settings."""

    step_class: Callable[..., object]
    tableau: Optional[ButcherTableau]


def _resolve_instrumented_step_configuration(
    algorithm: str,
    tableau: Optional[Union[str, ButcherTableau]],
) -> ResolvedInstrumentedStep:
    """Return the instrumented constructor and tableau for ``algorithm``."""

    try:
        step_constructor, resolved_tableau = resolve_alias(algorithm.lower())
    except KeyError as error:
        raise ValueError(
            f"Unknown instrumented algorithm '{algorithm}'."
        ) from error

    tableau_value = resolved_tableau
    if isinstance(tableau, str):
        try:
            override_constructor, tableau_override = resolve_alias(
            tableau.lower()
        )
        except KeyError as error:
            raise ValueError(
                f"Unknown {step_constructor.__name__} tableau '{tableau}'."
            ) from error
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
        override_constructor, tableau_override = resolve_supplied_tableau(
            tableau
        )
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau instance does not match the requested algorithm type."
            )
        tableau_value = tableau_override
    elif tableau is not None:
        raise TypeError(
            "Expected tableau alias or ButcherTableau instance, "
            f"received {type(tableau).__name__}."
        )

    try:
        step_class = STEP_CONSTRUCTOR_TO_INSTRUMENTED[step_constructor]
    except KeyError as error:
        raise ValueError(
            f"No instrumented implementation registered for {algorithm}."
        ) from error

    return ResolvedInstrumentedStep(
        step_class=step_class,
        tableau=tableau_value,
    )


@pytest.fixture(scope="session")
def instrumented_step_configuration(
    solver_settings,
) -> ResolvedInstrumentedStep:
    """Resolve the instrumented step constructor and tableau."""

    return _resolve_instrumented_step_configuration(
        solver_settings["algorithm"],
        solver_settings.get("tableau"),
    )


@pytest.fixture(scope="session")
def instrumented_step_class(
    instrumented_step_configuration: ResolvedInstrumentedStep,
) -> Callable[..., object]:
    """Return the instrumented step class resolved for the algorithm."""

    return instrumented_step_configuration.step_class


@pytest.fixture(scope="session")
def instrumented_step_object(
    instrumented_step_class: Callable[..., object],
    system,
    solver_settings,
    precision: np.dtype,
    driver_array,
    instrumented_step_configuration: ResolvedInstrumentedStep,
):
    """Instantiate the configured instrumented step implementation."""

    driver_function = (
        None if driver_array is None else driver_array.evaluation_function
    )
    step_kwargs = {
        "precision": precision,
        "n": system.sizes.states,
        "dt": solver_settings["dt"],
        "dxdt_function": system.dxdt_function,
        "observables_function": system.observables_function,
        "driver_function": driver_function,
        "get_solver_helper_fn": system.get_solver_helper,
        "preconditioner_order": solver_settings["preconditioner_order"],
        "krylov_tolerance": solver_settings["krylov_tolerance"],
        "max_linear_iters": solver_settings["max_linear_iters"],
        "linear_correction_type": solver_settings["correction_type"],
        "newton_tolerance": solver_settings["newton_tolerance"],
        "max_newton_iters": solver_settings["max_newton_iters"],
        "newton_damping": solver_settings["newton_damping"],
        "newton_max_backtracks": solver_settings["newton_max_backtracks"],
    }
    tableau = instrumented_step_configuration.tableau
    if tableau is not None:
        step_kwargs["tableau"] = tableau

    signature = inspect.signature(instrumented_step_class)
    filtered_kwargs = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in step_kwargs:
            filtered_kwargs[name] = step_kwargs[name]
        elif parameter.default is inspect._empty:
            raise TypeError(
                f"Missing required argument '{name}' for "
                f"{instrumented_step_class.__name__}."
            )
    return instrumented_step_class(**filtered_kwargs)


@pytest.fixture(scope="session")
def instrumented_step_kernel(
    instrumented_step_object,
    system,
    solver_settings,
    precision: np.dtype,
    driver_array,
    step_inputs,
) -> InstrumentedKernel:
    """Compile a CUDA kernel that executes the instrumented step."""

    step_function = instrumented_step_object.step_function
    numba_precision = from_dtype(precision)
    shared_elems = instrumented_step_object.shared_memory_required
    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, instrumented_step_object.persistent_local_required)
    driver_function = (
        None if driver_array is None else driver_array.evaluation_function
    )
    observables_function = system.observables_function
    state = np.asarray(step_inputs["state"], dtype=precision)
    params = np.asarray(step_inputs["parameters"], dtype=precision)
    driver_coefficients = np.asarray(
        step_inputs["driver_coefficients"], dtype=precision
    )

    @cuda.jit
    def kernel(
        state_vec,
        proposed_vec,
        params_vec,
        driver_coeffs_vec,
        drivers_vec,
        proposed_drivers_vec,
        observables_vec,
        proposed_observables_vec,
        error_vec,
        residuals_mat,
        jacobian_updates_mat,
        stage_states_mat,
        stage_derivatives_mat,
        stage_observables_mat,
        stage_drivers_mat,
        stage_increments_mat,
        solver_initial_guesses_mat,
        solver_solutions_mat,
        solver_iteration_guesses,
        solver_residuals,
        solver_residual_norms,
        solver_operator_outputs,
        solver_preconditioned_vectors,
        solver_iteration_end_x,
        solver_iteration_end_rhs,
        solver_iteration_scale,
        solver_iterations_vec,
        solver_status_vec,
        dt_scalar,
        time_scalar,
        status_vec,
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        if driver_function is not None:
            driver_function(precision(0.0), driver_coefficients, drivers_vec)
        observables_function(
            state,
            params_vec,
            drivers_vec,
            observables_vec,
            precision(0.0),
        )
        shared[:] = precision(0.0)
        result = step_function(
            state_vec,
            proposed_vec,
            params_vec,
            driver_coeffs_vec,
            drivers_vec,
            proposed_drivers_vec,
            observables_vec,
            proposed_observables_vec,
            error_vec,
            residuals_mat,
            jacobian_updates_mat,
            stage_states_mat,
            stage_derivatives_mat,
            stage_observables_mat,
            stage_drivers_mat,
            stage_increments_mat,
            solver_initial_guesses_mat,
            solver_solutions_mat,
            solver_iteration_guesses,
            solver_residuals,
            solver_residual_norms,
            solver_operator_outputs,
            solver_preconditioned_vectors,
            solver_iteration_end_x,
            solver_iteration_end_rhs,
            solver_iteration_scale,
            solver_iterations_vec,
            solver_status_vec,
            dt_scalar,
            time_scalar,
            shared,
            persistent,
        )
        status_vec[0] = result

    return InstrumentedKernel(
        kernel,
        shared_bytes,
        persistent_len,
        numba_precision,
    )


@pytest.fixture(scope="session")
def instrumented_step_results(
    instrumented_step_kernel: InstrumentedKernel,
    instrumented_step_object,
    step_inputs,
    solver_settings,
    system,
    precision: np.dtype,
) -> DeviceInstrumentedResult:
    """Execute the instrumented CUDA step and collect host-side arrays."""

    kernel = instrumented_step_kernel.function
    shared_bytes = instrumented_step_kernel.shared_bytes
    numba_precision = instrumented_step_kernel.numba_precision
    n_states = system.sizes.states
    n_observables = system.sizes.observables
    stage_count_attr = getattr(
        getattr(instrumented_step_object, "tableau", None),
        "stage_count",
        None,
    )
    if stage_count_attr is None:
        stage_count_attr = getattr(instrumented_step_object, "stage_count", 0)
    stage_count = int(stage_count_attr or 0)
    max_newton_iters = int(solver_settings["max_newton_iters"])

    state = np.asarray(step_inputs["state"], dtype=precision)
    params = np.asarray(step_inputs["parameters"], dtype=precision)
    driver_coefficients = np.asarray(
        step_inputs["driver_coefficients"], dtype=precision
    )
    drivers = np.asarray(step_inputs["drivers"], dtype=precision)
    proposed_state = np.zeros_like(state)
    proposed_drivers = np.zeros_like(drivers)
    error = np.zeros(n_states, dtype=precision)
    observables = np.zeros(n_observables, dtype=precision)
    proposed_observables = np.zeros_like(observables)
    residuals = np.zeros((stage_count, n_states), dtype=precision)
    jacobian_updates = np.zeros_like(residuals)
    stage_states = np.zeros_like(residuals)
    stage_derivatives = np.zeros_like(residuals)
    stage_observables = np.zeros(
        (stage_count, n_observables), dtype=precision
    )
    stage_drivers = np.zeros((stage_count, drivers.shape[0]), dtype=precision)
    stage_increments = np.zeros_like(residuals)
    solver_initial_guesses = np.zeros_like(residuals)
    solver_solutions = np.zeros_like(residuals)
    solver_iteration_guesses = np.zeros(
        (stage_count, max_newton_iters, n_states),
        dtype=precision,
    )
    solver_residuals = np.zeros_like(solver_iteration_guesses)
    solver_residual_norms = np.zeros_like(solver_iteration_guesses)
    solver_operator_outputs = np.zeros_like(solver_iteration_guesses)
    solver_preconditioned_vectors = np.zeros_like(solver_iteration_guesses)
    solver_iteration_end_x = np.zeros_like(solver_iteration_guesses)
    solver_iteration_end_rhs = np.zeros_like(solver_iteration_guesses)
    solver_iteration_scale = np.zeros(
        (stage_count, max_newton_iters), dtype=precision
    )
    solver_iterations = np.zeros(stage_count, dtype=np.int32)
    solver_status = np.zeros(stage_count, dtype=np.int32)
    status = np.zeros(1, dtype=np.int32)

    d_state = cuda.to_device(state)
    d_proposed = cuda.to_device(proposed_state)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)
    d_drivers = cuda.to_device(drivers)
    d_proposed_drivers = cuda.to_device(proposed_drivers)
    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(proposed_observables)
    d_error = cuda.to_device(error)
    d_residuals = cuda.to_device(residuals)
    d_jacobian_updates = cuda.to_device(jacobian_updates)
    d_stage_states = cuda.to_device(stage_states)
    d_stage_derivatives = cuda.to_device(stage_derivatives)
    d_stage_observables = cuda.to_device(stage_observables)
    d_stage_drivers = cuda.to_device(stage_drivers)
    d_stage_increments = cuda.to_device(stage_increments)
    d_solver_initial_guesses = cuda.to_device(solver_initial_guesses)
    d_solver_solutions = cuda.to_device(solver_solutions)
    d_solver_iteration_guesses = cuda.to_device(solver_iteration_guesses)
    d_solver_residuals = cuda.to_device(solver_residuals)
    d_solver_residual_norms = cuda.to_device(solver_residual_norms)
    d_solver_operator_outputs = cuda.to_device(solver_operator_outputs)
    d_solver_preconditioned_vectors = cuda.to_device(
        solver_preconditioned_vectors
    )
    d_solver_iteration_end_x = cuda.to_device(solver_iteration_end_x)
    d_solver_iteration_end_rhs = cuda.to_device(solver_iteration_end_rhs)
    d_solver_iteration_scale = cuda.to_device(solver_iteration_scale)
    d_solver_iterations = cuda.to_device(solver_iterations)
    d_solver_status = cuda.to_device(solver_status)
    d_status = cuda.to_device(status)

    dt_value = solver_settings["dt"]

    kernel[1, 1, 0, shared_bytes](
        d_state,
        d_proposed,
        d_params,
        d_driver_coeffs,
        d_drivers,
        d_proposed_drivers,
        d_observables,
        d_proposed_observables,
        d_error,
        d_residuals,
        d_jacobian_updates,
        d_stage_states,
        d_stage_derivatives,
        d_stage_observables,
        d_stage_drivers,
        d_stage_increments,
        d_solver_initial_guesses,
        d_solver_solutions,
        d_solver_iteration_guesses,
        d_solver_residuals,
        d_solver_residual_norms,
        d_solver_operator_outputs,
        d_solver_preconditioned_vectors,
        d_solver_iteration_end_x,
        d_solver_iteration_end_rhs,
        d_solver_iteration_scale,
        d_solver_iterations,
        d_solver_status,
        dt_value,
        numba_precision(0.0),
        d_status,
    )
    cuda.synchronize()

    status_value = int(d_status.copy_to_host()[0])
    solver_iteration_guesses_host = (
        d_solver_iteration_guesses.copy_to_host()
    )
    solver_residuals_host = d_solver_residuals.copy_to_host()
    solver_residual_norms_host = d_solver_residual_norms.copy_to_host()
    solver_operator_outputs_host = d_solver_operator_outputs.copy_to_host()
    solver_preconditioned_vectors_host = (
        d_solver_preconditioned_vectors.copy_to_host()
    )
    solver_iteration_end_x_host = d_solver_iteration_end_x.copy_to_host()
    solver_iteration_end_rhs_host = d_solver_iteration_end_rhs.copy_to_host()
    solver_iteration_scale_host = d_solver_iteration_scale.copy_to_host()
    return DeviceInstrumentedResult(
        state=d_proposed.copy_to_host(),
        observables=d_proposed_observables.copy_to_host(),
        error=d_error.copy_to_host(),
        residuals=d_residuals.copy_to_host(),
        jacobian_updates=d_jacobian_updates.copy_to_host(),
        stage_states=d_stage_states.copy_to_host(),
        stage_derivatives=d_stage_derivatives.copy_to_host(),
        stage_observables=d_stage_observables.copy_to_host(),
        stage_drivers=d_stage_drivers.copy_to_host(),
        stage_increments=d_stage_increments.copy_to_host(),
        solver_initial_guesses=d_solver_initial_guesses.copy_to_host(),
        solver_solutions=d_solver_solutions.copy_to_host(),
        solver_iteration_guesses=solver_iteration_guesses_host,
        solver_residuals=solver_residuals_host,
        solver_residual_norms=solver_residual_norms_host,
        solver_operator_outputs=solver_operator_outputs_host,
        solver_preconditioned_vectors=solver_preconditioned_vectors_host,
        solver_iteration_end_x=solver_iteration_end_x_host,
        solver_iteration_end_rhs=solver_iteration_end_rhs_host,
        solver_iteration_scale=solver_iteration_scale_host,
        solver_iterations=d_solver_iterations.copy_to_host().astype(
            np.int64
        ),
        solver_status=d_solver_status.copy_to_host().astype(np.int64),
        status=status_value & STATUS_MASK,
        niters=(status_value >> 16) & STATUS_MASK,
        extra_vectors={
            "solver_iteration_guesses": solver_iteration_guesses_host,
            "solver_residuals": solver_residuals_host,
            "solver_residual_norms": solver_residual_norms_host,
            "solver_operator_outputs": solver_operator_outputs_host,
            "solver_preconditioned_vectors": (
                solver_preconditioned_vectors_host
            ),
            "solver_iteration_end_x": solver_iteration_end_x_host,
            "solver_iteration_end_rhs": solver_iteration_end_rhs_host,
            "solver_iteration_scale": solver_iteration_scale_host,
        },
    )


@pytest.fixture(scope="session")
def instrumented_cpu_step_results(
    solver_settings,
    cpu_system: CPUODESystem,
    step_inputs,
    cpu_driver_evaluator,
    instrumented_step_object,
    instrumented_step_configuration: ResolvedInstrumentedStep,
) -> InstrumentedStepResult:
    """Execute the CPU reference step with instrumentation enabled."""

    tableau = instrumented_step_configuration.tableau
    state = np.asarray(
        step_inputs["state"], dtype=cpu_system.precision
    )
    params = np.asarray(
        step_inputs["parameters"], dtype=cpu_system.precision
    )
    if cpu_system.system.num_drivers > 0:
        driver_coefficients = step_inputs["driver_coefficients"]
        driver_evaluator = cpu_driver_evaluator.with_coefficients(
            driver_coefficients
        )
    else:
        driver_evaluator = cpu_driver_evaluator

    stepper = get_ref_stepper(
        cpu_system,
        driver_evaluator,
        solver_settings["algorithm"],
        newton_tol=solver_settings["newton_tolerance"],
        newton_max_iters=solver_settings["max_newton_iters"],
        linear_tol=solver_settings["krylov_tolerance"],
        linear_max_iters=solver_settings["max_linear_iters"],
        linear_correction_type=solver_settings["correction_type"],
        preconditioner_order=solver_settings["preconditioner_order"],
        tableau=tableau,
        instrument=True,
        newton_damping=solver_settings["newton_damping"],
        newton_max_backtracks=solver_settings["newton_max_backtracks"],
    )

    result = stepper(
        state=state,
        params=params,
        dt=solver_settings["dt"],
        time=0.0,
    )
    if not isinstance(result, InstrumentedStepResult):
        raise TypeError(
            "Expected InstrumentedStepResult from CPU reference step."
        )
    stage_count = int(result.stage_count)
    max_newton_iters = int(solver_settings["max_newton_iters"])
    state_dim = int(result.state.shape[0]) if result.state.ndim == 1 else 0
    shape = (stage_count, max_newton_iters, state_dim)
    zeros = np.zeros(shape, dtype=result.state.dtype)
    vector_defaults = {
        "solver_iteration_guesses": zeros,
        "solver_residuals": zeros,
        "solver_residual_norms": zeros,
        "solver_operator_outputs": zeros,
        "solver_preconditioned_vectors": zeros,
        "solver_iteration_end_x": zeros,
        "solver_iteration_end_rhs": zeros,
    }
    for name, default in vector_defaults.items():
        result.extra_vectors.setdefault(name, default)
    scale_shape = (stage_count, max_newton_iters)
    scale_zeros = np.zeros(scale_shape, dtype=result.state.dtype)
    result.extra_vectors.setdefault("solver_iteration_scale", scale_zeros)
    return result


def _format_array(values: np.ndarray) -> str:
    """Return ``values`` formatted for console comparison output."""

    return np.array2string(
        values,
        precision=8,
        floatmode="fixed",
        suppress_small=False,
        max_line_width=79,
    )


def _numeric_delta(
    cpu_values: np.ndarray,
    gpu_values: np.ndarray,
) -> np.ndarray:
    """Return the difference ``gpu_values - cpu_values`` in GPU precision."""

    gpu_array = np.asarray(gpu_values)
    cpu_array = np.asarray(cpu_values)
    if gpu_array.dtype != cpu_array.dtype:
        cpu_array = cpu_array.astype(gpu_array.dtype, copy=False)
    return gpu_array - cpu_array


def _print_section(
    name: str,
    cpu_values: np.ndarray,
    gpu_values: np.ndarray,
    device_values: Optional[np.ndarray],
) -> None:
    """Print CPU, GPU, and delta arrays for ``name``."""

    delta = _numeric_delta(cpu_values, gpu_values)
    print(f"{name} cpu:\n{_format_array(cpu_values)}")
    print(f"{name} gpu:\n{_format_array(gpu_values)}")
    if device_values is not None:
        print(f"{name} device:\n{_format_array(device_values)}")
    print(f"{name} delta:\n{_format_array(delta)}")
    print("")


def print_comparison(
    cpu_result: InstrumentedStepResult,
    gpu_result: DeviceInstrumentedResult,
    device_result,
) -> None:
    """Print side-by-side comparisons for CPU and CUDA instrumented outputs."""

    def _device_array(name: str) -> Optional[np.ndarray]:
        value = getattr(device_result, name, None)
        if value is None:
            return None
        return np.asarray(value)

    comparisons = [
        (
            "state",
            cpu_result.state,
            gpu_result.state,
            _device_array("state"),
        ),
        (
            "observables",
            cpu_result.observables,
            gpu_result.observables,
            _device_array("observables"),
        ),
        (
            "error",
            cpu_result.error,
            gpu_result.error,
            _device_array("error"),
        ),
        (
            "residuals",
            cpu_result.residuals,
            gpu_result.residuals,
            None,
        ),
        (
            "jacobian_updates",
            cpu_result.jacobian_updates,
            gpu_result.jacobian_updates,
            None,
        ),
        (
            "stage_states",
            cpu_result.stage_states,
            gpu_result.stage_states,
            None,
        ),
        (
            "stage_derivatives",
            cpu_result.stage_derivatives,
            gpu_result.stage_derivatives,
            None,
        ),
        (
            "stage_observables",
            cpu_result.stage_observables,
            gpu_result.stage_observables,
            None,
        ),
        (
            "stage_drivers",
            cpu_result.stage_drivers,
            gpu_result.stage_drivers,
            None,
        ),
        (
            "stage_increments",
            cpu_result.stage_increments,
            gpu_result.stage_increments,
            None,
        ),
        (
            "solver_initial_guesses",
            cpu_result.solver_initial_guesses,
            gpu_result.solver_initial_guesses,
            None,
        ),
        (
            "solver_solutions",
            cpu_result.solver_solutions,
            gpu_result.solver_solutions,
            None,
        ),
        (
            "solver_iterations",
            cpu_result.solver_iterations,
            gpu_result.solver_iterations,
            None,
        ),
        (
            "solver_status",
            cpu_result.solver_status,
            gpu_result.solver_status,
            None,
        ),
    ]

    optional_pairs = (
        (
            "newton_initial_guesses",
            cpu_result.newton_initial_guesses,
            gpu_result.newton_initial_guesses,
        ),
        (
            "newton_iteration_guesses",
            cpu_result.newton_iteration_guesses,
            gpu_result.newton_iteration_guesses,
        ),
        (
            "newton_residuals",
            cpu_result.newton_residuals,
            gpu_result.newton_residuals,
        ),
        (
            "newton_squared_norms",
            cpu_result.newton_squared_norms,
            gpu_result.newton_squared_norms,
        ),
        (
            "linear_initial_guesses",
            cpu_result.linear_initial_guesses,
            gpu_result.linear_initial_guesses,
        ),
        (
            "linear_iteration_guesses",
            cpu_result.linear_iteration_guesses,
            gpu_result.linear_iteration_guesses,
        ),
        (
            "linear_residuals",
            cpu_result.linear_residuals,
            gpu_result.linear_residuals,
        ),
        (
            "linear_squared_norms",
            cpu_result.linear_squared_norms,
            gpu_result.linear_squared_norms,
        ),
        (
            "linear_preconditioned_vectors",
            cpu_result.linear_preconditioned_vectors,
            gpu_result.linear_preconditioned_vectors,
        ),
    )

    for name, cpu_values, gpu_values in optional_pairs:
        if cpu_values is None or gpu_values is None:
            continue
        comparisons.append((name, cpu_values, gpu_values, None))

    for name, cpu_values, gpu_values, device_values in comparisons:
        device_array = (
            np.asarray(device_values)
            if device_values is not None
            else None
        )
        _print_section(
            name,
            np.asarray(cpu_values),
            np.asarray(gpu_values),
            device_array,
        )

    extra_names = sorted(
        set(cpu_result.extra_vectors.keys())
        | set(gpu_result.extra_vectors.keys())
    )
    for name in extra_names:
        cpu_values = cpu_result.extra_vectors.get(name)
        gpu_values = gpu_result.extra_vectors.get(name)
        if cpu_values is None and gpu_values is None:
            continue
        if cpu_values is None and gpu_values is not None:
            cpu_values = np.zeros_like(gpu_values)
        if gpu_values is None and cpu_values is not None:
            gpu_values = np.zeros_like(cpu_values)
        if cpu_values is None or gpu_values is None:
            continue
        _print_section(
            f"extra[{name}]",
            np.asarray(cpu_values),
            np.asarray(gpu_values),
            None,
        )

    status_delta = gpu_result.status - cpu_result.status
    niters_delta = gpu_result.niters - cpu_result.niters
    print(
        "status cpu={:d} gpu={:d} delta={:d}".format(
            int(cpu_result.status),
            int(gpu_result.status),
            int(status_delta),
        )
    )
    print(
        "niters cpu={:d} gpu={:d} delta={:d}".format(
            int(cpu_result.niters),
            int(gpu_result.niters),
            int(niters_delta),
        )
    )
    print(
        "stage_count cpu={:d} gpu={:d}".format(
            int(cpu_result.stage_count),
            int(gpu_result.residuals.shape[0]),
        )
    )
