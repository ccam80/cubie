import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pytest
from numba import cuda, from_dtype, int16

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
from tests.integrators.algorithms.instrumented import (
    create_instrumentation_host_buffers,
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

INSTRUMENTATION_DEVICE_FIELDS = (
    "residuals",
    "jacobian_updates",
    "stage_states",
    "stage_derivatives",
    "stage_observables",
    "stage_drivers",
    "stage_increments",
    "newton_initial_guesses",
    "newton_iteration_guesses",
    "newton_residuals",
    "newton_squared_norms",
    "newton_iteration_scale",
    "linear_initial_guesses",
    "linear_iteration_guesses",
    "linear_residuals",
    "linear_squared_norms",
    "linear_preconditioned_vectors",
)

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
    newton_initial_guesses: Optional[np.ndarray] = None
    newton_iteration_guesses: Optional[np.ndarray] = None
    newton_residuals: Optional[np.ndarray] = None
    newton_squared_norms: Optional[np.ndarray] = None
    newton_iteration_scale: Optional[np.ndarray] = None
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
        newton_initial_guesses,
        newton_iteration_guesses,
        newton_residuals,
        newton_squared_norms,
        newton_iteration_scale,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
        first_step_flag,
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
        first_step_flag = int16(1)
        accepted_flag = int16(1)
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
            newton_initial_guesses,
            newton_iteration_guesses,
            newton_residuals,
            newton_squared_norms,
            newton_iteration_scale,
            linear_initial_guesses,
            linear_iteration_guesses,
            linear_residuals,
            linear_squared_norms,
            linear_preconditioned_vectors,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
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


def _copy_device_instrumentation(
    d_proposed,
    d_proposed_observables,
    d_error,
    device_buffers,
    host_buffers,
    d_status,
) -> DeviceInstrumentedResult:
    """Return ``DeviceInstrumentedResult`` populated from device buffers."""

    status_value = int(d_status.copy_to_host()[0])
    host_results = {}
    for name, device_array in device_buffers.items():
        host_array = getattr(host_buffers, name)
        host_results[name] = device_array.copy_to_host(host_array)

    newton_initial_guesses_host = host_results["newton_initial_guesses"]
    newton_iteration_guesses_host = host_results["newton_iteration_guesses"]
    newton_residuals_host = host_results["newton_residuals"]
    newton_squared_norms_host = host_results["newton_squared_norms"]
    newton_iteration_scale_host = host_results["newton_iteration_scale"]
    linear_initial_guesses_host = host_results["linear_initial_guesses"]
    linear_iteration_guesses_host = host_results["linear_iteration_guesses"]
    linear_residuals_host = host_results["linear_residuals"]
    linear_squared_norms_host = host_results["linear_squared_norms"]
    linear_preconditioned_vectors_host = host_results[
        "linear_preconditioned_vectors"
    ]
    return DeviceInstrumentedResult(
        state=d_proposed.copy_to_host(),
        observables=d_proposed_observables.copy_to_host(),
        error=d_error.copy_to_host(),
        residuals=host_results["residuals"].copy(),
        jacobian_updates=host_results["jacobian_updates"].copy(),
        stage_states=host_results["stage_states"].copy(),
        stage_derivatives=host_results["stage_derivatives"].copy(),
        stage_observables=host_results["stage_observables"].copy(),
        stage_drivers=host_results["stage_drivers"].copy(),
        stage_increments=host_results["stage_increments"].copy(),
        newton_initial_guesses=newton_initial_guesses_host,
        newton_iteration_guesses=newton_iteration_guesses_host,
        newton_residuals=newton_residuals_host,
        newton_squared_norms=newton_squared_norms_host,
        newton_iteration_scale=newton_iteration_scale_host,
        linear_initial_guesses=linear_initial_guesses_host,
        linear_iteration_guesses=linear_iteration_guesses_host,
        linear_residuals=linear_residuals_host,
        linear_squared_norms=linear_squared_norms_host,
        linear_preconditioned_vectors=linear_preconditioned_vectors_host,
        status=status_value & STATUS_MASK,
        niters=(status_value >> 16) & STATUS_MASK,
        extra_vectors={},
    )


@pytest.fixture(scope="session")
def instrumented_step_results(
    instrumented_step_kernel: InstrumentedKernel,
    instrumented_step_object,
    step_inputs,
    solver_settings,
    system,
    precision,
) -> List[DeviceInstrumentedResult]:
    """Execute the instrumented CUDA step twice and collect host arrays."""

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
    max_newton_backtracks = int(solver_settings["newton_max_backtracks"])
    linear_max_iters = int(solver_settings["max_linear_iters"])

    params = np.asarray(step_inputs["parameters"], dtype=precision)
    driver_coefficients = np.asarray(
        step_inputs["driver_coefficients"], dtype=precision
    )
    drivers = np.asarray(step_inputs["drivers"], dtype=precision)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    dt_value = precision(solver_settings["dt"])

    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)

    def _run_step(
        current_state,
        current_drivers,
        current_observables,
        time_value,
    ):
        host_buffers = create_instrumentation_host_buffers(
            precision=precision,
            stage_count=stage_count,
            state_size=n_states,
            observable_size=n_observables,
            driver_size=current_drivers.shape[0],
            newton_max_iters=max_newton_iters,
            newton_max_backtracks=max_newton_backtracks,
            linear_max_iters=linear_max_iters,
        )
        status = np.zeros(1, dtype=np.int32)

        d_state = cuda.to_device(current_state)
        d_proposed = cuda.to_device(np.zeros_like(current_state))
        d_drivers = cuda.to_device(current_drivers)
        d_proposed_drivers = cuda.to_device(np.zeros_like(current_drivers))
        d_observables = cuda.to_device(current_observables)
        d_proposed_observables = cuda.to_device(
            np.zeros_like(current_observables)
        )
        d_error = cuda.to_device(np.zeros(n_states, dtype=precision))
        device_buffers = {
            name: cuda.to_device(getattr(host_buffers, name))
            for name in INSTRUMENTATION_DEVICE_FIELDS
        }
        d_status = cuda.to_device(status)

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
            device_buffers["residuals"],
            device_buffers["jacobian_updates"],
            device_buffers["stage_states"],
            device_buffers["stage_derivatives"],
            device_buffers["stage_observables"],
            device_buffers["stage_drivers"],
            device_buffers["stage_increments"],
            device_buffers["newton_initial_guesses"],
            device_buffers["newton_iteration_guesses"],
            device_buffers["newton_residuals"],
            device_buffers["newton_squared_norms"],
            device_buffers["newton_iteration_scale"],
            device_buffers["linear_initial_guesses"],
            device_buffers["linear_iteration_guesses"],
            device_buffers["linear_residuals"],
            device_buffers["linear_squared_norms"],
            device_buffers["linear_preconditioned_vectors"],
            dt_value,
            numba_precision(time_value),
            d_status,
        )
        cuda.synchronize()

        result = _copy_device_instrumentation(
            d_proposed,
            d_proposed_observables,
            d_error,
            device_buffers,
            host_buffers,
            d_status,
        )
        next_state = np.array(result.state, dtype=precision, copy=True)
        next_drivers = d_proposed_drivers.copy_to_host()
        next_observables = np.array(
            result.observables, dtype=precision, copy=True
        )
        return result, next_state, next_drivers, next_observables

    initial_state = np.asarray(step_inputs["state"], dtype=precision)
    first_result, first_state, first_drivers, first_observables = _run_step(
        initial_state,
        drivers,
        observables,
        precision(0.0),
    )
    second_result, _, _, _ = _run_step(
        first_state,
        first_drivers,
        first_observables,
        dt_value,
    )

    return [first_result, second_result]


@pytest.fixture(scope="session")
def instrumented_cpu_step_results(
    solver_settings,
    cpu_system: CPUODESystem,
    step_inputs,
    cpu_driver_evaluator,
    instrumented_step_object,
    instrumented_step_configuration: ResolvedInstrumentedStep,
) -> List[InstrumentedStepResult]:
    """Execute the CPU reference step with instrumentation enabled twice."""

    def _copy_array(values):
        if values is None:
            return None
        return np.asarray(values, dtype=cpu_system.precision).copy()

    def _copy_result(result):
        extras = {
            name: np.asarray(values, dtype=cpu_system.precision).copy()
            for name, values in result.extra_vectors.items()
        }
        return InstrumentedStepResult(
            state=np.asarray(result.state, dtype=cpu_system.precision).copy(),
            observables=np.asarray(
                result.observables, dtype=cpu_system.precision
            ).copy(),
            error=np.asarray(result.error, dtype=cpu_system.precision).copy(),
            residuals=_copy_array(result.residuals),
            jacobian_updates=_copy_array(result.jacobian_updates),
            stage_drivers=_copy_array(result.stage_drivers),
            stage_increments=_copy_array(result.stage_increments),
            status=int(result.status),
            niters=int(result.niters),
            stage_count=int(result.stage_count),
            stage_states=_copy_array(result.stage_states),
            stage_derivatives=_copy_array(result.stage_derivatives),
            stage_observables=_copy_array(result.stage_observables),
            newton_initial_guesses=_copy_array(result.newton_initial_guesses),
            newton_iteration_guesses=_copy_array(
                result.newton_iteration_guesses
            ),
            newton_residuals=_copy_array(result.newton_residuals),
            newton_squared_norms=_copy_array(result.newton_squared_norms),
            newton_iteration_scale=_copy_array(result.newton_iteration_scale),
            linear_initial_guesses=_copy_array(result.linear_initial_guesses),
            linear_iteration_guesses=_copy_array(
                result.linear_iteration_guesses
            ),
            linear_residuals=_copy_array(result.linear_residuals),
            linear_squared_norms=_copy_array(result.linear_squared_norms),
            linear_preconditioned_vectors=_copy_array(
                result.linear_preconditioned_vectors
            ),
            extra_vectors=extras,
        )

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

    first_result = stepper(
        state=state,
        params=params,
        dt=solver_settings["dt"],
        time=0.0,
    )
    if not isinstance(first_result, InstrumentedStepResult):
        raise TypeError(
            "Expected InstrumentedStepResult from CPU reference step."
        )

    second_state = np.asarray(
        first_result.state, dtype=cpu_system.precision
    ).copy()
    second_result = stepper(
        state=second_state,
        params=params,
        dt=solver_settings["dt"],
        time=solver_settings["dt"],
    )
    if not isinstance(second_result, InstrumentedStepResult):
        raise TypeError(
            "Expected InstrumentedStepResult from CPU reference step."
        )

    return [_copy_result(first_result), _copy_result(second_result)]


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


def _print_grouped_section(
    name: str,
    cpu_series: List[Optional[np.ndarray]],
    gpu_series: List[Optional[np.ndarray]],
    device_series: List[Optional[np.ndarray]],
    delta_series: List[Optional[np.ndarray]],
) -> None:
    """Print grouped CPU, GPU, device, and delta arrays for ``name``."""

    if all(values is None for values in cpu_series):
        return

    print(f"{name} cpu:")
    for values in cpu_series:
        if values is None:
            continue
        print(_format_array(values))
    print(f"{name} gpu:")
    for values in gpu_series:
        if values is None:
            continue
        print(_format_array(values))
    if any(values is not None for values in device_series):
        print(f"{name} device:")
        for values in device_series:
            if values is None:
                continue
            print(_format_array(values))
    print(f"{name} delta:")
    for values in delta_series:
        if values is None:
            continue
        print(_format_array(values))
    print("")


def print_comparison(cpu_result, gpu_result, device_result) -> None:
    """Print comparisons for CPU and CUDA instrumented outputs."""

    def _as_sequence(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    cpu_results = _as_sequence(cpu_result)
    gpu_results = _as_sequence(gpu_result)
    device_results = _as_sequence(device_result)

    if len(cpu_results) != len(gpu_results):
        raise ValueError("CPU and GPU result counts must match.")

    if len(device_results) < len(cpu_results):
        pad_count = len(cpu_results) - len(device_results)
        device_results.extend([None] * pad_count)

    step_count = len(cpu_results)
    comparison_order: List[str] = []
    comparisons: Dict[str, Dict[str, List[Optional[np.ndarray]]]] = {}

    def _ensure_entry(name: str) -> Dict[str, List[Optional[np.ndarray]]]:
        if name not in comparisons:
            comparisons[name] = {
                "cpu": [None] * step_count,
                "gpu": [None] * step_count,
                "device": [None] * step_count,
                "delta": [None] * step_count,
            }
            comparison_order.append(name)
        return comparisons[name]

    def _add_entry(
        name: str,
        cpu_values: Union[np.ndarray, int, float],
        gpu_values: Union[np.ndarray, int, float],
        device_values: Optional[Union[np.ndarray, int, float]],
        step_index: int,
    ) -> None:
        entry = _ensure_entry(name)
        cpu_array = np.asarray(cpu_values)
        gpu_array = np.asarray(gpu_values)
        entry["cpu"][step_index] = cpu_array
        entry["gpu"][step_index] = gpu_array
        entry["delta"][step_index] = _numeric_delta(cpu_array, gpu_array)
        if device_values is not None:
            entry["device"][step_index] = np.asarray(device_values)

    for index, (cpu_step, gpu_step, device_step) in enumerate(
        zip(cpu_results, gpu_results, device_results)
    ):

        def _device_array(name: str) -> Optional[np.ndarray]:
            if device_step is None:
                return None
            value = getattr(device_step, name, None)
            if value is None:
                return None
            return np.asarray(value)

        _add_entry(
            "state",
            cpu_step.state,
            gpu_step.state,
            _device_array("state"),
            index,
        )
        _add_entry(
            "observables",
            cpu_step.observables,
            gpu_step.observables,
            _device_array("observables"),
            index,
        )
        _add_entry(
            "error",
            cpu_step.error,
            gpu_step.error,
            _device_array("error"),
            index,
        )
        _add_entry(
            "residuals",
            cpu_step.residuals,
            gpu_step.residuals,
            None,
            index,
        )
        _add_entry(
            "jacobian_updates",
            cpu_step.jacobian_updates,
            gpu_step.jacobian_updates,
            None,
            index,
        )
        _add_entry(
            "stage_states",
            cpu_step.stage_states,
            gpu_step.stage_states,
            None,
            index,
        )
        _add_entry(
            "stage_derivatives",
            cpu_step.stage_derivatives,
            gpu_step.stage_derivatives,
            None,
            index,
        )
        _add_entry(
            "stage_observables",
            cpu_step.stage_observables,
            gpu_step.stage_observables,
            None,
            index,
        )
        _add_entry(
            "stage_drivers",
            cpu_step.stage_drivers,
            gpu_step.stage_drivers,
            None,
            index,
        )
        _add_entry(
            "stage_increments",
            cpu_step.stage_increments,
            gpu_step.stage_increments,
            None,
            index,
        )

        optional_pairs = (
            (
                "newton_initial_guesses",
                cpu_step.newton_initial_guesses,
                gpu_step.newton_initial_guesses,
            ),
            (
                "newton_iteration_guesses",
                cpu_step.newton_iteration_guesses,
                gpu_step.newton_iteration_guesses,
            ),
            (
                "newton_residuals",
                cpu_step.newton_residuals,
                gpu_step.newton_residuals,
            ),
            (
                "newton_squared_norms",
                cpu_step.newton_squared_norms,
                gpu_step.newton_squared_norms,
            ),
            (
                "newton_iteration_scale",
                cpu_step.newton_iteration_scale,
                gpu_step.newton_iteration_scale,
            ),
            (
                "linear_initial_guesses",
                cpu_step.linear_initial_guesses,
                gpu_step.linear_initial_guesses,
            ),
            (
                "linear_iteration_guesses",
                cpu_step.linear_iteration_guesses,
                gpu_step.linear_iteration_guesses,
            ),
            (
                "linear_residuals",
                cpu_step.linear_residuals,
                gpu_step.linear_residuals,
            ),
            (
                "linear_squared_norms",
                cpu_step.linear_squared_norms,
                gpu_step.linear_squared_norms,
            ),
            (
                "linear_preconditioned_vectors",
                cpu_step.linear_preconditioned_vectors,
                gpu_step.linear_preconditioned_vectors,
            ),
        )

        for name, cpu_values, gpu_values in optional_pairs:
            if cpu_values is None or gpu_values is None:
                continue
            _add_entry(name, cpu_values, gpu_values, None, index)

        extra_names = sorted(
            set(cpu_step.extra_vectors.keys())
            | set(gpu_step.extra_vectors.keys())
        )
        for name in extra_names:
            cpu_values = cpu_step.extra_vectors.get(name)
            gpu_values = gpu_step.extra_vectors.get(name)
            if cpu_values is None and gpu_values is None:
                continue
            if cpu_values is None and gpu_values is not None:
                cpu_values = np.zeros_like(gpu_values)
            if gpu_values is None and cpu_values is not None:
                gpu_values = np.zeros_like(cpu_values)
            if cpu_values is None or gpu_values is None:
                continue
            _add_entry(f"extra[{name}]", cpu_values, gpu_values, None, index)

        _add_entry("status", cpu_step.status, gpu_step.status, None, index)
        _add_entry("niters", cpu_step.niters, gpu_step.niters, None, index)
        _add_entry(
            "stage_count",
            cpu_step.stage_count,
            gpu_step.residuals.shape[0],
            None,
            index,
        )

    for name in comparison_order:
        entry = comparisons[name]
        _print_grouped_section(
            name,
            entry["cpu"],
            entry["gpu"],
            entry["device"],
            entry["delta"],
        )
