import attrs
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pytest
from numba import cuda, from_dtype, int32
from numba.core.types import int32

from tests._utils import _build_enhanced_algorithm_settings
from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    ButcherTableau,
    CrankNicolsonStep,
    DIRKStep,
    ERKStep,
    ExplicitEulerStep,
    FIRKStep,
    GenericRosenbrockWStep,
    resolve_alias,
    resolve_supplied_tableau,
    BaseAlgorithmStep,
)
from cubie._utils import split_applicable_settings

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
from .generic_firk import FIRKStep as InstrumentedFIRKStep
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

STEP_CONSTRUCTOR_TO_INSTRUMENTED: Dict[type, BaseAlgorithmStep] = {
    ExplicitEulerStep: InstrumentedExplicitEulerStep,
    BackwardsEulerStep: InstrumentedBackwardsEulerStep,
    BackwardsEulerPCStep: InstrumentedBackwardsEulerPCStep,
    CrankNicolsonStep: InstrumentedCrankNicolsonStep,
    ERKStep: InstrumentedERKStep,
    DIRKStep: InstrumentedDIRKStep,
    FIRKStep: InstrumentedFIRKStep,
    GenericRosenbrockWStep: InstrumentedRosenbrockStep,
}


def get_instrumented_step(
    precision: type,
    settings: Optional[Dict[str, object]] = None,
    warn_on_unused: bool = True,
    **kwargs: Any,
) -> BaseAlgorithmStep:
    """Factory mirroring ``get_algorithm_step`` but returning instrumented step.

    This is a near-direct copy of the public ``get_algorithm_step`` logic with a
    single change: after resolving the algorithm constructor we map it to the
    instrumented implementation using ``STEP_CONSTRUCTOR_TO_INSTRUMENTED`` and
    then filter and instantiate that class.
    """

    algorithm_settings: Dict[str, object] = {}
    if settings is not None:
        algorithm_settings.update(settings)
    algorithm_settings.update(kwargs)

    algorithm_value = algorithm_settings.pop("algorithm", None)
    if algorithm_value is None:
        raise ValueError("Algorithm settings must include 'algorithm'.")

    if isinstance(algorithm_value, str):
        try:
            algorithm_type, resolved_tableau = resolve_alias(algorithm_value)
        except KeyError as exc:
            raise ValueError(f"Unknown algorithm '{algorithm_value}'.") from exc
    elif isinstance(algorithm_value, ButcherTableau):
        algorithm_type, resolved_tableau = resolve_supplied_tableau(
            algorithm_value
        )
    else:
        raise TypeError(
            "Expected algorithm name or ButcherTableau instance, "
            f"received {type(algorithm_value).__name__}."
        )

    algorithm_settings["precision"] = precision

    try:
        step_class = STEP_CONSTRUCTOR_TO_INSTRUMENTED[algorithm_type]
    except KeyError as error:
        raise ValueError(
            f"No instrumented implementation registered for {algorithm_type}."
        ) from error

    filtered, missing, unused = split_applicable_settings(
        step_class,
        algorithm_settings,
        warn_on_unused=warn_on_unused,
    )
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"{step_class.__name__} requires settings for: {missing_keys}"
        )

    if resolved_tableau is not None:
        filtered["tableau"] = resolved_tableau

    return step_class(**filtered)


@pytest.fixture(scope="session")
def num_steps() -> int:
    """Number of consecutive step executions for instrumentation."""

    return 6

@pytest.fixture(scope="session")
def dts(num_steps, solver_settings, precision, instrumented_step_object) -> Tuple[int, ...]:
    """random time."""
    forty_randoms = (solver_settings['dt'], 1e-3, 1e-4, 1e-5, 1e-2, 1e-2,
                     1e-3,
                     5e-3, 5e-4,
                     5e-5, 1e-2, 1e-3, 1e-7, 1e-6, 1e-7, 1e-2, 1e-3, 5e-3,
                     5e-4, 5e-5, 1e-2, 1e-3, 1e-4, 1e-5, 1e-2, 1e-2, 1e-3, 5e-3, 5e-4,
                     5e-5, 1e-2, 1e-3, 1e-4, 1e-5, 1e-2, 1e-2, 1e-3, 5e-3,
                     5e-4, 5e-5)
    forty_randoms = tuple(precision(step) for step in forty_randoms)
    haserror=True
    hastableau = getattr(instrumented_step_object, "tableau")
    if hastableau is not None:
        haserror = instrumented_step_object.tableau.has_error_estimate
    if (
        (
            solver_settings["algorithm"]
            in ["euler", "backwards_euler", "backwards_euler_pc"]
        )
        or not haserror
    ):
        dt_value = precision(solver_settings["dt"])
        dts = tuple(dt_value for _ in range(num_steps))
    else:
        dts = forty_randoms[:num_steps]
    return dts

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
    driver_array,
) -> dict:
    """State, parameters, and drivers for a single step execution."""
    width = system.num_drivers
    driver_coefficients = np.array(
        driver_array.coefficients, dtype=precision, copy=True
    )
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(precision),
        "drivers": np.zeros(width, dtype=precision),
        "driver_coefficients": driver_coefficients,
    }

@attrs.define
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
    stage_count: Optional[int] = None
    extra_vectors: Optional[Dict[str, np.ndarray]] = None

@pytest.fixture(scope="session")
def instrumented_step_object(
    system,
    algorithm_settings,
    driver_array,
    precision,
):
    """Instantiate the configured instrumented step implementation."""

    # Use the mirrored factory to instantiate the instrumented implementation.
    # We pass the whole solver_settings mapping so the factory can filter
    # arguments identically to the production path.
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
            algorithm_settings, system, driver_array
    )
    return get_instrumented_step(precision, enhanced_algorithm_settings)


@pytest.fixture(scope="session")
def instrumented_step_results(
    instrumented_step_object,
    step_inputs,
    solver_settings,
    system,
    precision,
    dts,
    num_steps,
    driver_array,
) -> List[DeviceInstrumentedResult]:
    """Execute the instrumented CUDA step loop and collect host arrays.

    Fetch shared memory and sizing values from instrumented_step_object.
    """

    # Bind step-specific functions/values in the same way as device_step_results
    step_function = instrumented_step_object.step_function
    numba_precision = from_dtype(precision)
    shared_elems = instrumented_step_object.shared_memory_required
    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, instrumented_step_object.persistent_local_required)
    driver_function = None if driver_array is None else driver_array.evaluation_function
    # use system-provided observables function
    observables_function = system.observables_function

    n_states = system.sizes.states
    n_observables = system.sizes.observables
    stage_count_attr = getattr(
        getattr(instrumented_step_object, "tableau", None),
        "stage_count",
        None,
    )
    if stage_count_attr is None:
        stage_count_attr = getattr(instrumented_step_object, "stage_count", 0)
    stage_count = int32(stage_count_attr or 1)
    max_newton_iters = int32(solver_settings["max_newton_iters"])
    max_newton_backtracks = int32(solver_settings["newton_max_backtracks"])
    linear_max_iters = int32(solver_settings["max_linear_iters"])

    params = np.asarray(step_inputs["parameters"], dtype=precision)
    driver_coefficients = np.asarray(
        step_inputs["driver_coefficients"], dtype=precision
    )
    drivers = np.asarray(step_inputs["drivers"], dtype=precision)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    d_dts = cuda.to_device(dts)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)

    # Define the CUDA kernel inside the results fixture so closures match device fixture
    @cuda.jit()
    def kernel(
        state_vec,
        proposed_matrix,
        params_vec,
        driver_coeffs_vec,
        drivers_vec,
        proposed_drivers_matrix,
        observables_vec,
        proposed_observables_matrix,
        error_matrix,
        residuals_tensor,
        jacobian_updates_tensor,
        stage_states_tensor,
        stage_derivatives_tensor,
        stage_observables_tensor,
        stage_drivers_tensor,
        stage_increments_tensor,
        newton_initial_guesses_tensor,
        newton_iteration_guesses_tensor,
        newton_residuals_tensor,
        newton_squared_norms_tensor,
        newton_iteration_scale_tensor,
        linear_initial_guesses_tensor,
        linear_iteration_guesses_tensor,
        linear_residuals_tensor,
        linear_squared_norms_tensor,
        linear_preconditioned_vectors_tensor,
        status_vec,
        dts_array,
        time_scalar,
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        counters = cuda.local.array(2, dtype=int32)
        zero = numba_precision(0.0)
        for cache_idx in range(shared_elems):
            shared[cache_idx] = zero
        current_time = time_scalar
        # Call driver/evaluator and observables similarly to device fixture
        if driver_function is not None:
            driver_function(current_time, driver_coeffs_vec, drivers_vec)
        observables_function(
            state_vec,
            params_vec,
            drivers_vec,
            observables_vec,
            current_time,
            )

        accepted_flag = int32(1)
        state_length = state_vec.shape[0]
        driver_length = drivers_vec.shape[0]
        observable_length = observables_vec.shape[0]
        for step_idx in range(num_steps):
            first_step_flag = int32(1) if step_idx == 0 else int32(0)
            # Read dt for this step from device array
            dt_scalar = numba_precision(dts_array[step_idx])
            result = step_function(
                state_vec,
                proposed_matrix[step_idx],
                params_vec,
                driver_coeffs_vec,
                drivers_vec,
                proposed_drivers_matrix[step_idx],
                observables_vec,
                proposed_observables_matrix[step_idx],
                error_matrix[step_idx],
                residuals_tensor[step_idx],
                jacobian_updates_tensor[step_idx],
                stage_states_tensor[step_idx],
                stage_derivatives_tensor[step_idx],
                stage_observables_tensor[step_idx],
                stage_drivers_tensor[step_idx],
                stage_increments_tensor[step_idx],
                newton_initial_guesses_tensor[step_idx],
                newton_iteration_guesses_tensor[step_idx],
                newton_residuals_tensor[step_idx],
                newton_squared_norms_tensor[step_idx],
                newton_iteration_scale_tensor[step_idx],
                linear_initial_guesses_tensor[step_idx],
                linear_iteration_guesses_tensor[step_idx],
                linear_residuals_tensor[step_idx],
                linear_squared_norms_tensor[step_idx],
                linear_preconditioned_vectors_tensor[step_idx],
                dt_scalar,
                current_time,
                first_step_flag,
                accepted_flag,
                shared,
                persistent,
                counters,
            )
            status_vec[step_idx] = result

            if step_idx == 5 or step_idx == 8:
                accepted_flag = int32(0)
            else:
                if step_idx + 1 >= num_steps:
                    continue
                for elem in range(state_length):
                    state_vec[elem] = proposed_matrix[step_idx, elem]
                for drv_idx in range(driver_length):
                    drivers_vec[drv_idx] = proposed_drivers_matrix[step_idx, drv_idx]
                for obs_idx in range(observable_length):
                    observables_vec[obs_idx] = proposed_observables_matrix[
                        step_idx, obs_idx
                    ]
                accepted_flag = int32(1)
                current_time = current_time + dt_scalar


    initial_state = np.asarray(step_inputs["state"], dtype=precision)

    # Check if this is a FIRK step which requires flattened solver buffers
    from tests.integrators.algorithms.instrumented.generic_firk import FIRKStep as InstrumentedFIRKStep
    is_firk = isinstance(instrumented_step_object, InstrumentedFIRKStep)

    host_buffers = create_instrumentation_host_buffers(
        precision=precision,
        stage_count=stage_count,
        state_size=n_states,
        observable_size=n_observables,
        driver_size=drivers.shape[0],
        newton_max_iters=max_newton_iters,
        newton_max_backtracks=max_newton_backtracks,
        linear_max_iters=linear_max_iters,
        num_steps=num_steps,
        flattened_solver=is_firk,
    )
    status = np.zeros(num_steps, dtype=np.int32)

    d_state = cuda.to_device(initial_state)
    d_proposed = cuda.to_device(
        np.zeros((num_steps, initial_state.shape[0]), dtype=precision)
    )
    d_drivers = cuda.to_device(drivers)
    d_proposed_drivers = cuda.to_device(
        np.zeros((num_steps, drivers.shape[0]), dtype=precision)
    )
    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(
        np.zeros((num_steps, observables.shape[0]), dtype=precision)
    )
    d_error = cuda.to_device(np.zeros((num_steps, n_states), dtype=precision))
    device_buffers = {
        name: cuda.to_device(getattr(host_buffers, name))
        for name in INSTRUMENTATION_DEVICE_FIELDS
    }
    d_status = cuda.to_device(status)

    # Unpack device buffers into local variables to avoid dict-closure issues
    # when passing them to the kernel and to mirror the device fixture's
    # ordering/scope exactly.
    d_residuals = device_buffers["residuals"]
    d_jacobian_updates = device_buffers["jacobian_updates"]
    d_stage_states = device_buffers["stage_states"]
    d_stage_derivatives = device_buffers["stage_derivatives"]
    d_stage_observables = device_buffers["stage_observables"]
    d_stage_drivers = device_buffers["stage_drivers"]
    d_stage_increments = device_buffers["stage_increments"]
    d_newton_initial_guesses = device_buffers["newton_initial_guesses"]
    d_newton_iteration_guesses = device_buffers["newton_iteration_guesses"]
    d_newton_residuals = device_buffers["newton_residuals"]
    d_newton_squared_norms = device_buffers["newton_squared_norms"]
    d_newton_iteration_scale = device_buffers["newton_iteration_scale"]
    d_linear_initial_guesses = device_buffers["linear_initial_guesses"]
    d_linear_iteration_guesses = device_buffers["linear_iteration_guesses"]
    d_linear_residuals = device_buffers["linear_residuals"]
    d_linear_squared_norms = device_buffers["linear_squared_norms"]
    d_linear_preconditioned_vectors = device_buffers["linear_preconditioned_vectors"]

    # Launch kernel (single block/grid as before) and synchronize
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
        d_newton_initial_guesses,
        d_newton_iteration_guesses,
        d_newton_residuals,
        d_newton_squared_norms,
        d_newton_iteration_scale,
        d_linear_initial_guesses,
        d_linear_iteration_guesses,
        d_linear_residuals,
        d_linear_squared_norms,
        d_linear_preconditioned_vectors,
        d_status,
        d_dts,
        numba_precision(0.0),
    )
    cuda.synchronize()

    results = _copy_device_instrumentation(
        d_proposed,
        d_proposed_observables,
        d_error,
        device_buffers,
        host_buffers,
        d_status,
        num_steps=num_steps,
    )

    return results


def _copy_device_instrumentation(
    d_proposed,
    d_proposed_observables,
    d_error,
    device_buffers,
    host_buffers,
    d_status,
    *,
    num_steps: int,
) -> List[DeviceInstrumentedResult]:
    """Return ``DeviceInstrumentedResult`` items copied from device buffers."""

    status_values = d_status.copy_to_host()
    host_results: Dict[str, np.ndarray] = {}
    for name, device_array in device_buffers.items():
        host_array = getattr(host_buffers, name)
        host_results[name] = device_array.copy_to_host(host_array)

    proposed_states = d_proposed.copy_to_host()
    proposed_observables = d_proposed_observables.copy_to_host()
    error_matrix = d_error.copy_to_host()

    results: List[DeviceInstrumentedResult] = []
    for step_idx in range(num_steps):
        status_value = int(status_values[step_idx])
        result = DeviceInstrumentedResult(
            state=np.asarray(proposed_states[step_idx]).copy(),
            observables=np.asarray(proposed_observables[step_idx]).copy(),
            error=np.asarray(error_matrix[step_idx]).copy(),
            residuals=np.asarray(host_results["residuals"][step_idx]).copy(),
            jacobian_updates=np.asarray(
                host_results["jacobian_updates"][step_idx]
            ).copy(),
            stage_states=np.asarray(host_results["stage_states"][step_idx]).copy(),
            stage_derivatives=np.asarray(
                host_results["stage_derivatives"][step_idx]
            ).copy(),
            stage_observables=np.asarray(
                host_results["stage_observables"][step_idx]
            ).copy(),
            stage_drivers=np.asarray(
                host_results["stage_drivers"][step_idx]
            ).copy(),
            stage_increments=np.asarray(
                host_results["stage_increments"][step_idx]
            ).copy(),
            newton_initial_guesses=np.asarray(
                host_results["newton_initial_guesses"][step_idx]
            ).copy(),
            newton_iteration_guesses=np.asarray(
                host_results["newton_iteration_guesses"][step_idx]
            ).copy(),
            newton_residuals=np.asarray(
                host_results["newton_residuals"][step_idx]
            ).copy(),
            newton_squared_norms=np.asarray(
                host_results["newton_squared_norms"][step_idx]
            ).copy(),
            newton_iteration_scale=np.asarray(
                host_results["newton_iteration_scale"][step_idx]
            ).copy(),
            linear_initial_guesses=np.asarray(
                host_results["linear_initial_guesses"][step_idx]
            ).copy(),
            linear_iteration_guesses=np.asarray(
                host_results["linear_iteration_guesses"][step_idx]
            ).copy(),
            linear_residuals=np.asarray(
                host_results["linear_residuals"][step_idx]
            ).copy(),
            linear_squared_norms=np.asarray(
                host_results["linear_squared_norms"][step_idx]
            ).copy(),
            linear_preconditioned_vectors=np.asarray(
                host_results["linear_preconditioned_vectors"][step_idx]
            ).copy(),
            status=status_value & STATUS_MASK,
            niters=(status_value >> 16) & STATUS_MASK,
            stage_count=int(host_buffers.stage_count),
            extra_vectors={},
        )
        results.append(result)

    return results


@pytest.fixture(scope="session")
def instrumented_cpu_step_results(
    solver_settings,
    cpu_system: CPUODESystem,
    step_inputs,
    cpu_driver_evaluator,
    instrumented_step_object,
    num_steps,
    dts
) -> List[InstrumentedStepResult]:
    """Execute the CPU reference step with instrumentation enabled."""

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

    # Use the tableau resolved for the instrumented step object
    tableau = getattr(instrumented_step_object, "tableau", None)
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




    current_state = state.copy()
    time_value = cpu_system.precision(0.0)
    results: List[InstrumentedStepResult] = []
    for step_idx in range(num_steps):
        result = stepper.step(
            state=current_state,
            params=params,
            dt=dts[step_idx],
            time=time_value,
        )
        if not isinstance(result, InstrumentedStepResult):
            raise TypeError(
                "Expected InstrumentedStepResult from CPU reference step."
            )
        results.append(_copy_result(result))
        if step_idx != 5 and step_idx != 8:  # simulate rejections at two
            # arbitrary points
            current_state = np.asarray(
                result.state, dtype=cpu_system.precision
            ).copy()
            time_value = cpu_system.precision(time_value + dts[step_idx])

    return results


def _format_array(values: np.ndarray) -> str:
    """Return ``values`` formatted for console comparison output."""
    if not np.any(np.asarray(values)):
        return "0.0"
    return np.array2string(
        values,
        precision=8,
        floatmode="fixed",
        suppress_small=False,
        max_line_width=79,
    )


def _print_trimmed_array(values: np.ndarray) -> None:
    """Print ``values`` while skipping zero-only slices on outer axes."""

    array = np.asarray(values)
    if not np.any(array):
        print("0.0")
        return

    _print_nonzero_subarrays(array, ())


def _print_nonzero_subarrays(
    array: np.ndarray,
    indices: Tuple[int, ...],
) -> None:
    """Recursively print subarrays limited to nonzero outer indices."""

    if array.ndim <= 1:
        prefix = "".join(f"[{idx}]" for idx in indices)
        if prefix:
            print(f"{prefix} {_format_array(array)}")
        else:
            print(_format_array(array))
        return

    for idx in range(array.shape[0]):
        subarray = array[idx]
        if np.any(subarray):
            _print_nonzero_subarrays(subarray, indices + (idx,))


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

def _is_none_or_all_zero(value):
    if value is None:
        return True
    arr = np.asarray(value)
    # treat arrays with no non-zero entries as "zero"
    return not np.any(arr)

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

    if all(_is_none_or_all_zero(v) for v in gpu_series):
        print(f"{name} delta:")
        print("0.0")
        return

    print(f"{name} delta:")

    for values in delta_series:
        if values is None:
            continue
        _print_trimmed_array(values)
    if any(values is not None for values in device_series):
        print(f"{name} device:")
        for values in device_series:
            if values is None:
                continue
            _print_trimmed_array(values)
    print(f"{name} gpu:")
    for values in gpu_series:
        if values is None:
            continue
        _print_trimmed_array(values)

    print(f"{name} cpu:")
    for values in cpu_series:
        if values is None:
            continue
        _print_trimmed_array(values)
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
                "delta": [None] * step_count,
                "device": [None] * step_count,
                "gpu": [None] * step_count,
                "cpu": [None] * step_count,
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
        if not np.any(cpu_array) and not np.any(gpu_array):
            cpu_array = np.zeros(1)
            gpu_array = np.zeros(1)
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
            if not np.any(cpu_values) and not np.any(gpu_values):
                cpu_values = np.zeros(1)
                gpu_values = np.zeros(1)
            _add_entry(name, cpu_values, gpu_values, None, index)
            _add_entry(f"extra[{name}]", cpu_values, gpu_values, None, index)

        _add_entry("status", cpu_step.status, gpu_step.status, None, index)
        _add_entry("niters", cpu_step.niters, gpu_step.niters, None, index)
        _add_entry(
            "stage_count",
            cpu_step.stage_count,
            gpu_step.stage_count,
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
