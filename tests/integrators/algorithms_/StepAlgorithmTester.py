"""Reusable tester for single-step integration algorithms."""


import numpy as np
import pytest
from numba import cuda
from numba import from_dtype

from integrators.cpu_reference import get_ref_step_fn, CPUODESystem


class StepResult:
    def __init__(self, state, observables, error, status):
        self.state = state
        self.observables = observables
        self.error = error
        self.status = status

@pytest.fixture(scope="function")
def device_step_results(
        step_obj,
        precision,
        x0: np.ndarray,
        step_size: float,
        system,
) -> StepResult:
    """Execute a single step on the device."""

    step_fn = step_obj.step_function
    n_states = step_obj.compile_settings.n
    params = np.array(system.parameters.values_array, dtype=precision, copy=True)
    drivers = np.zeros(system.sizes.drivers, dtype=precision)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    state = np.array(x0[:n_states], dtype=precision, copy=True)
    proposed_state = np.zeros_like(state)
    work_len = max(step_obj.local_scratch_required, n_states)
    work_buffer = np.zeros(work_len, dtype=precision)
    error = np.zeros(n_states, dtype=precision)
    flag = np.full(1, -1, dtype=np.int32)

    persistent_len = max(1, step_obj.persistent_local_required)
    shared_elems = max(0, step_obj.shared_memory_required)
    shared_bytes = int(shared_elems * precision.itemsize)

    numba_precision = from_dtype(precision)
    dt_value = numba_precision(step_size)

    @cuda.jit
    def kernel(
            state_vec,
            proposed_vec,
            work_vec,
            params_vec,
            drivers_vec,
            observables_vec,
            error_vec,
            flag_vec,
            dt_scalar,
    ):
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        status = step_fn(
                state_vec,
                proposed_vec,
                work_vec,
                params_vec,
                drivers_vec,
                observables_vec,
                error_vec,
                dt_scalar,
                shared,
                persistent,
        )
        flag_vec[0] = status

    kernel[1, 1, 0, shared_bytes](
            state,
            proposed_state,
            work_buffer,
            params,
            drivers,
            observables,
            error,
            flag,
            dt_value,
    )
    cuda.synchronize()
    return StepResult(
            state=proposed_state,
            observables=observables,
            error=error,
            status=int(flag[0]))

@pytest.fixture(scope="function")
def cpu_step_results(solver_settings,
                     system,
                     input):
    """Get CPU reference stepper."""
    step_fn = get_ref_step_fn(
            solver_settings['algorithm'],
            CPUODESystem(system),
            solver_settings)
    #
    # results = step_fn(state=state,
    #                   params=params,
    #                   drivers_now=drivers_now,
    #                   drivers_next=drivers_next,
    #                   dt=dt)



@pytest.mark.parametrize("solver_settings",
                         ({'algorithm':'euler'},
                          {'algorithm':'backwards_euler'},
                          {'algorithm':'backwards_euler_predict_correct'},
                          {'algorithm':'crank_nicolson'},),
                         indirect=True)
class StepAlgorithmTester:

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_build(self, step_object, step_size, tolerances) -> None:
        assert callable(step_object.step_function)

