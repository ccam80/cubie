"""Tests for DIRK dense-prediction ownership and seeding policy."""

import attrs
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import numba_from_dtype as from_dtype
from cubie.integrators.algorithms.generic_dirk import (
    DIRKStep,
    stage_seed_sources,
)
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRKTableau,
    IMPLICIT_MIDPOINT_TABLEAU,
    KVAERNO3_TABLEAU,
    L_STABLE_DIRK3_TABLEAU,
)
from tests.integrators.cpu_reference.algorithms import CPUDIRKStep


def opened(tableau, ceiling=8.0):
    """Return the tableau with sweep-open ratio ceilings."""

    return attrs.evolve(
        tableau,
        dense_prediction_ratio_float32=ceiling,
        dense_prediction_ratio_float64=ceiling,
    )


NON_ADJACENT_REPEAT_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.25, 0.0, 0.0),
        (0.25, 0.5, 0.25, 0.0),
        (0.125, 0.125, 0.0, 0.25),
    ),
    b=(0.25, 0.25, 0.25, 0.25),
    c=(0.0, 0.5, 1.0, 0.5),
    order=2,
    dense_prediction_ratio_float32=4.0,
    dense_prediction_ratio_float64=4.0,
)


def test_stage_seed_sources_mappings():
    """Exact predecessor mappings for every repeat layout."""
    # Unique nodes seed from their own predicted rows.
    assert stage_seed_sources((0.0, 0.5, 1.0)) == (0, 1, 2)
    # Adjacent repeats seed from the immediately preceding stage.
    assert stage_seed_sources((0.0, 0.87, 1.0, 1.0)) == (0, 1, 2, 2)
    # Non-adjacent repeats reach back past intervening stages.
    assert stage_seed_sources((0.0, 0.5, 1.0, 0.5)) == (0, 1, 2, 1)
    # An explicit first stage can be a repeated node's source.
    assert stage_seed_sources((0.0, 0.5, 0.0)) == (0, 1, 0)
    # Multiple repeats use the most recent equal-node predecessor.
    assert stage_seed_sources((0.5, 0.5, 0.5)) == (0, 0, 1)


def test_update_rederives_predict_first_stage():
    """Tableau updates in both directions refresh the predictor's
    first-stage row through the single update path."""
    step = DIRKStep(
        precision=np.float64,
        n=3,
        tableau=opened(KVAERNO3_TABLEAU),
    )
    predictor = step.dense_predictor
    assert predictor.compile_settings.predict_first_stage is False
    step.update(tableau=opened(L_STABLE_DIRK3_TABLEAU))
    assert predictor.compile_settings.predict_first_stage is True
    assert (
        step.compile_settings.tableau.stage_count
        == L_STABLE_DIRK3_TABLEAU.stage_count
    )
    step.update(tableau=opened(KVAERNO3_TABLEAU))
    assert predictor.compile_settings.predict_first_stage is False
    assert (
        predictor.compile_settings.stage_count
        == KVAERNO3_TABLEAU.stage_count
    )


def test_previous_step_size_owned_by_algorithm():
    """The previous-step-size scalar lives on the DIRK config; the
    predictor has no such setting."""
    step = DIRKStep(
        precision=np.float32,
        n=2,
        tableau=opened(L_STABLE_DIRK3_TABLEAU),
    )
    assert (
        step.compile_settings.previous_step_size_location == "local"
    )
    step.update(previous_step_size_location="shared")
    assert (
        step.compile_settings.previous_step_size_location == "shared"
    )
    assert not hasattr(
        step.dense_predictor.compile_settings,
        "previous_step_size_location",
    )


def test_single_stage_midpoint_predicts_its_stage():
    """Implicit midpoint's sole implicit stage keeps its predicted
    row."""
    step = DIRKStep(
        precision=np.float64,
        n=2,
        tableau=opened(IMPLICIT_MIDPOINT_TABLEAU),
    )
    assert step.dense_prediction
    assert (
        step.dense_predictor.compile_settings.predict_first_stage
        is True
    )


def _run_schedule(step_object, system, precision, state, params,
                  schedule):
    """Run consecutive accepted steps through the device step.

    Returns the final state and the final step's Newton iterations.
    """
    numba_precision = from_dtype(precision)
    step_fn = step_object.step_function
    shared_elems = int(step_object.shared_buffer_size)
    shared_bytes = precision(0).itemsize * max(shared_elems, 1)
    persistent_len = max(
        1, int(step_object.persistent_local_buffer_size)
    )
    n = int(system.sizes.states)
    n_obs = max(1, int(system.sizes.observables))
    n_drv = max(1, int(system.sizes.drivers))

    @cuda.jit
    def kernel(state_io, params_vec, driver_coeffs, dt_schedule,
               out_iters, status_vec):
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_len, dtype=numba_precision
        )
        state_vec = cuda.local.array(n, dtype=numba_precision)
        proposed = cuda.local.array(n, dtype=numba_precision)
        error = cuda.local.array(n, dtype=numba_precision)
        drivers = cuda.local.array(n_drv, dtype=numba_precision)
        proposed_drivers = cuda.local.array(
            n_drv, dtype=numba_precision
        )
        observables = cuda.local.array(n_obs, dtype=numba_precision)
        proposed_observables = cuda.local.array(
            n_obs, dtype=numba_precision
        )
        counters = cuda.local.array(2, dtype=int32)
        for i in range(persistent_len):
            persistent[i] = numba_precision(0.0)
        for i in range(n):
            state_vec[i] = state_io[i]
            proposed[i] = numba_precision(0.0)
            error[i] = numba_precision(0.0)
        for i in range(n_drv):
            drivers[i] = numba_precision(0.0)
            proposed_drivers[i] = numba_precision(0.0)
        for i in range(n_obs):
            observables[i] = numba_precision(0.0)
            proposed_observables[i] = numba_precision(0.0)
        counters[0] = 0
        counters[1] = 0
        time_value = numba_precision(0.0)
        status = int32(0)
        n_steps = dt_schedule.shape[0]
        for step_index in range(n_steps):
            if step_index == n_steps - 1:
                counters[0] = 0
                counters[1] = 0
            first_flag = int32(1) if step_index == 0 else int32(0)
            result = step_fn(
                state_vec,
                proposed,
                params_vec,
                driver_coeffs,
                drivers,
                proposed_drivers,
                observables,
                proposed_observables,
                error,
                dt_schedule[step_index],
                time_value,
                first_flag,
                int32(1),
                shared,
                persistent,
                counters,
            )
            status = int32(status | result)
            time_value += dt_schedule[step_index]
            for i in range(n):
                state_vec[i] = proposed[i]
        for i in range(n):
            state_io[i] = state_vec[i]
        out_iters[0] = counters[0]
        status_vec[0] = status

    d_state = cuda.to_device(np.asarray(state, dtype=precision))
    d_params = cuda.to_device(np.asarray(params, dtype=precision))
    d_coeffs = cuda.to_device(np.zeros((1, 1, 1), dtype=precision))
    d_schedule = cuda.to_device(
        np.asarray(schedule, dtype=precision)
    )
    d_iters = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_status = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1, 0, shared_bytes](
        d_state, d_params, d_coeffs, d_schedule, d_iters, d_status
    )
    cuda.synchronize()
    assert int(d_status.copy_to_host()[0]) == 0
    return d_state.copy_to_host(), int(d_iters.copy_to_host()[0])


LORENZ_F64 = {
    "system_type": "lorenz_julia",
    "precision": np.float64,
    "algorithm": "l_stable_dirk_3",
    "step_controller": "fixed",
    "dt": 0.005,
}


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_F64], indirect=True
)
def test_ratio_ceiling_boundary_on_device(system, precision):
    """The hoisted ratio gate applies at the ceiling and carries
    above it.

    Three otherwise-identical steps differ only in their tableau's
    calibrated ceiling. The schedule ends with an exactly
    representable ratio of 2.0, so ceilings 2.0 and 8.0 both apply
    prediction — bit-identical execution — while ceiling 1.0 carries
    and its differently seeded Newton solves land on different bits.
    The loose tolerance stops Newton after one iteration so the
    result keeps the seed's imprint; at tight tolerances Newton
    converges to the same bits from either seed.
    """
    ceilings = [2.0, 8.0, 1.0]
    params = np.asarray(
        system.parameters.values_array, dtype=precision
    ).copy()
    params[system.parameters.get_index_of_key("rho")] = 28.0
    state = np.asarray(
        system.initial_values.values_array, dtype=precision
    )
    small_dt = precision(0.005)
    schedule = [small_dt] * 3 + [precision(2.0) * small_dt]
    final_states = {}
    for ceiling in ceilings:
        step = DIRKStep(
            precision=precision,
            n=int(system.sizes.states),
            evaluate_f=system.evaluate_f,
            evaluate_observables=system.evaluate_observables,
            get_solver_helper_fn=system.get_solver_helper,
            tableau=opened(L_STABLE_DIRK3_TABLEAU, ceiling=ceiling),
            newton_atol=1e-3,
            newton_rtol=1e-3,
            krylov_atol=1e-4,
            krylov_rtol=1e-4,
            newton_max_iters=50,
            krylov_max_iters=100,
        )
        final_states[ceiling], _ = _run_schedule(
            step, system, precision, state, params, schedule
        )
    assert np.array_equal(final_states[2.0], final_states[8.0])
    assert not np.array_equal(final_states[1.0], final_states[2.0])


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_F64], indirect=True
)
def test_non_adjacent_repeat_matches_cpu_reference(
    system, precision, cpu_system, cpu_driver_evaluator
):
    """A non-adjacent repeated node integrates identically on device
    and CPU reference.

    The repeated source (stage 1), the target's own predicted row
    (stage 3), and the intervening live increment (stage 2) all
    differ, so an adjacent-only carry would diverge even though each
    solve still converges.
    """
    n = int(system.sizes.states)
    step = DIRKStep(
        precision=precision,
        n=n,
        evaluate_f=system.evaluate_f,
        evaluate_observables=system.evaluate_observables,
        get_solver_helper_fn=system.get_solver_helper,
        tableau=NON_ADJACENT_REPEAT_TABLEAU,
        newton_atol=1e-10,
        newton_rtol=1e-10,
        krylov_atol=1e-10,
        krylov_rtol=1e-10,
        newton_max_iters=50,
        krylov_max_iters=100,
    )
    assert step.dense_prediction
    params = np.asarray(
        system.parameters.values_array, dtype=precision
    )
    state = np.asarray(
        system.initial_values.values_array, dtype=precision
    )
    base_dt = precision(0.005)
    schedule = [
        base_dt,
        base_dt,
        precision(1.5) * base_dt,
        precision(0.75) * base_dt,
    ]
    device_state, _ = _run_schedule(
        step, system, precision, state, params, schedule
    )

    cpu_step = CPUDIRKStep(
        cpu_system,
        cpu_driver_evaluator,
        newton_tol=1e-10,
        newton_rtol=1e-10,
        newton_max_iters=50,
        linear_tol=1e-10,
        linear_rtol=1e-10,
        linear_max_iters=100,
        tableau=NON_ADJACENT_REPEAT_TABLEAU,
    )
    cpu_state = state.copy()
    time_value = 0.0
    for dt_value in schedule:
        result = cpu_step.step(
            state=cpu_state,
            params=params,
            dt=float(dt_value),
            time=time_value,
            prev_accepted=True,
        )
        cpu_state = np.asarray(result.state, dtype=precision)
        time_value += float(dt_value)

    assert_allclose(device_state, cpu_state, rtol=1e-6, atol=1e-9)
