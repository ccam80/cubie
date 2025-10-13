"""Diagnostic helpers for inspecting Rosenbrock intermediates."""

import numpy as np
import pytest
from numba import cuda, from_dtype, int32

from cubie.integrators.algorithms.rosenbrock import RosenbrockTableau
from tests.integrators.cpu_reference import (
    get_ref_step_function,
    _tableau_matrix,
    _tableau_vector,
)

ros1 = RosenbrockTableau(
    a=((0.0,),),
    C=((0.0,),),
    b=(1.0,),
    d=(1.0,),
    c=(0.0,),
    gamma=1.0,
)

ros2 = RosenbrockTableau(
    a=((0.0, 0.0),
       (1.0, 0.0)),
    C=((0.0, 0.0),
       (1.0, 0.0)),
    b=(0.5, 0.5),
    d=(1.0, 1.0),
    c=(0.0, 1.0),
    gamma=0.5,
)

ros3 = RosenbrockTableau(
    a=((0.0, 0.0, 0.0),
       (1.0, 0.0, 0.0),
       (0.75, 0.25, 0.0)),
    C=((0.0, 0.0, 0.0),
       (1.0, 0.0, 0.0),
       (0.75, 0.25, 0.0)),
    b=(2.0/9.0, 1.0/3.0, 4.0/9.0),
    d=(1.0, 1.0, 1.0),
    c=(0.0, 1.0, 0.5),
    gamma=1.0/3.0,
)

ros4 = RosenbrockTableau(
    a=((0.0, 0.0, 0.0, 0.0),
       (0.386, 0.0, 0.0, 0.0),
       (0.21, 0.63, 0.0, 0.0),
       (0.63, -2.0, 1.37, 0.0)),
    C=((0.0, 0.0, 0.0, 0.0),
       (0.386, 0.0, 0.0, 0.0),
       (0.21, 0.63, 0.0, 0.0),
       (0.63, -2.0, 1.37, 0.0)),
    b=(0.25, 0.25, 0.25, 0.25),
    d=(1.0, 1.0, 1.0, 1.0),
    c=(0.0, 0.386, 0.84, 1.0),
    gamma=0.25,
)

ROSENBROCK_W6S4OS_TABLEAU = RosenbrockTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.5812383407115008,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.9039624413714670,
            1.8615191555345010,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.0765797196750000,
            0.1884255381414796,
            1.8701589674910320,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.4355506384843120,
            5.4571817986101890,
            4.6163507880689300,
            3.1181119524023610,
            0.0,
            0.0,
        ),
        (
            10.791701698483260,
            -10.056915225841310,
            14.995644854284190,
            5.2743399543909430,
            1.4297308712611900,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -2.661294105131369,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -3.128450202373838,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -6.920335474535658,
            -1.202675288266817,
            -9.733561811413620,
            0.0,
            0.0,
            0.0,
        ),
        (
            -28.095306291026950,
            20.371262954793770,
            -41.043752753028690,
            -19.663731756208950,
            0.0,
            0.0,
        ),
        (
            9.7998186780974000,
            11.935792886603180,
            3.6738749290132010,
            14.807828541095500,
            0.8318583998690680,
            0.0,
        ),
    ),
    b=(
        6.4562170746532350,
        -4.8531413177680530,
        9.7653183340692600,
        2.0810841772787230,
        0.6603936866352417,
        0.6000000000000000,
    ),
    d=(
        0.2500000000000000,
        0.0836691184292894,
        0.0544718623516351,
        -0.3402289722355864,
        0.0337651588339529,
        -0.0903074267618540,
    ),
    c=(
        0.0,
        0.1453095851778752,
        0.3817422770256738,
        0.6367813704374599,
        0.7560744496323561,
        0.9271047239875670,
    ),
    gamma=0.25,
)

def _build_rosenbrock_debug_kernel(step_object, driver_count, observable_count):
    """Return a CUDA kernel mirroring the Rosenbrock step implementation."""

    config = step_object.compile_settings
    tableau = config.tableau
    stage_count = tableau.stage_count
    numba_precision = config.numba_precision
    has_driver_function = config.driver_function is not None
    dxdt_fn = config.dxdt_function
    observables_function = config.observables_function
    driver_function = config.driver_function

    stage_rhs_coeffs = step_object.tableau.typed_rows(
        tableau.a, numba_precision
    )
    jacobian_update_coeffs = step_object.tableau.typed_rows(
        tableau.C, numba_precision
    )
    solution_weights = tuple(numba_precision(value) for value in tableau.b)
    error_weights = tuple(numba_precision(value) for value in tableau.d)
    stage_time_fractions = tuple(
        numba_precision(value) for value in tableau.c
    )
    typed_zero = numba_precision(0.0)
    cached_auxiliary_count = step_object.cached_auxiliary_count
    n = config.n

    (
        linear_solver,
        prepare_jacobian,
        cached_jvp,
    ) = step_object.build_implicit_helpers()

    @cuda.jit(
        # (
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision[:, :, :],
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision[:],
        #     numba_precision,
        #     numba_precision,
        #     numba_precision[:, :],
        #     numba_precision[:, :],
        #     numba_precision[:],
        #     numba_precision[:, :],
        #     numba_precision[:, :],
        #     numba_precision[:, :],
        #     numba_precision[:, :],
        #     numba_precision[:, :],
        #     numba_precision[:],
        # ),
        device=True,
        inline=True,
    )
    def debug_step(
        state,
        proposed_state,
        parameters,
        driver_coeffs,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        state_accumulator,
        jacobian_accumulator,
        cached_auxiliaries,
        stage_states,
        stage_rhs_before_solve,
        stage_rhs_values,
        stage_increments,
        jacobian_products,
        driver_samples,
        observable_samples,
        final_state,
    ):
        dt_value = dt_scalar
        current_time = time_scalar
        end_time = current_time + dt_value

        status_code = int32(0)

        rows = state_accumulator.shape[0]
        for row in range(rows):
            for col in range(n):
                state_accumulator[row, col] = typed_zero
                jacobian_accumulator[row, col] = typed_zero

        for idx in range(cached_auxiliary_count):
            cached_auxiliaries[idx] = typed_zero

        stages = stage_states.shape[0]
        for row in range(stages):
            for col in range(n):
                stage_states[row, col] = typed_zero
                stage_rhs_before_solve[row, col] = typed_zero
                stage_rhs_values[row, col] = typed_zero
                stage_increments[row, col] = typed_zero
                jacobian_products[row, col] = typed_zero

        driver_rows = driver_samples.shape[0]
        driver_cols = driver_samples.shape[1]
        for row in range(driver_rows):
            for col in range(driver_cols):
                driver_samples[row, col] = typed_zero

        observable_rows = observable_samples.shape[0]
        observable_cols = observable_samples.shape[1]
        for row in range(observable_rows):
            for col in range(observable_cols):
                observable_samples[row, col] = typed_zero

        for idx in range(n):
            proposed_state[idx] = typed_zero
            error[idx] = typed_zero
            final_state[idx] = typed_zero
            stage_states[0, idx] = state[idx]

        for col in range(driver_cols):
            if col < drivers_buffer.size:
                driver_samples[0, col] = drivers_buffer[col]
            else:
                driver_samples[0, col] = typed_zero

        prepare_jacobian(
            state,
            parameters,
            drivers_buffer,
            cached_auxiliaries,
        )

        dxdt_fn(
            state,
            parameters,
            drivers_buffer,
            observables,
            stage_rhs_values[0],
            current_time,
        )

        for idx in range(n):
            stage_rhs_values[0, idx] = (
                dt_value * stage_rhs_values[0, idx]
            )

        for idx in range(n):
            stage_rhs_before_solve[0, idx] = stage_rhs_values[0, idx]

        status_code |= linear_solver(
            state,
            parameters,
            drivers_buffer,
            cached_auxiliaries,
            dt_value,
            stage_rhs_values[0],
            stage_increments[0],
        )

        for idx in range(n):
            increment = stage_increments[0, idx]
            proposed_state[idx] += solution_weights[0] * increment
            error[idx] += error_weights[0] * increment

        cached_jvp(
            state,
            parameters,
            drivers_buffer,
            cached_auxiliaries,
            stage_increments[0],
            jacobian_products[0],
        )

        for stage_idx in range(1, stage_count):
            prev_idx = stage_idx - 1
            successor_count = stage_count - stage_idx
            for successor_offset in range(successor_count):
                successor_idx = stage_idx + successor_offset
                state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                jac_coeff = jacobian_update_coeffs[successor_idx][prev_idx]
                base = successor_idx - 1
                if state_coeff != typed_zero:
                    for col in range(n):
                        state_accumulator[base, col] += (
                            state_coeff * stage_increments[prev_idx, col]
                        )
                if jac_coeff != typed_zero:
                    for col in range(n):
                        jacobian_accumulator[base, col] += (
                            jac_coeff * jacobian_products[prev_idx, col]
                        )

            stage_offset = stage_idx - 1
            stage_time = (
                current_time + dt_value * stage_time_fractions[stage_idx]
            )

            for idx in range(n):
                value = state[idx]
                if stage_offset < rows:
                    value += state_accumulator[stage_offset, idx]
                stage_states[stage_idx, idx] = value

            stage_drivers = proposed_drivers
            for col in range(driver_cols):
                stage_drivers[col] = typed_zero

            if has_driver_function:
                driver_function(
                    stage_time,
                    driver_coeffs,
                    stage_drivers,
                )

            for col in range(driver_cols):
                driver_samples[stage_idx, col] = stage_drivers[col]

            observables_function(
                stage_states[stage_idx],
                parameters,
                stage_drivers,
                proposed_observables,
                stage_time,
            )

            for col in range(observable_cols):
                observable_samples[stage_idx, col] = (
                    proposed_observables[col]
                )

            dxdt_fn(
                stage_states[stage_idx],
                parameters,
                stage_drivers,
                proposed_observables,
                stage_rhs_values[stage_idx],
                stage_time,
            )

            for idx in range(n):
                rhs_value = stage_rhs_values[stage_idx, idx]
                if stage_offset < rows:
                    rhs_value += jacobian_accumulator[stage_offset, idx]
                stage_increments[stage_idx, idx] = typed_zero
                stage_rhs_values[stage_idx, idx] = dt_value * rhs_value

            for idx in range(n):
                stage_rhs_before_solve[stage_idx, idx] = (
                    stage_rhs_values[stage_idx, idx]
                )

            status_code |= linear_solver(
                state,
                parameters,
                drivers_buffer,
                cached_auxiliaries,
                dt_value,
                stage_rhs_values[stage_idx],
                stage_increments[stage_idx],
            )

            solution_weight = solution_weights[stage_idx]
            error_weight = error_weights[stage_idx]
            for idx in range(n):
                increment = stage_increments[stage_idx, idx]
                proposed_state[idx] += solution_weight * increment
                error[idx] += error_weight * increment

            if stage_idx < stage_count - 1:
                cached_jvp(
                    state,
                    parameters,
                    drivers_buffer,
                    cached_auxiliaries,
                    stage_increments[stage_idx],
                    jacobian_products[stage_idx],
                )

        final_time = end_time
        if has_driver_function:
            driver_function(
                final_time,
                driver_coeffs,
                proposed_drivers,
            )

        for col in range(driver_cols):
            driver_samples[stage_count, col] = proposed_drivers[col]

        for idx in range(n):
            value = state[idx] + proposed_state[idx]
            proposed_state[idx] = value
            final_state[idx] = value

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            final_time,
        )

        for col in range(observable_cols):
            observable_samples[stage_count, col] = proposed_observables[col]

        return status_code

    @cuda.jit
    def kernel(
        state,
        proposed_state,
        parameters,
        driver_coeffs,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        state_accumulator,
        jacobian_accumulator,
        cached_auxiliaries,
        stage_states,
        stage_rhs_before_solve,
        stage_rhs_values,
        stage_increments,
        jacobian_products,
        driver_samples,
        observable_samples,
        final_state,
        status_out,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return
        status_out[0] = debug_step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            state_accumulator,
            jacobian_accumulator,
            cached_auxiliaries,
            stage_states,
            stage_rhs_before_solve,
            stage_rhs_values,
            stage_increments,
            jacobian_products,
            driver_samples,
            observable_samples,
            final_state,
        )

    return kernel


def _collect_cpu_rosenbrock_intermediates(
    cpu_system,
    driver_evaluator,
    state,
    params,
    dt_value,
    tableau,
    driver_coefficients,
):
    """Evaluate the CPU Rosenbrock reference and capture intermediates."""

    precision = cpu_system.precision
    state_vec = np.array(state, dtype=precision, copy=True)
    params_vec = np.array(params, dtype=precision, copy=True)
    dt_scalar = precision(dt_value)
    current_time = precision(0.0)

    if cpu_system.system.num_drivers > 0:
        evaluator = driver_evaluator.with_coefficients(driver_coefficients)
    else:
        evaluator = driver_evaluator

    stage_count = tableau.stage_count
    n = state_vec.size
    driver_width = evaluator.coefficients.shape[1]
    observable_width = cpu_system.n_observables

    a_matrix = _tableau_matrix(tableau.a, stage_count, precision)
    C_matrix = _tableau_matrix(tableau.C, stage_count, precision)
    b_weights = _tableau_vector(tableau.b, precision)
    d_weights = _tableau_vector(tableau.d, precision)
    c_nodes = _tableau_vector(tableau.c, precision)
    gamma = precision(tableau.gamma)

    drivers_now = evaluator(float(current_time))
    observables_now = cpu_system.observables(
        state_vec,
        params_vec,
        drivers_now,
        current_time,
    )
    f_now, _ = cpu_system.rhs(
        state_vec,
        params_vec,
        drivers_now,
        observables_now,
        current_time,
    )
    jacobian = cpu_system.jacobian(
        state_vec,
        params_vec,
        drivers_now,
        observables_now,
        current_time,
    )

    identity = np.eye(n, dtype=precision)
    lhs = identity - dt_scalar * gamma * jacobian

    stage_states = np.zeros((stage_count, n), dtype=precision)
    stage_rhs_pre_solve = np.zeros_like(stage_states)
    stage_rhs_post_solve = np.zeros_like(stage_states)
    stage_increments = np.zeros_like(stage_states)
    jacobian_products = np.zeros_like(stage_states)
    state_shifts = np.zeros((stage_count, n), dtype=precision)
    jacobian_shifts = np.zeros((stage_count, n), dtype=precision)
    state_accumulator = np.zeros(
        (max(stage_count - 1, 0), n), dtype=precision
    )
    jacobian_accumulator = np.zeros_like(state_accumulator)
    driver_samples = np.zeros(
        (stage_count + 1, driver_width), dtype=precision
    )
    observable_samples = np.zeros(
        (stage_count + 1, observable_width), dtype=precision
    )

    state_states0 = stage_states[0]
    state_states0[:] = state_vec
    stage_rhs_pre_solve[0] = dt_scalar * f_now
    driver_samples[0] = drivers_now
    observable_samples[0] = observables_now

    stage_increments[0] = np.linalg.solve(lhs, stage_rhs_pre_solve[0])
    stage_rhs_post_solve[0] = (
        lhs @ stage_increments[0] - stage_rhs_pre_solve[0]
    )
    jacobian_products[0] = jacobian @ stage_increments[0]

    solution_accum = b_weights[0] * stage_increments[0]
    error_accum = d_weights[0] * stage_increments[0]

    for successor in range(1, stage_count):
        a_coeff = a_matrix[successor, 0]
        c_coeff = C_matrix[successor, 0]
        if a_coeff != 0:
            state_shifts[successor] += a_coeff * stage_increments[0]
            state_accumulator[successor - 1] += (
                a_coeff * stage_increments[0]
            )
        if c_coeff != 0:
            jacobian_shifts[successor] += c_coeff * stage_increments[0]
            jacobian_accumulator[successor - 1] += (
                c_coeff * jacobian_products[0]
            )

    for stage_idx in range(1, stage_count):
        stage_time = (
            current_time + c_nodes[stage_idx] * dt_scalar
        )
        stage_state = state_vec + state_shifts[stage_idx]
        stage_states[stage_idx] = stage_state

        drivers_stage = evaluator(float(stage_time))
        driver_samples[stage_idx] = drivers_stage
        observables_stage = cpu_system.observables(
            stage_state,
            params_vec,
            drivers_stage,
            stage_time,
        )
        observable_samples[stage_idx] = observables_stage
        f_stage, _ = cpu_system.rhs(
            stage_state,
            params_vec,
            drivers_stage,
            observables_stage,
            stage_time,
        )

        stage_rhs_pre_solve[stage_idx] = dt_scalar * f_stage
        jac_term = jacobian @ jacobian_shifts[stage_idx]
        stage_rhs_pre_solve[stage_idx] += dt_scalar * jac_term

        stage_increments[stage_idx] = np.linalg.solve(
            lhs, stage_rhs_pre_solve[stage_idx]
        )
        stage_rhs_post_solve[stage_idx] = (
            lhs @ stage_increments[stage_idx]
            - stage_rhs_pre_solve[stage_idx]
        )
        jacobian_products[stage_idx] = (
            jacobian @ stage_increments[stage_idx]
        )
        solution_accum += b_weights[stage_idx] * stage_increments[stage_idx]
        error_accum += d_weights[stage_idx] * stage_increments[stage_idx]

        for successor in range(stage_idx + 1, stage_count):
            a_coeff = a_matrix[successor, stage_idx]
            c_coeff = C_matrix[successor, stage_idx]
            if a_coeff != 0:
                state_shifts[successor] += (
                    a_coeff * stage_increments[stage_idx]
                )
                state_accumulator[successor - 1] += (
                    a_coeff * stage_increments[stage_idx]
                )
            if c_coeff != 0:
                jacobian_shifts[successor] += (
                    c_coeff * stage_increments[stage_idx]
                )
                jacobian_accumulator[successor - 1] += (
                    c_coeff * jacobian_products[stage_idx]
                )

    jacobian_products_device_like = jacobian_products.copy()
    if stage_count > 1:
        jacobian_products_device_like[-1].fill(precision(0.0))

    final_state = state_vec + solution_accum
    final_time = current_time + dt_scalar
    drivers_end = evaluator(float(final_time))
    driver_samples[stage_count] = drivers_end
    observables_final = cpu_system.observables(
        final_state,
        params_vec,
        drivers_end,
        final_time,
    )
    observable_samples[stage_count] = observables_final

    return {
        "stage_states": stage_states,
        "stage_rhs_pre_solve": stage_rhs_pre_solve,
        "stage_rhs_post_solve": stage_rhs_post_solve,
        "stage_increments": stage_increments,
        "jacobian_products": jacobian_products,
        "jacobian_products_device_like": jacobian_products_device_like,
        "state_accumulator": state_accumulator,
        "jacobian_accumulator": jacobian_accumulator,
        "driver_samples": driver_samples,
        "observable_samples": observable_samples,
        "final_state": final_state,
        "error": error_accum,
        "status": 0,
    }


def _collect_device_rosenbrock_intermediates(
    step_object,
    solver_settings,
    system,
    precision,
    state,
    parameters,
    drivers_buffer,
    driver_coefficients,
):
    """Execute the debug kernel and gather device intermediates."""

    config = step_object.compile_settings
    stage_count = config.tableau.stage_count
    n = config.n
    driver_count = system.num_drivers
    observable_count = system.sizes.observables

    kernel = _build_rosenbrock_debug_kernel(
        step_object,
        driver_count,
        observable_count,
    )

    dt_value = precision(solver_settings["dt"])
    numba_precision = from_dtype(precision)
    time_scalar = precision(0.0)

    state = np.array(state, dtype=precision, copy=True)
    params = np.array(parameters, dtype=precision, copy=True)
    drivers_buffer = np.array(drivers_buffer, dtype=precision, copy=True)
    driver_coeffs = np.array(
        driver_coefficients, dtype=precision, copy=True
    )
    observables = np.zeros(system.sizes.observables, dtype=precision)
    proposed_state = np.zeros_like(state)
    proposed_drivers = np.zeros_like(drivers_buffer)
    proposed_observables = np.zeros_like(observables)
    error = np.zeros(n, dtype=precision)

    state_accumulator = np.zeros(
        (max(stage_count - 1, 0), n), dtype=precision
    )
    jacobian_accumulator = np.zeros_like(state_accumulator)
    cached_auxiliaries = np.zeros(
        step_object.cached_auxiliary_count, dtype=precision
    )
    stage_states = np.zeros((stage_count, n), dtype=precision)
    stage_rhs_before_solve = np.zeros_like(stage_states)
    stage_rhs_values = np.zeros_like(stage_states)
    stage_increments = np.zeros_like(stage_states)
    jacobian_products = np.zeros_like(stage_states)
    driver_samples = np.zeros(
        (stage_count + 1, driver_count), dtype=precision
    )
    observable_samples = np.zeros(
        (stage_count + 1, observable_count), dtype=precision
    )
    final_state = np.zeros(n, dtype=precision)
    status_out = np.zeros(1, dtype=np.int32)

    d_state = cuda.to_device(state)
    d_proposed_state = cuda.to_device(proposed_state)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coeffs)
    d_drivers = cuda.to_device(drivers_buffer)
    d_proposed_drivers = cuda.to_device(proposed_drivers)
    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(proposed_observables)
    d_error = cuda.to_device(error)
    d_state_accumulator = cuda.to_device(state_accumulator)
    d_jacobian_accumulator = cuda.to_device(jacobian_accumulator)
    d_cached_auxiliaries = cuda.to_device(cached_auxiliaries)
    d_stage_states = cuda.to_device(stage_states)
    d_stage_rhs_before_solve = cuda.to_device(stage_rhs_before_solve)
    d_stage_rhs_values = cuda.to_device(stage_rhs_values)
    d_stage_increments = cuda.to_device(stage_increments)
    d_jacobian_products = cuda.to_device(jacobian_products)
    d_driver_samples = cuda.to_device(driver_samples)
    d_observable_samples = cuda.to_device(observable_samples)
    d_final_state = cuda.to_device(final_state)
    d_status = cuda.to_device(status_out)

    kernel[1, 1](
        d_state,
        d_proposed_state,
        d_params,
        d_driver_coeffs,
        d_drivers,
        d_proposed_drivers,
        d_observables,
        d_proposed_observables,
        d_error,
        numba_precision(dt_value),
        numba_precision(time_scalar),
        d_state_accumulator,
        d_jacobian_accumulator,
        d_cached_auxiliaries,
        d_stage_states,
        d_stage_rhs_before_solve,
        d_stage_rhs_values,
        d_stage_increments,
        d_jacobian_products,
        d_driver_samples,
        d_observable_samples,
        d_final_state,
        d_status,
    )
    cuda.synchronize()

    return {
        "stage_states": d_stage_states.copy_to_host(),
        "stage_rhs_pre_solve": d_stage_rhs_before_solve.copy_to_host(),
        "stage_rhs": d_stage_rhs_values.copy_to_host(),
        "stage_increments": d_stage_increments.copy_to_host(),
        "jacobian_products": d_jacobian_products.copy_to_host(),
        "state_accumulator": d_state_accumulator.copy_to_host(),
        "jacobian_accumulator": d_jacobian_accumulator.copy_to_host(),
        "driver_samples": d_driver_samples.copy_to_host(),
        "observable_samples": d_observable_samples.copy_to_host(),
        "final_state": d_final_state.copy_to_host(),
        "error": d_error.copy_to_host(),
        "status": int(d_status.copy_to_host()[0]),
    }


def _summarise_difference(cpu_data, device_data):
    """Return absolute and relative maxima for the collected arrays."""

    comparisons = {
        "stage_states": ("stage_states", "stage_states"),
        "stage_rhs_pre_solve": (
            "stage_rhs_pre_solve",
            "stage_rhs_pre_solve",
        ),
        "stage_rhs_post_solve": (
            "stage_rhs",
            "stage_rhs_post_solve",
        ),
        "stage_increments": ("stage_increments", "stage_increments"),
        "jacobian_products": (
            "jacobian_products",
            "jacobian_products_device_like",
        ),
        "state_accumulator": (
            "state_accumulator",
            "state_accumulator",
        ),
        "jacobian_accumulator": (
            "jacobian_accumulator",
            "jacobian_accumulator",
        ),
        "driver_samples": (
            "driver_samples",
            "driver_samples",
        ),
        "observable_samples": (
            "observable_samples",
            "observable_samples",
        ),
        "final_state": ("final_state", "final_state"),
        "error": ("error", "error"),
    }

    summary = {}
    for key, (device_key, cpu_key) in comparisons.items():
        device_values = device_data[device_key]
        cpu_values = cpu_data[cpu_key]
        if device_values.shape != cpu_values.shape:
            summary[key] = {
                "abs_max": float("nan"),
                "rel_max": float("nan"),
            }
            continue
        if device_values.size == 0:
            summary[key] = {"abs_max": 0.0, "rel_max": 0.0}
            continue
        diff = device_values - cpu_values
        abs_max = float(np.max(np.abs(diff)))
        denom = np.maximum(
            np.maximum(np.abs(cpu_values), np.abs(device_values)),
            1e-12,
        )
        rel_max = float(np.max(np.abs(diff) / denom))
        summary[key] = {"abs_max": abs_max, "rel_max": rel_max}
    return summary


def _run_cpu_step(
    cpu_system,
    cpu_driver_evaluator,
    state,
    parameters,
    dt_value,
    tableau,
    driver_coefficients,
    solver_settings,
):
    """Execute the CPU Rosenbrock reference step."""

    step_function = get_ref_step_function(
        "rosenbrock", tableau=tableau
    )
    precision = cpu_system.precision
    state_vec = np.array(state, dtype=precision, copy=True)
    params_vec = np.array(parameters, dtype=precision, copy=True)
    if cpu_system.system.num_drivers > 0:
        driver_eval = cpu_driver_evaluator.with_coefficients(
            driver_coefficients
        )
    else:
        driver_eval = cpu_driver_evaluator

    return step_function(
        cpu_system,
        driver_eval,
        state=state_vec,
        params=params_vec,
        dt=dt_value,
        tol=solver_settings["newton_tolerance"],
        time=0.0,
    )


def _run_device_step(
    step_object,
    solver_settings,
    system,
    precision,
    state,
    parameters,
    driver_coefficients,
    drivers_buffer,
):
    """Execute the compiled device step function for comparison."""

    step_function = step_object.step_function
    state_vec = np.array(state, dtype=precision, copy=True)
    params_vec = np.array(parameters, dtype=precision, copy=True)
    driver_coeffs_vec = np.array(
        driver_coefficients, dtype=precision, copy=True
    )
    drivers_vec = np.array(drivers_buffer, dtype=precision, copy=True)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    proposed_state = np.zeros_like(state_vec)
    proposed_drivers = np.zeros_like(drivers_vec)
    proposed_observables = np.zeros_like(observables)
    error = np.zeros(system.sizes.states, dtype=precision)
    status = np.zeros(1, dtype=np.int32)

    shared_elems = step_object.shared_memory_required
    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_required)
    numba_precision = from_dtype(precision)
    dt_value = precision(solver_settings["dt"])

    d_state = cuda.to_device(state_vec)
    d_proposed = cuda.to_device(proposed_state)
    d_params = cuda.to_device(params_vec)
    d_driver_coeffs = cuda.to_device(driver_coeffs_vec)
    d_drivers = cuda.to_device(drivers_vec)
    d_proposed_drivers = cuda.to_device(proposed_drivers)
    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(proposed_observables)
    d_error = cuda.to_device(error)
    d_status = cuda.to_device(status)

    @cuda.jit
    def kernel(
        state_vec_device,
        proposed_vec,
        params_vec_device,
        driver_coeffs_vec_device,
        drivers_vec_device,
        proposed_drivers_vec,
        observables_vec,
        proposed_observables_vec,
        error_vec,
        status_vec,
        dt_scalar,
        time_scalar,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        status_vec[0] = step_function(
            state_vec_device,
            proposed_vec,
            params_vec_device,
            driver_coeffs_vec_device,
            drivers_vec_device,
            proposed_drivers_vec,
            observables_vec,
            proposed_observables_vec,
            error_vec,
            dt_scalar,
            time_scalar,
            shared,
            persistent,
        )

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
        d_status,
        numba_precision(dt_value),
        numba_precision(precision(0.0)),
    )
    cuda.synchronize()

    return {
        "state": d_proposed.copy_to_host(),
        "error": d_error.copy_to_host(),
        "status": int(d_status.copy_to_host()[0]),
    }

@pytest.mark.parametrize("solver_settings_override",
                         [{"tableau": ros1,
                           "algorithm": "rosenbrock",
                           "step_controller": "pi",
                           },
                          {"tableau": ros2,
                           "algorithm": "rosenbrock",
                           "step_controller": "pi",
                           },
                          {"tableau": ros3,
                           "algorithm": "rosenbrock",
                           "step_controller": "pi",
                           },
                          {"tableau": ros4,
                           "algorithm": "rosenbrock",
                           "step_controller": "pi",
                           },
                          {"tableau": ROSENBROCK_W6S4OS_TABLEAU,
                           "algorithm": "rosenbrock",
                           "step_controller": "pi",
                           }
                          ],
                         ids=["euler", "ROS2", "ROS3", "ROS4", "W6S40S"],
                         indirect=True
                         )
def test_rosenbrock_intermediate_scratch(
    step_object,
    solver_settings,
    system,
    precision,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
):
    """Run CPU and device Rosenbrock steps and compare intermediates."""

    step_object = step_object
    tableau = getattr(
        step_object,
        "tableau",
        solver_settings.get("tableau"),
    )

    parameters = system.parameters.values_array.astype(precision)
    driver_coefficients = np.array(
        cpu_driver_evaluator.coefficients,
        dtype=precision,
        copy=True,
    )
    drivers_buffer = np.array(
        cpu_driver_evaluator.evaluate(float(precision(0.0))),
        dtype=precision,
        copy=True,
    )

    cpu_data = _collect_cpu_rosenbrock_intermediates(
        cpu_system,
        cpu_driver_evaluator,
        initial_state,
        parameters,
        solver_settings["dt"],
        tableau,
        driver_coefficients,
    )
    device_data = _collect_device_rosenbrock_intermediates(
        step_object,
        solver_settings,
        system,
        precision,
        initial_state,
        parameters,
        drivers_buffer,
        driver_coefficients,
    )
    cpu_step = _run_cpu_step(
        cpu_system,
        cpu_driver_evaluator,
        initial_state,
        parameters,
        solver_settings["dt"],
        tableau,
        driver_coefficients,
        solver_settings,
    )
    device_step = _run_device_step(
        step_object,
        solver_settings,
        system,
        precision,
        initial_state,
        parameters,
        driver_coefficients,
        drivers_buffer,
    )

    summary = _summarise_difference(cpu_data, device_data)

    print("Rosenbrock intermediate comparison (abs_max / rel_max):")
    for key in [
        "stage_states",
        "stage_rhs_pre_solve",
        "stage_rhs_post_solve",
        "stage_increments",
        "jacobian_products",
        "state_accumulator",
        "jacobian_accumulator",
        "driver_samples",
        "observable_samples",
        "final_state",
        "error",
    ]:
        metrics = summary[key]
        print(
            f"{key}: {metrics['abs_max']:.6e} / "
            f"{metrics['rel_max']:.6e}"
        )

    print(
        "CPU status:",
        cpu_step.status,
        "Device status:",
        device_step["status"],
        "Debug status:",
        device_data["status"],
    )

    print(
        "CPU final norm:",
        float(np.linalg.norm(cpu_step.state)),
        "Device final norm:",
        float(np.linalg.norm(device_step["state"])),
        "Debug final norm:",
        float(np.linalg.norm(device_data["final_state"])),
    )
