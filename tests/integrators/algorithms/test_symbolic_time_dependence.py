import numpy as np
from numpy.testing import assert_allclose
from numba import cuda

from cubie import SymbolicODE


def _build_time_dependent_system() -> SymbolicODE:
    system = SymbolicODE.create(
        dxdt=["dx = x + t"],
        states={"x": 0.5},
        parameters={},
        constants={},
        observables=[],
        drivers=[],
        name="time_dependent_helper",
        strict=True,
    )
    system.build()
    return system


def test_dirk_helpers_accept_time_argument() -> None:
    system = _build_time_dependent_system()
    precision = system.precision
    numba_precision = system.numba_precision

    mass_matrix = np.eye(system.sizes.states, dtype=precision)
    operator_apply = system.get_solver_helper(
        "linear_operator",
        beta=precision(1.0),
        gamma=precision(1.0),
        mass=mass_matrix,
    )
    residual_fn = system.get_solver_helper(
        "stage_residual",
        beta=precision(1.0),
        gamma=precision(1.0),
        mass=mass_matrix,
    )
    preconditioner = system.get_solver_helper(
        "neumann_preconditioner",
        beta=precision(1.0),
        gamma=precision(1.0),
        mass=mass_matrix,
        preconditioner_order=1,
    )

    state = np.array([0.5], dtype=precision)
    parameters = np.empty(0, dtype=precision)
    drivers = np.empty(0, dtype=precision)
    vector = np.array([0.25], dtype=precision)
    increment = np.array([0.2], dtype=precision)
    base_state = np.array([0.5], dtype=precision)

    h_value = precision(0.1)
    t_value = precision(0.3)
    a_ij_value = precision(0.5)

    state_dev = cuda.to_device(state)
    parameters_dev = cuda.to_device(parameters)
    drivers_dev = cuda.to_device(drivers)
    vector_dev = cuda.to_device(vector)
    operator_out_dev = cuda.to_device(np.zeros_like(vector))

    increment_dev = cuda.to_device(increment)
    base_state_dev = cuda.to_device(base_state)
    residual_out_dev = cuda.to_device(np.zeros_like(increment))

    scratch_dev = cuda.to_device(np.zeros_like(increment))
    preconditioned_out_dev = cuda.to_device(np.zeros_like(increment))

    @cuda.jit
    def operator_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        t_scalar,
        h_scalar,
        vec,
        out,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            operator_apply(
                state_vec,
                params_vec,
                drivers_vec,
                t_scalar,
                h_scalar,
                vec,
                out,
            )

    @cuda.jit
    def residual_kernel(
        increment_vec,
        params_vec,
        drivers_vec,
        t_scalar,
        h_scalar,
        a_scalar,
        base_vec,
        out,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            residual_fn(
                increment_vec,
                params_vec,
                drivers_vec,
                t_scalar,
                h_scalar,
                a_scalar,
                base_vec,
                out,
            )

    @cuda.jit
    def preconditioner_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        t_scalar,
        h_scalar,
        rhs_vec,
        out_vec,
        scratch_vec,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            preconditioner(
                state_vec,
                params_vec,
                drivers_vec,
                t_scalar,
                h_scalar,
                rhs_vec,
                out_vec,
                scratch_vec,
            )

    operator_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        numba_precision(t_value),
        numba_precision(h_value),
        vector_dev,
        operator_out_dev,
    )
    cuda.synchronize()

    residual_kernel[1, 1](
        increment_dev,
        parameters_dev,
        drivers_dev,
        numba_precision(t_value),
        numba_precision(h_value),
        numba_precision(a_ij_value),
        base_state_dev,
        residual_out_dev,
    )
    cuda.synchronize()

    residual_host = residual_out_dev.copy_to_host()
    rhs_dev = cuda.to_device(residual_host)

    preconditioner_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        numba_precision(t_value),
        numba_precision(h_value),
        rhs_dev,
        preconditioned_out_dev,
        scratch_dev,
    )
    cuda.synchronize()

    operator_host = operator_out_dev.copy_to_host()
    residual_result = residual_host.copy()
    preconditioned_host = preconditioned_out_dev.copy_to_host()

    expected_operator = vector * (precision(1.0) - h_value)
    assert_allclose(operator_host, expected_operator)

    stage_state = base_state + a_ij_value * increment
    expected_residual = increment - h_value * (stage_state + t_value)
    assert_allclose(residual_result, expected_residual)

    expected_preconditioned = expected_residual * (precision(1.0) + h_value)
    assert_allclose(preconditioned_host, expected_preconditioned, rtol=1e-6)


def test_rosenbrock_helpers_accept_time_argument() -> None:
    system = _build_time_dependent_system()
    precision = system.precision
    numba_precision = system.numba_precision

    mass_matrix = np.eye(system.sizes.states, dtype=precision)
    prepare_jacobian = system.get_solver_helper(
        "prepare_jac",
        preconditioner_order=1,
    )
    cached_aux_count = system.get_solver_helper("cached_aux_count")
    cached_aux = np.zeros(cached_aux_count, dtype=precision)

    operator_cached = system.get_solver_helper(
        "linear_operator_cached",
        beta=precision(1.0),
        gamma=precision(1.0),
        mass=mass_matrix,
    )
    cached_jvp = system.get_solver_helper(
        "calculate_cached_jvp",
        beta=precision(0.0),
        gamma=precision(-1.0),
        mass=mass_matrix,
    )
    preconditioner_cached = system.get_solver_helper(
        "neumann_preconditioner_cached",
        beta=precision(1.0),
        gamma=precision(1.0),
        mass=mass_matrix,
        preconditioner_order=1,
    )

    state = np.array([0.5], dtype=precision)
    parameters = np.empty(0, dtype=precision)
    drivers = np.empty(0, dtype=precision)
    vector = np.array([0.25], dtype=precision)
    stage_increment = np.array([0.2], dtype=precision)

    h_value = precision(0.1)
    t_value = precision(0.3)

    state_dev = cuda.to_device(state)
    parameters_dev = cuda.to_device(parameters)
    drivers_dev = cuda.to_device(drivers)
    cached_aux_dev = cuda.to_device(cached_aux)

    vector_dev = cuda.to_device(vector)
    operator_out_dev = cuda.to_device(np.zeros_like(vector))

    jvp_out_dev = cuda.to_device(np.zeros_like(stage_increment))
    stage_increment_dev = cuda.to_device(stage_increment)

    scratch_dev = cuda.to_device(np.zeros_like(stage_increment))
    preconditioned_out_dev = cuda.to_device(np.zeros_like(stage_increment))

    residual_value = (
        stage_increment
        - h_value * (state + stage_increment + t_value)
    )
    residual_dev = cuda.to_device(residual_value)

    @cuda.jit
    def prepare_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        t_scalar,
        cached_vec,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            prepare_jacobian(
                state_vec,
                params_vec,
                drivers_vec,
                t_scalar,
                cached_vec,
            )

    @cuda.jit
    def operator_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        cached_vec,
        t_scalar,
        h_scalar,
        vec,
        out,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            operator_cached(
                state_vec,
                params_vec,
                drivers_vec,
                cached_vec,
                t_scalar,
                h_scalar,
                vec,
                out,
            )

    @cuda.jit
    def jvp_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        cached_vec,
        t_scalar,
        vec,
        out,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            cached_jvp(
                state_vec,
                params_vec,
                drivers_vec,
                cached_vec,
                t_scalar,
                vec,
                out,
            )

    @cuda.jit
    def preconditioner_kernel(
        state_vec,
        params_vec,
        drivers_vec,
        cached_vec,
        t_scalar,
        h_scalar,
        rhs_vec,
        out_vec,
        scratch_vec,
    ):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            preconditioner_cached(
                state_vec,
                params_vec,
                drivers_vec,
                cached_vec,
                t_scalar,
                h_scalar,
                rhs_vec,
                out_vec,
                scratch_vec,
            )

    prepare_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        numba_precision(t_value),
        cached_aux_dev,
    )
    cuda.synchronize()

    operator_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        cached_aux_dev,
        numba_precision(t_value),
        numba_precision(h_value),
        vector_dev,
        operator_out_dev,
    )
    cuda.synchronize()

    jvp_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        cached_aux_dev,
        numba_precision(t_value),
        stage_increment_dev,
        jvp_out_dev,
    )
    cuda.synchronize()

    preconditioner_kernel[1, 1](
        state_dev,
        parameters_dev,
        drivers_dev,
        cached_aux_dev,
        numba_precision(t_value),
        numba_precision(h_value),
        residual_dev,
        preconditioned_out_dev,
        scratch_dev,
    )
    cuda.synchronize()

    operator_host = operator_out_dev.copy_to_host()
    jvp_host = jvp_out_dev.copy_to_host()
    preconditioned_host = preconditioned_out_dev.copy_to_host()

    expected_operator = vector * (precision(1.0) - h_value)
    assert_allclose(operator_host, expected_operator)

    assert_allclose(jvp_host, stage_increment)

    expected_preconditioned = residual_value * (precision(1.0) + h_value)
    assert_allclose(preconditioned_host, expected_preconditioned, rtol=1e-6)
