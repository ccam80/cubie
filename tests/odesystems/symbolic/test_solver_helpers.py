import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from cubie.odesystems.symbolic.solver_helpers import (
    generate_cached_jvp_code,
    generate_cached_operator_apply_code,
    generate_neumann_preconditioner_cached_code,
    generate_operator_apply_code,
    generate_neumann_preconditioner_code,
    generate_prepare_jac_code,
    generate_stage_residual_code,
)


@pytest.fixture(scope="function")
def operator_system():
    """Build a linear system with a constant Jacobian."""

    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    return system


def _build_operator_factory(system, precision):
    def factory(beta, gamma, M):
        fname = (
            "operator_apply_factory_"
            f"{abs(hash((beta, gamma, M.tobytes())))}"
        )
        code = generate_operator_apply_code(
            system.equations,
            system.indices,
            M=M,
            func_name=fname,
        )
        op_fac = system.gen_file.import_function(fname, code)
        return op_fac(
            system.constants.values_dict,
            from_dtype(system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def operator_factory(operator_system, precision):
    """Return a factory producing operator_apply device functions."""

    return _build_operator_factory(operator_system, precision)


@pytest.fixture(scope="function")
def operator_kernel(precision):
    """Kernel applying operator_apply to a vector."""

    n = 2

    def make_kernel(op):
        @cuda.jit
        def kernel(t, h, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            op(state, parameters, drivers, t, h, vec, out)

        return kernel

    return make_kernel


@pytest.fixture(scope="function")
def cached_system():
    """Build a nonlinear system with state-dependent Jacobian."""

    dxdt = [
        "dx0 = a*x0*x1 + b*sin(x0)",
        "dx1 = c*x0*x1 + d*cos(x1)",
    ]
    constants = {"a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    return system


@pytest.fixture(scope="function")
def prepare_jac_factory(cached_system, precision):
    """Return a factory producing prepare_jac device functions."""

    def factory():
        fname = "prepare_jac_factory"
        code, aux_count = generate_prepare_jac_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        prep_fac = cached_system.gen_file.import_function(fname, code)
        prepare = prep_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )
        return prepare, aux_count

    return factory


@pytest.fixture(scope="function")
def cached_jvp_factory(cached_system, precision):
    """Return a factory producing calculate_cached_jvp device functions."""

    def factory():
        fname = "cached_jvp_factory"
        code = generate_cached_jvp_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        jvp_fac = cached_system.gen_file.import_function(fname, code)
        return jvp_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )

    return factory


@pytest.fixture(scope="function")
def cached_jvp_kernel(cached_system, precision):
    """Apply cached JVP outputs for comparison with analytic Jacobian."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, cached_jvp, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            vec,
            out_cached,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            cached_jvp(
                state,
                parameters,
                drivers,
                cached_aux,
                t,
                vec,
                out_cached,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="function")
def cached_operator_factory(cached_system, precision):
    """Return a factory producing cached operator device functions."""

    def factory(beta, gamma, M):
        fname = (
            "cached_operator_factory_"
            f"{abs(hash((beta, gamma, M.tobytes())))}"
        )
        code = generate_cached_operator_apply_code(
            cached_system.equations,
            cached_system.indices,
            M=M,
            func_name=fname,
        )
        op_fac = cached_system.gen_file.import_function(fname, code)
        return op_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def cached_operator_kernel(cached_system, precision):
    """Kernel applying cached operator to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, op, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            vec,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            op(state, parameters, drivers, cached_aux, t, h, vec, out)

        return kernel

    return make_kernel


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    operator_factory,
    operator_kernel,
    precision,
    tolerance,
):
    """Evaluate operator_apply for specific scalings and mass matrices."""

    op = operator_factory(beta, gamma, M)
    kernel = operator_kernel(op)
    v = np.array([1.0, -1.0], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(0.0), precision(h), v, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ v - gamma * h * J @ v
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_operator_apply_constant_unpacking(operator_system):
    """Ensure constants are defined as individual variables."""
    code = generate_operator_apply_code(
        operator_system.equations, operator_system.indices
    )
    assert "a = precision(constants['a'])" in code


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
def test_cached_jvp_matches_jacobian(
    cached_system,
    prepare_jac_factory,
    cached_jvp_factory,
    cached_jvp_kernel,
    precision,
    tolerance,
):
    """Ensure cached JVP equals the analytic Jacobian-vector product."""

    prepare, aux_count = prepare_jac_factory()
    cached_jvp = cached_jvp_factory()
    kernel = cached_jvp_kernel(prepare, cached_jvp, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out_cached = np.zeros(state_len, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        vec,
        out_cached,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    expected = jacobian @ vec

    assert np.allclose(
        out_cached,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 0.25, np.eye(2)),
        (1.0, 1.0, 0.25, np.diag([1.2, 0.8])),
        (0.5, 1.7, 0.15, np.array([[1.0, 0.3], [0.4, 1.5]])),
    ],
)
def test_cached_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    cached_operator_factory,
    cached_operator_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
):
    """Evaluate cached operator using precomputed auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    op = cached_operator_factory(beta, gamma, M)
    kernel = cached_operator_kernel(prepare, op, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        vec,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    h_val = precision(h)
    mass = np.array(M, dtype=precision)
    expected = (
        beta_val * mass @ vec - gamma_val * h_val * jacobian @ vec
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


# ---------------------------------------------------------------------------
# Neumann preconditioner expression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def neumann_factory(operator_system, precision):
    """Return a factory producing Neumann preconditioner device functions."""

    def factory(beta, gamma, order):
        fname = (f"neumann_preconditioner_factory_{int(beta)}_{int(gamma)}"
                 f"_{order}")
        code = generate_neumann_preconditioner_code(
            operator_system.equations,
            operator_system.indices,
            func_name=fname,
        )
        pre_fac = operator_system.gen_file.import_function(fname, code)
        return pre_fac(
            operator_system.constants.values_dict,
            from_dtype(operator_system.precision),
            beta=beta,
            gamma=gamma,
            order=order,
        )

    return factory


@pytest.fixture(scope="function")
def neumann_kernel(precision):
    """Apply the Neumann preconditioner to a vector, passing scratch."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            scratch = cuda.local.array(n, precision)
            pre(state, parameters, drivers, t, h, vec, out, scratch)

        return kernel

    return make_kernel


@pytest.fixture(scope="function")
def neumann_cached_factory(cached_system, precision):
    """Return a factory producing cached Neumann preconditioners."""

    def factory(beta, gamma, order):
        fname = (
            "neumann_cached_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}_{order}"
        )
        code = generate_neumann_preconditioner_cached_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        pre_fac = cached_system.gen_file.import_function(fname, code)
        return pre_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
            order=order,
        )

    return factory


@pytest.fixture(scope="function")
def neumann_cached_kernel(cached_system, precision):
    """Apply cached Neumann preconditioner to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, pre, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            vec,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            jvp = cuda.local.array(n_state, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                t,
                h,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 2.0, 0.1, 3),
    ],
)
def test_neumann_preconditioner_expression(
    beta,
    gamma,
    h,
    order,
    neumann_factory,
    neumann_kernel,
    precision,
    tolerance,
):
    """Validate Neumann preconditioner against a truncated series.

    System: dx/dt = J x with J = [[a, b], [c, d]] = [[1, 2], [3, 4]].
    Preconditioner approximates (beta*I - gamma*h*J)^{-1} via truncated series.
    """
    pre = neumann_factory(beta, gamma, order)
    kernel = neumann_kernel(pre)

    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](precision(0.0), precision(h), v, out)

    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    beta_inv = 1.0 / beta
    T = (gamma * beta_inv) * h * J

    # Truncated Neumann series: beta^{-1} sum_{k=0}^{order} (T^k) v
    expected = np.zeros_like(v)
    Tk_v = v.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 1.5, 0.1, 3),
    ],
)
def test_neumann_preconditioner_cached_expression(
    beta,
    gamma,
    h,
    order,
    neumann_cached_factory,
    neumann_cached_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
):
    """Validate cached Neumann preconditioner with stored auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    pre = neumann_cached_factory(beta, gamma, order)
    kernel = neumann_cached_kernel(prepare, pre, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.7, -1.3], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        vec,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    beta_inv = precision(1.0) / beta_val
    h_val = precision(h)
    T = (gamma_val * beta_inv) * h_val * jacobian

    expected = np.zeros(state_len, dtype=precision)
    Tk_v = vec.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.fixture(scope="function")
def residual_system():
    """Linear system with constant Jacobian for residual tests."""

    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    system.build()
    return system


@pytest.fixture(scope="function")
def stage_residual_factory(residual_system, precision):
    def factory(beta, gamma, a_ii, M):
        base = cuda.to_device(np.array([0.25, -0.25], dtype=precision))
        fname = f"stage_residual_factory_{abs(hash(M.tobytes()))}"
        code = generate_stage_residual_code(
            residual_system.equations,
            residual_system.indices,
            M=M,
            func_name=fname,
        )
        res_fac = residual_system.gen_file.import_function(fname, code)
        return res_fac(
            residual_system.constants.values_dict,
            from_dtype(residual_system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def residual_kernel(precision):
    n = 2

    def make_kernel(residual):
        @cuda.jit
        def kernel(t, h, aij, vec, base_state, out):
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            residual(vec, parameters, drivers, t, h, aij, base_state, out)

        return kernel

    return make_kernel


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,a_ii,M",
    [
        (1.0, 1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, 0.5, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, 0.25, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_stage_residual(
    beta,
    gamma,
    h,
    a_ii,
    M,
    stage_residual_factory,
    residual_kernel,
    precision,
    tolerance,
):
    residual = stage_residual_factory(beta, gamma, a_ii, M)
    kernel = residual_kernel(residual)
    stage = np.array([0.5, -0.3], dtype=precision)
    base = np.array([0.25, -0.25], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(0.0), precision(h), precision(a_ii), stage, base, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    eval_point = base + a_ii * stage
    expected = beta * (M @ stage) - gamma * h * (J @ eval_point)
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )
