import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.odesystems.symbolic.operator_apply import (
    generate_residual_end_state_code,
    generate_stage_residual_code,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


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
def end_residual_factory(residual_system, precision):
    def factory(beta, gamma, M):
        base = cuda.to_device(np.array([0.5, -0.5], dtype=precision))
        fname = f"residual_end_factory_{abs(hash(M.tobytes()))}"
        code = generate_residual_end_state_code(
            residual_system.indices, M=M, func_name=fname
        )
        res_fac = residual_system.gen_file.import_function(fname, code)
        return res_fac(
            residual_system.constants.values_dict,
            from_dtype(residual_system.precision),
            residual_system.dxdt_function,
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def stage_residual_factory(residual_system, precision):
    def factory(beta, gamma, a_ii, M):
        base = cuda.to_device(np.array([0.25, -0.25], dtype=precision))
        fname = f"stage_residual_factory_{abs(hash(M.tobytes()))}"
        code = generate_stage_residual_code(
            residual_system.indices, M=M, func_name=fname
        )
        res_fac = residual_system.gen_file.import_function(fname, code)
        return res_fac(
            residual_system.constants.values_dict,
            from_dtype(residual_system.precision),
            residual_system.dxdt_function,
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def residual_kernel(precision):
    n = 2

    def make_kernel(residual):
        @cuda.jit
        def kernel(h, aij, vec, base_state, out):
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            tmp = cuda.local.array(n, precision)
            residual(vec, parameters, drivers, h, aij, base_state, tmp, out)

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
def test_residual_end_state(beta, gamma, h, M, end_residual_factory, residual_kernel, precision):
    residual = end_residual_factory(beta, gamma, M)
    kernel = residual_kernel(residual)
    state = np.array([1.0, -1.0], dtype=precision)
    base = np.array([0.5, -0.5], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(h), precision(1.0), state, base, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ (state - base) - gamma * h * (J @ state)
    assert np.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize(
    "beta,gamma,h,a_ii,M",
    [
        (1.0, 1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, 0.5, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, 0.25, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_stage_residual(beta, gamma, h, a_ii, M, stage_residual_factory, residual_kernel, precision):
    residual = stage_residual_factory(beta, gamma, a_ii, M)
    kernel = residual_kernel(residual)
    stage = np.array([0.5, -0.3], dtype=precision)
    base = np.array([0.25, -0.25], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(h), precision(a_ii), stage, base, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    eval_point = base + a_ii * stage
    expected = beta * (M @ stage) - gamma * h * (J @ eval_point)
    assert np.allclose(out, expected, atol=1e-6)
