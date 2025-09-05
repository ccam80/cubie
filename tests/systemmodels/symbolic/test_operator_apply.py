import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.systemmodels.symbolic.symbolicODE import create_ODE_system
from cubie.systemmodels.symbolic.operator_apply import (
    generate_operator_apply_code,
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


@pytest.fixture(scope="function")
def operator_factory(operator_system, precision):
    """Return a factory producing operator_apply device functions."""

    def factory(beta, gamma):
        code = generate_operator_apply_code(
            operator_system.equations, operator_system.indices
        )
        op_fac = operator_system.gen_file.import_function(
            "operator_apply_factory", code
        )
        return op_fac(
            operator_system.constants.values_array,
            from_dtype(operator_system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="function")
def operator_kernel(precision):
    """Kernel applying operator_apply to a vector."""

    n = 2

    def make_kernel(op):
        @cuda.jit
        def kernel(h, M, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            op(state, parameters, drivers, h, M, vec, out)

        return kernel

    return make_kernel


@pytest.mark.parametrize(
    "beta,gamma,M",
    [
        (1.0, 1.0, np.eye(2)),
        (0.5, 2.0, np.diag([2.0, 3.0])),
        (0.0, 1.0, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_operator_apply_dense(
    beta, gamma, M, operator_factory, operator_kernel, precision
):
    """Evaluate operator_apply for several scalings and mass matrices."""

    op = operator_factory(beta, gamma)
    kernel = operator_kernel(op)
    v = np.array([1.0, -1.0], dtype=precision)
    h = precision(0.25)
    M_dev = cuda.to_device(M.astype(precision))
    v_dev = cuda.to_device(v)
    out_dev = cuda.device_array(2, precision)
    kernel[1, 1](h, M_dev, v_dev, out_dev)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ v - gamma * h * J @ v
    assert np.allclose(out_dev.copy_to_host(), expected, atol=1e-6)
