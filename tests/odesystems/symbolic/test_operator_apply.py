import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from cubie.odesystems.symbolic.operator_apply import (
    generate_operator_apply_code,
    generate_neumann_preconditioner_code,
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

    def factory(beta, gamma, M):
        fname = f"operator_apply_factory_{abs(hash(M.tobytes()))}"
        code = generate_operator_apply_code(
            operator_system.equations, operator_system.indices, M=M, func_name=fname
        )
        op_fac = operator_system.gen_file.import_function(fname, code)
        return op_fac(
            operator_system.constants.values_dict,
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
        def kernel(h, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            op(state, parameters, drivers, h, vec, out)

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
def test_operator_apply_dense(beta, gamma, h, M, operator_factory, operator_kernel, precision):
    """Evaluate operator_apply for specific scalings and mass matrices."""

    op = operator_factory(beta, gamma, M)
    kernel = operator_kernel(op)
    v = np.array([1.0, -1.0], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(h), v, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ v - gamma * h * J @ v
    assert np.allclose(out, expected, atol=1e-6)


def test_operator_apply_constant_unpacking(operator_system):
    """Ensure constants are defined as individual variables."""
    code = generate_operator_apply_code(
        operator_system.equations, operator_system.indices
    )
    assert "a = precision(constants['a'])" in code


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
    """Kernel applying the Neumann preconditioner to a vector, passing scratch."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(h, vec, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            scratch = cuda.local.array(n, precision)
            pre(state, parameters, drivers, h, vec, out, scratch)

        return kernel

    return make_kernel


@pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
@pytest.mark.parametrize("beta,gamma,h,order", [
    (1.0, 1.0, 0.25, 0),
    (1.0, 1.0, 0.25, 1),
    (1.0, 1.0, 0.25, 2),
    (0.5, 2.0, 0.1, 3),
])
def test_neumann_preconditioner_expression(beta, gamma, h, order, neumann_factory, neumann_kernel, precision):
    """Validate Neumann preconditioner equals truncated series on a known system.

    System: dx/dt = J x with J = [[a, b], [c, d]] = [[1, 2], [3, 4]].
    Preconditioner approximates (beta*I - gamma*h*J)^{-1} via truncated series.
    """
    pre = neumann_factory(beta, gamma, order)
    kernel = neumann_kernel(pre)

    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)

    kernel[1, 1](precision(h), v, out)

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

    assert np.allclose(out, expected, atol=1e-6)

