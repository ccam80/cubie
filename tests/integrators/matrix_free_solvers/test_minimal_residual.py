import numpy as np
import pytest
from numba import cuda



@pytest.fixture(scope="function")
def mr_factory_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def mr_factory_settings(mr_factory_settings_override):
    settings = {"tolerance": 1e-6, "max_iters": 64}
    settings.update(mr_factory_settings_override)
    return settings


@pytest.fixture(scope="function")
def mr_solver_device(mr_factory_settings):
    return minimal_residual_solver_factory(**mr_factory_settings)


@pytest.mark.parametrize(
    "system_setup",
    ["linear"],
    indirect=True,
)
def test_minimal_residual_solver(system_setup, mr_solver_device, precision):
    n = system_setup["n"]
    jvp = system_setup["jvp"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]

    @cuda.jit
    def kernel(x, rhs):
        residual = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        precond_temp = cuda.local.array(n, precision)
        state = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        for i in range(n):
            state[i] = expected[i]
            x[i] = precision(0.0)
        mr_solver_device(
            jvp,
            state,
            parameters,
            drivers,
            h,
            rhs,
            x,
            residual,
            z_vec,
            v_vec,
            None,
            precond_temp,
        )

    x = cuda.to_device(np.zeros(n, dtype=precision))
    rhs = cuda.to_device(rhs_vec)
    kernel[1, 1](x, rhs)
    assert np.allclose(x.copy_to_host(), expected, atol=1e-4)
