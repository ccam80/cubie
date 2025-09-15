import numpy as np
import pytest
from numba import cuda
from math import log

from cubie import create_ODE_system
from cubie.integrators.algorithms_ import (
    ExplicitEulerStep,
    BackwardsEulerStep,
)

# System fixtures
@pytest.fixture(scope="function")
def linearsystem(precision):
    """Linear system extended to 2 states to match nonlinear system size"""
    sys = create_ODE_system(precision=precision,
                            dxdt=["dx0 = -1.0 * x0",
                                  "dx1 = -0.5 * x1"])
    return sys

@pytest.fixture(scope="function")
def nonlinearsystem(precision):
    """Nonlinear system with 2 states"""
    sys = create_ODE_system(precision=precision,
                            dxdt=["dx0 = -1.0 * x0**2",
                                  "dx1 = 2.0 * x0*x1**2"])
    return sys

# System type parametrization
@pytest.fixture(scope="function", params=["linear", "nonlinear"])
def system_type(request):
    """Parametrize system type"""
    return request.param

@pytest.fixture(scope="function")
def stepsystem(system_type, precision, linearsystem, nonlinearsystem):
    """Get the appropriate system based on system_type parameter"""
    if system_type == "nonlinear":
        return nonlinearsystem
    else:
        return linearsystem

@pytest.fixture(scope="function")
def step_size():
    """Default step size"""
    return 0.1

# Step method parametrization
@pytest.fixture(scope="function", params=["explicit", "implicit"])
def step_method(request):
    """Parametrize step method type"""
    return request.param

@pytest.fixture(scope="function")
def step_obj(step_method, precision, stepsystem, step_size):
    """Get the appropriate step object based on step_method parameter"""
    dxdt = stepsystem.dxdt_function

    if step_method == "explicit":
        return ExplicitEulerStep(dxdt_function=dxdt,
                               precision=precision,
                               n=stepsystem.sizes.states,
                               step_size=step_size)
    else:  # implicit
        get_helpers = stepsystem.get_solver_helper
        return BackwardsEulerStep(dxdt_function=dxdt,
                                get_solver_helper_fn=get_helpers,
                                precision=precision,
                                n=stepsystem.sizes.states,
                                atol=1e-6,
                                rtol=1e-6,
                                preconditioner_order=2,)

# Starting state parametrization
@pytest.fixture(scope="function", params=[
    [2.0, 0.5],    # alternative starting state
])
def x0(request):
    """Parametrize starting states for both systems"""
    return np.array(request.param)

# Expected solution functions
def linear_expected(x0, t):
    """Expected solution for linear system"""
    x_0 = x0[0] * np.exp(-1.0 * t)
    x_1 = x0[1] * np.exp(-0.5 * t)
    return np.array([x_0, x_1])

def nonlinear_expected(x0, t):
    """Expected solution for nonlinear system"""
    x_0 = x0[0] / (1 + x0[0] * t)
    x_1 = 1 / (1/x0[1] - 2*log(1 + x0[0] * t))
    return np.array([x_0, x_1])

@pytest.fixture(scope="function")
def expected_solution(system_type, x0, step_method, step_size):
    """Get expected solution based on system type and parameters"""


    if system_type == "linear":
        return linear_expected(x0, step_size)
    else:  # nonlinear
        return nonlinear_expected(x0, step_size)

def _run_step(step_obj, precision, x0, step_size):
    """Run a single step with given initial conditions"""
    step_obj.build()
    step_fn = step_obj.step_function
    n = step_obj.compile_settings.n
    state = np.array(x0[:n], dtype=precision)
    params = np.empty(1, dtype=precision)
    drivers = np.empty(1, dtype=precision)
    observables = np.empty(1, dtype=precision)
    dxdt_buffer = np.zeros(n, dtype=precision)
    local_req = max(1, step_obj.persistent_local_required)
    sharedmem = max(1, step_obj.shared_memory_required)
    flag = np.ones(1, dtype=np.int32) * -1
    @cuda.jit
    def kernel(state, params, drivers, observables, dxdt_buffer, flag):
        shared = cuda.shared.array(0, dtype=precision)
        local = cuda.local.array(local_req, dtype=precision)
        dt = cuda.local.array(1, dtype=precision)
        dt[0] = step_size
        error = cuda.local.array(n, dtype=precision)
        flag[0] = step_fn(state, params, drivers, observables, dxdt_buffer,
                error, dt, shared, local)

    kernel[1, 1, 0, sharedmem](state, params, drivers, observables,
                               dxdt_buffer, flag)
    return state, flag

# Updated tests using new parametrized fixtures
def test_step_methods(step_obj, precision, x0, expected_solution,
                      system_type, step_method, step_size):
    """Test step methods with parametrized systems, methods, and initial conditions"""
    state, flag = _run_step(step_obj, precision, x0, step_size)

    # For explicit methods, we can compare against analytical solutions
    if system_type == "linear":
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol=1e-1
        atol=1e-1

    assert flag[0] == 0
    assert np.allclose(state, expected_solution, rtol=rtol, atol=atol)
