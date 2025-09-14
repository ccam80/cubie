import numpy as np
from numba import cuda

from cubie.integrators.algorithms_ import (
    ExplicitEulerStep,
    BackwardsEulerStep,
    ExplicitStepConfig,
    ImplicitStepConfig,
)


@cuda.jit(device=True, inline=True)
def _dxdt(state, parameters, drivers, observables, out):
    out[0] = -state[0]


def _run_step(step, precision):
    device_func = step.build()
    n = step.compile_settings.n
    state = np.array([1.0], dtype=precision)
    params = np.empty(0, dtype=precision)
    drivers = np.empty(0, dtype=precision)
    observables = np.empty(0, dtype=precision)
    out = np.zeros(n, dtype=precision)
    shared = np.zeros(step.shared_memory_required, dtype=precision)
    local = np.zeros(step.local_memory_required, dtype=precision)

    @cuda.jit
    def kernel(state, params, drivers, observables, out, shared, local):
        device_func(state, params, drivers, observables, out, shared, local)

    kernel[1, 1](state, params, drivers, observables, out, shared, local)
    return state


def test_explicit_euler_step(precision):
    config = ExplicitStepConfig(
        precision=precision, n=1, dt=float(precision(0.1)), dxdt_function=_dxdt
    )
    step = ExplicitEulerStep(config)
    state = _run_step(step, precision)
    assert np.allclose(state[0], precision(0.9))


def test_backwards_euler_step(precision):
    config = ImplicitStepConfig(
        precision=precision, n=1, dt=float(precision(0.1)), dxdt_function=_dxdt
    )
    step = BackwardsEulerStep(config)
    state = _run_step(step, precision)
    expected = precision(1.0 / (1.0 + 0.1))
    assert np.allclose(state[0], expected, rtol=1e-2)
