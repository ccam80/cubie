"""Tests for building integrator loops with development steps."""

from typing import Callable

import numpy as np
import pytest
from numba import cuda

from cubie.integrators.algorithms_ import (
    BackwardsEulerStep,
    ExplicitEulerStep,
    ExplicitStepConfig,
)
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.step_control.fixed_step_controller import (
    FixedStepController,
)
from cubie.outputhandling import LoopBufferSizes, OutputCompileFlags


@cuda.jit(device=True, inline=True)
def _dxdt(state, params, drivers, observables, out):
    """Simple linear derivative."""
    out[0] = -state[0]


@cuda.jit(device=True, inline=True)
def _save_state(state, obs, state_out, obs_out, t):
    """Placeholder state-saving function."""
    return


@cuda.jit(device=True, inline=True)
def _update_summaries(state, obs, state_sum, obs_sum, saves):
    """Placeholder summary update."""
    return


@cuda.jit(device=True, inline=True)
def _save_summaries(state_sum, obs_sum, state_out, obs_out, saves):
    """Placeholder summary save."""
    return


def _dummy_helper(name: str, **kwargs: int) -> Callable:
    """Return no-op device helpers for implicit solvers."""

    @cuda.jit(device=True, inline=True)
    def _linear_operator(state, params, drivers, h, inp, out):
        return

    @cuda.jit(device=True, inline=True)
    def _residual(state, params, drivers, h, a_ij, base_state, out):
        return

    @cuda.jit(device=True, inline=True)
    def _preconditioner(state, params, drivers, h, residual, z, scratch):
        return

    mapping = {
        "linear_operator": _linear_operator,
        "end_residual": _residual,
        "stage_residual": _residual,
        "neumann_preconditioner": _preconditioner,
    }
    return mapping[name]


@pytest.fixture(scope="function")
def buffer_sizes() -> LoopBufferSizes:
    """Provide minimal buffer sizes for loop construction."""
    return LoopBufferSizes(state=1, observables=1, dxdt=1,
                           parameters=1, drivers=1)


@pytest.fixture(scope="function")
def compile_flags() -> OutputCompileFlags:
    """Disable optional output features."""
    return OutputCompileFlags()


@pytest.fixture(scope="function")
def controller() -> FixedStepController:
    """Fixed-step controller for tests."""
    return FixedStepController(np.float32, 0.01)


def _build_loop(step_obj: object,
                buffer_sizes: LoopBufferSizes,
                compile_flags: OutputCompileFlags,
                controller: FixedStepController) -> Callable:
    """Construct an IVPLoop and return its device function."""
    loop = IVPLoop(
        precision=np.float32,
        dt_save=0.01,
        dt_summarise=0.1,
        step_controller=controller,
        step_object=step_obj,
        buffer_sizes=buffer_sizes,
        compile_flags=compile_flags,
        save_state_func=_save_state,
        update_summaries_func=_update_summaries,
        save_summaries_func=_save_summaries,
    )
    return loop.build()


def test_loop_builds(buffer_sizes: LoopBufferSizes,
                     compile_flags: OutputCompileFlags,
                     controller: FixedStepController) -> None:
    """Build loops for explicit and backward Euler steps."""
    explicit_cfg = ExplicitStepConfig(
        precision=np.float32,
        buffer_sizes=buffer_sizes,
        dxdt_fn=_dxdt,
        fixed_step_size=0.01,
    )
    explicit_step = ExplicitEulerStep(explicit_cfg)
    _build_loop(explicit_step, buffer_sizes, compile_flags, controller)

    backward_step = BackwardsEulerStep(np.float32, 1)
    backward_step.update(
        dxdt_fn=_dxdt,
        get_solver_helper=_dummy_helper,
        buffer_sizes=buffer_sizes,
        precision=np.float32,
    )
    _build_loop(backward_step, buffer_sizes, compile_flags, controller)
