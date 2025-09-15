"""Reusable tester for IVP loop algorithms."""

from typing import Any, Tuple

import numpy as np
import pytest
from numba import cuda
from numba import from_dtype

from numpy.testing import assert_allclose

from cubie import is_devfunc
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.step_control.fixed_step_controller import (
    FixedStepController,
)
from cubie.integrators.step_control.adaptive_I_controller import AdaptiveIController
from cubie.integrators.step_control.adaptive_PI_controller import AdaptivePIController
from cubie.integrators.step_control.adaptive_PID_controller import AdaptivePIDController
from cubie.integrators.step_control.gustafsson_controller import GustafssonController
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from cubie.outputhandling import LoopBufferSizes, OutputCompileFlags


@cuda.jit(device=True, inline=True)
def _save_state(state, obs, state_out, obs_out, t):
    """Store current state in the first output row."""
    state_out[0] = state[0]
    state_out[1] = state[1]


@cuda.jit(device=True, inline=True)
def _update_summaries(state, obs, state_sum, obs_sum, saves):
    """No-op summary updater."""
    return


@cuda.jit(device=True, inline=True)
def _save_summaries(state_sum, obs_sum, state_out, obs_out, saves):
    """No-op summary saver."""
    return


class ODELoopTester:
    """Base class for testing :class:`IVPLoop` instances."""

    # ------------------------------------------------------------------
    # System fixture
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def system(self, precision) -> Any:
        """Two-state linear symbolic system."""

        dxdt = [
            "dx0 = -a*x0 + c0",
            "dx1 = -b*x1 + c1",
        ]
        params = {"a": 1.0, "b": 0.5}
        consts = {"c0": 1.0, "c1": -0.5}
        system = create_ODE_system(
            dxdt,
            states=["x0", "x1"],
            parameters=params,
            constants=consts,
            precision=precision,
        )
        return system

    @pytest.fixture(scope="function")
    def initial_state(self) -> np.ndarray:
        """Initial state vector for loop execution."""

        return np.array([2.0, 1.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Loop construction fixtures
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def buffer_sizes(self, system) -> LoopBufferSizes:
        """Minimal buffer sizes derived from the system."""

        return LoopBufferSizes(
            state=system.sizes.states,
            observables=1,
            dxdt=system.sizes.states,
            parameters=system.sizes.parameters,
            drivers=system.num_drivers,
        )

    @pytest.fixture(scope="function")
    def compile_flags(self) -> OutputCompileFlags:
        """Enable state saving only."""

        return OutputCompileFlags(save_state=True)

    @pytest.fixture(scope="function", params=[
        "fixed",
        "I",
        "PI",
        "PID",
        "gustafsson"
    ])
    def step_controller(self, request, precision):
        """Provide different step controllers based on parameter."""

        controller_type = request.param
        precision = np.float32

        if controller_type == "fixed":
            ctrl = FixedStepController(precision, 0.1)
        elif controller_type == "I":
            ctrl = AdaptiveIController(precision, dt_min=0.01, dt_max=1.0, n=1)
        elif controller_type == "PI":
            ctrl = AdaptivePIController(precision, dt_min=0.01, dt_max=1.0, n=1)
        elif controller_type == "PID":
            ctrl = AdaptivePIDController(precision, dt_min=0.01, dt_max=1.0, n=1)
        elif controller_type == "gustafsson":
            ctrl = GustafssonController(precision, dt_min=0.01, dt_max=1.0, n=1)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        return ctrl

    @pytest.fixture(scope="function")
    def loop(
        self,
        system,
        step_object,
        buffer_sizes,
        compile_flags,
        step_controller,
    ) -> IVPLoop:
        """Construct the loop under test."""
        loop = IVPLoop(
            precision=np.float32,
            dt_save=0.1,
            dt_summarise=0.2,
            step_controller=step_controller,
            step_object=step_object,
            buffer_sizes=buffer_sizes,
            compile_flags=compile_flags,
            save_state_func=_save_state,
            update_summaries_func=_update_summaries,
            save_summaries_func=_save_summaries,
        )
        return loop


    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------
    def _run_loop(
        self, loop: IVPLoop, system, init: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Execute the built loop on a single thread."""
        precision = np.float32
        numba_precision = from_dtype(precision)
        params = np.array(
            list(system.parameters.values_dict.values()), dtype=precision
        )
        drivers = np.zeros((1,system.num_drivers), dtype=precision)
        state_output = np.zeros((1, loop.algorithm.compile_settings.n),
                                 dtype=precision)
        obs_output = np.zeros((1, 1), dtype=precision)
        state_summary = np.zeros((1, 1), dtype=precision)
        obs_summary = np.zeros((1, 1), dtype=precision)
        status = np.zeros(1, dtype=np.int32)

        d_init = cuda.to_device(init)
        d_params = cuda.to_device(params)
        d_drivers = cuda.to_device(drivers)
        d_state_out = cuda.to_device(state_output)
        d_obs_out = cuda.to_device(obs_output)
        d_state_sum = cuda.to_device(state_summary)
        d_obs_sum = cuda.to_device(obs_summary)
        d_status = cuda.to_device(status)

        shared_elems = loop.buffer_indices.local_end
        shared_bytes = precision().itemsize * shared_elems
        n = loop.algorithm.compile_settings.n
        local_req = (
            n
            + 3
            + loop.step_controller.local_memory_required
            + loop.algorithm.persistent_local_required
        )
        loop_fn = loop.device_function

        @cuda.jit
        def kernel(init, params, drivers, state_out, obs_out, state_sum, obs_sum, status_arr):
            shared = cuda.shared.array(0, dtype=numba_precision)
            local = cuda.local.array(local_req, dtype=numba_precision)
            status_arr[0] = loop_fn(
                init,
                params,
                drivers,
                shared,
                local,
                state_out,
                obs_out,
                state_sum,
                obs_sum,
                precision(0.1),
                precision(0.0),
            )

        kernel[1, 1, 0, shared_bytes](
            d_init,
            d_params,
            d_drivers,
            d_state_out,
            d_obs_out,
            d_state_sum,
            d_obs_sum,
            d_status,
        )
        cuda.synchronize()
        return d_state_out.copy_to_host()[0], int(d_status.copy_to_host()[0])

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_instantiation_and_build(self, loop: IVPLoop) -> None:
        loop_fn = loop.device_function
        assert is_devfunc(loop_fn)

    def test_numerical_kernel(
        self,
        loop: IVPLoop,
        expected_state: np.ndarray,
        system,
        initial_state: np.ndarray,
    ) -> None:
        state, status = self._run_loop(loop, system, initial_state)
        assert status == 0
        assert_allclose(state, expected_state, rtol=1e-2, atol=1e-2)

    def test_getters(self, loop: IVPLoop, buffer_sizes: LoopBufferSizes) -> None:
        assert loop.dt0 == pytest.approx(0.1)
        assert loop.dt_min == pytest.approx(0.1)
        assert loop.dt_max == pytest.approx(0.1)
        assert loop.dt_save == pytest.approx(0.1)
        assert loop.dt_summarise == pytest.approx(0.1)
        assert loop.precision is np.float32
        assert not loop.is_adaptive
        idx = loop.buffer_indices
        assert idx.state.stop - idx.state.start == buffer_sizes.state
        assert loop.shared_memory_indices is not None

    @pytest.fixture(scope="function")
    def algorithm_update_param(self):
        return {}

    def test_update(self, loop: IVPLoop, algorithm_update_param: dict) -> None:
        params = {"dt_max": 0.05, "dt_save": 0.05}
        params.update(algorithm_update_param)
        recognised = loop.update(params)
        expected_keys = set(params.keys())
        assert expected_keys.issubset(recognised)
        assert loop.dt_max == pytest.approx(0.05)

