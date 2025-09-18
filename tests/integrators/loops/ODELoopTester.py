"""Reusable tester for IVP loop algorithms."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from cubie import is_devfunc
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.step_control.adaptive_I_controller import AdaptiveIController
from cubie.integrators.step_control.adaptive_PI_controller import AdaptivePIController
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)
from cubie.integrators.step_control.fixed_step_controller import FixedStepController
from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonController,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from cubie.odesystems.systems.threeCM import (
    default_initial_values as threecm_initial_values,
    default_observable_names as threecm_observables,
    default_parameters as threecm_parameters,
)
from cubie.outputhandling.output_sizes import LoopBufferSizes, OutputArrayHeights
from tests.integrators.cpu_reference import run_reference_loop


THREECM_EQUATIONS = [
    "P_a = E_a * V_a",
    "P_v = E_v * V_v",
    "P_h = E_h * V_h * d1",
    "Q_i = (P_v - P_h) / R_i if P_v > P_h else 0",
    "Q_o = (P_h - P_a) / R_o if P_h > P_a else 0",
    "Q_c = (P_a - P_v) / R_c",
    "dV_h = Q_i - Q_o",
    "dV_a = Q_o - Q_c",
    "dV_v = Q_c - Q_i",
]

DECAYS_EQUATIONS = [
    "dx0 = -x0",
    "dx1 = -x1/2",
    "dx2 = -x2/3",
    "o0 = dx0 * p0 + c0 + d0",
    "o1 = dx1 * p1 + c1 + d0",
    "o2 = dx2 * p2 + c2 + d0",
]

DECAYS_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
DECAYS_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
DECAYS_CONSTANTS = {"c0": 0.0, "c1": 1.0, "c2": 2.0}
DECAYS_OBSERVABLES = ["o0", "o1", "o2"]

@pytest.fixture(scope="function")
def expected_answer(
    system,
    loop_compile_settings,
    solverkernel,
    inputs,
    precision,
    output_functions,
):
    dt = loop_compile_settings["dt_min"]
    output_dt = loop_compile_settings["dt_save"]
    warmup = solverkernel.warmup
    duration = solverkernel.duration
    solver_settings = {
        "dt_min": dt,
        "dt_max": loop_compile_settings["dt_max"],
        "dt_save": output_dt,
        "dt_summarise": loop_compile_settings["dt_summarise"],
        "warmup": warmup,
        "duration": duration,
        "atol": loop_compile_settings["atol"],
        "rtol": loop_compile_settings["rtol"],
    }

    result = run_reference_loop(
        system=system,
        inputs=inputs,
        solver_settings=solver_settings,
        loop_compile_settings=loop_compile_settings,
        output_functions=output_functions,
        stepper="explicit_euler",
        step_controller_settings={"kind": "fixed", "dt": dt},
    )

    return result["state"], result["observables"]

class ODELoopTester:
    """Base class for testing :class:`IVPLoop` instances."""


    @pytest.fixture(scope="function")
    def linear_system(self, precision: np.dtype):
        """Two-state linear symbolic system."""

        dxdt = ["dx0 = -a*x0 + c0", "dx1 = -b*x1 + c1"]
        states = {"x0": 2.0, "x1": 1.0}
        params = {"a": 1.0, "b": 0.5}
        consts = {"c0": 1.0, "c1": -0.5}
        system = create_ODE_system(
            dxdt=dxdt,
            states=states,
            parameters=params,
            constants=consts,
            precision=precision,
            name="symbolic_linear_system",
        )
        system.build()
        return system

    @pytest.fixture(scope="function")
    def nonlinear_system(self, precision: np.dtype):
        """Nonlinear two-state symbolic system."""

        dxdt = ["dx0 = -a*x0**2", "dx1 = b*x0*x1**2"]
        states = {"x0": 1.0, "x1": 0.5}
        params = {"a": 1.0}
        consts = {"b": 2.0}
        system = create_ODE_system(
            dxdt=dxdt,
            states=states,
            parameters=params,
            constants=consts,
            precision=precision,
            name="symbolic_nonlinear_system",
        )
        system.build()
        return system

    @pytest.fixture(scope="function")
    def symbolic_threecm(self, precision: np.dtype):
        """Symbolic version of the three chamber cardiovascular model."""

        system = create_ODE_system(
            dxdt=THREECM_EQUATIONS,
            states=threecm_initial_values,
            parameters=threecm_parameters,
            observables=threecm_observables,
            drivers=["d1"],
            precision=precision,
            name="symbolic_threecm",
            strict=True,
        )
        system.build()
        return system

    @pytest.fixture(scope="function")
    def symbolic_decays(self, precision: np.dtype):
        """Symbolic exponential decay system."""

        system = create_ODE_system(
            dxdt=DECAYS_EQUATIONS,
            states=DECAYS_STATES,
            parameters=DECAYS_PARAMETERS,
            constants=DECAYS_CONSTANTS,
            observables=DECAYS_OBSERVABLES,
            drivers=["d0"],
            precision=precision,
            name="symbolic_decays",
        )
        system.build()
        return system

    @pytest.fixture(scope="function")
    def system_override(self, request) -> str:
        """Optional override for the system fixture."""

        return request.param if hasattr(request, "param") else "linear"

    @pytest.fixture(scope="function")
    def system(
        self,
        system_override: str,
        linear_system,
        nonlinear_system,
        symbolic_threecm,
        symbolic_decays,
    ):
        """Select the system based on the override parameter."""

        mapping = {
            "linear": linear_system,
            "nonlinear": nonlinear_system,
            "threecm": symbolic_threecm,
            "decays": symbolic_decays,
        }
        if system_override not in mapping:
            raise ValueError(f"Unknown system override '{system_override}'.")
        return mapping[system_override]

    @pytest.fixture(scope="function")
    def initial_state(
        self,
        system,
        precision: np.dtype,
    ) -> NDArray[np.floating]:
        """Initial state vector for loop execution."""

        return system.initial_values.values_array.astype(precision, copy=True)

    # ------------------------------------------------------------------
    # Loop construction fixtures
    # ------------------------------------------------------------------
    @pytest.fixture(scope="function")
    def buffer_sizes(
        self,
        system,
        output_functions,
    ) -> LoopBufferSizes:
        """Loop buffer sizes derived from the system and output functions."""

        return LoopBufferSizes.from_system_and_output_fns(system, output_functions)

    @pytest.fixture(scope="function")
    def step_controller_settings_override(self, request) -> Dict[str, Any]:
        """Optional override for the step controller configuration."""

        return request.param if hasattr(request, "param") else {}

    @pytest.fixture(scope="function")
    def step_controller_settings(
        self,
        loop_compile_settings: Dict[str, Any],
        system,
        step_controller_settings_override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Base configuration for step controllers."""

        defaults: Dict[str, Any] = {
            "kind": "fixed",
            "dt_min": loop_compile_settings["dt_min"],
            "dt_max": loop_compile_settings["dt_max"],
            "atol": loop_compile_settings["atol"],
            "rtol": loop_compile_settings["rtol"],
            "order": 1,
            "n": system.sizes.states,
        }
        defaults.update(step_controller_settings_override)
        return defaults

    @pytest.fixture(scope="function")
    def step_controller(
        self,
        precision: np.dtype,
        step_controller_settings: Dict[str, Any],
    ):
        """Instantiate the requested step controller."""

        kind = step_controller_settings["kind"]
        dt_min = step_controller_settings["dt_min"]
        dt_max = step_controller_settings["dt_max"]
        atol = step_controller_settings["atol"]
        rtol = step_controller_settings["rtol"]
        order = step_controller_settings.get("order", 1)
        n_states = step_controller_settings["n"]

        if kind == "fixed":
            return FixedStepController(precision, dt_min)
        if kind == "I":
            return AdaptiveIController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
            )
        if kind == "PI":
            return AdaptivePIController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
            )
        if kind == "PID":
            return AdaptivePIDController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
            )
        if kind == "gustafsson":
            return GustafssonController(
                precision,
                dt_min=dt_min,
                dt_max=dt_max,
                atol=atol,
                rtol=rtol,
                algorithm_order=order,
                n=n_states,
            )
        raise ValueError(f"Unknown controller type '{kind}'.")

    @pytest.fixture(scope="function")
    def loop(
        self,
        precision: np.dtype,
        system,
        step_object,
        buffer_sizes: LoopBufferSizes,
        output_functions,
        step_controller,
        loop_compile_settings: Dict[str, Any],
    ) -> IVPLoop:
        """Construct the loop under test."""

        return IVPLoop(
            precision=precision,
            dt_save=loop_compile_settings["dt_save"],
            dt_summarise=loop_compile_settings["dt_summarise"],
            step_controller=step_controller,
            step_object=step_object,
            buffer_sizes=buffer_sizes,
            compile_flags=output_functions.compile_flags,
            save_state_func=output_functions.save_state_func,
            update_summaries_func=output_functions.update_summaries_func,
            save_summaries_func=output_functions.save_summary_metrics_func,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _driver_sequence(
        self,
        duration: float,
        dt_save: float,
        n_drivers: int,
        precision: np.dtype,
    ) -> NDArray[np.floating]:
        """Generate a deterministic driver sequence."""

        samples = max(int(np.ceil(duration / dt_save)), 1)
        width = max(n_drivers, 1)
        drivers = np.zeros((samples, width), dtype=precision)
        if n_drivers > 0 and duration > 0.0:
            times = np.linspace(0.0, duration, samples, dtype=float)
            for idx in range(n_drivers):
                drivers[:, idx] = precision(
                    0.5
                    * (1.0 + np.sin(2 * np.pi * (idx + 1) * times / duration))
                )
        return drivers

    def _reference_inputs(
        self,
        system,
        initial_state: NDArray[np.floating],
        solver_config: Dict[str, float],
    ) -> Dict[str, NDArray[np.floating]]:
        """Build inputs for the CPU reference helpers."""

        precision = system.precision
        dt_min = float(solver_config["dt_min"])
        duration = float(solver_config["duration"])
        warmup = float(solver_config.get("warmup", 0.0))
        total_time = max(duration + warmup, dt_min)
        samples = max(int(np.ceil(total_time / dt_min)), 1)
        n_drivers = system.num_drivers
        forcing = np.zeros((n_drivers, samples), dtype=precision)
        if n_drivers > 0 and total_time > 0.0:
            times = np.linspace(0.0, total_time, samples, dtype=float)
            for idx in range(n_drivers):
                forcing[idx, :] = precision(
                    0.5
                    * (
                        1.0
                        + np.sin(2 * np.pi * (idx + 1) * times / total_time)
                    )
                )

        return {
            "initial_values": np.array(initial_state, dtype=precision, copy=True),
            "parameters": np.array(
                system.parameters.values_array, dtype=precision, copy=True
            ),
            "forcing_vectors": forcing,
        }

    def _reference_controller_settings(
        self,
        controller_settings: Dict[str, Any],
        solver_config: Dict[str, float],
    ) -> Dict[str, float | int | str]:
        """Translate controller settings for the CPU reference loop."""

        dt_initial = controller_settings.get(
            "dt",
            controller_settings.get("dt_min", solver_config["dt_min"]),
        )
        dt_max = controller_settings.get("dt_max", solver_config["dt_max"])
        return {
            "kind": str(controller_settings.get("kind", "fixed")),
            "dt": float(dt_initial),
            "dt_max": float(dt_max),
            "order": int(controller_settings.get("order", 1)),
        }

    def _reference_state_output(
        self,
        *,
        system,
        initial_state: NDArray[np.floating],
        loop_compile_settings: Dict[str, Any],
        solver_settings: Dict[str, Any],
        output_functions,
        stepper: str,
        step_controller_settings: Dict[str, Any],
    ) -> NDArray[np.floating]:
        """Execute the CPU reference loop and return saved states."""

        solver_config = {
            "dt_min": float(solver_settings["dt_min"]),
            "dt_max": float(solver_settings["dt_max"]),
            "dt_save": float(solver_settings["dt_save"]),
            "dt_summarise": float(solver_settings["dt_summarise"]),
            "warmup": float(solver_settings.get("warmup", 0.0)),
            "duration": float(solver_settings["duration"]),
            "atol": float(solver_settings["atol"]),
            "rtol": float(solver_settings["rtol"]),
        }
        inputs = self._reference_inputs(system, initial_state, solver_config)
        controller_config = self._reference_controller_settings(
            step_controller_settings,
            solver_config,
        )
        result = run_reference_loop(
            system=system,
            inputs=inputs,
            solver_settings=solver_config,
            loop_compile_settings=loop_compile_settings,
            output_functions=output_functions,
            stepper=stepper,
            step_controller_settings=controller_config,
        )
        return result["state"]

    def _run_loop(
        self,
        loop: IVPLoop,
        system,
        init: NDArray[np.floating],
        output_functions,
        duration: float,
    ) -> Dict[str, Any]:
        """Execute the built loop on a single thread."""

        precision = loop.precision
        dt_save = loop.dt_save
        saves = int(np.ceil(precision(duration) / precision(dt_save)))

        heights = OutputArrayHeights.from_output_fns(output_functions)
        state_width = max(heights.state, 1)
        observable_width = max(heights.observables, 1)
        state_summary_width = max(heights.state_summaries, 1)
        observable_summary_width = max(heights.observable_summaries, 1)
        summary_samples = max(int(np.ceil(duration / loop.dt_summarise)), 1)

        state_output = np.zeros((saves, state_width), dtype=precision)
        observables_output = np.zeros((saves, observable_width), dtype=precision)
        state_summary_output = np.zeros(
            (summary_samples, state_summary_width), dtype=precision
        )
        observable_summary_output = np.zeros(
            (summary_samples, observable_summary_width), dtype=precision
        )

        params = np.array(system.parameters.values_array, dtype=precision, copy=True)
        drivers = self._driver_sequence(
            duration,
            dt_save,
            system.num_drivers,
            precision,
        )
        init_state = np.array(init, dtype=precision, copy=True)
        status = np.zeros(1, dtype=np.int32)

        d_init = cuda.to_device(init_state)
        d_params = cuda.to_device(params)
        d_drivers = cuda.to_device(drivers)
        d_state_out = cuda.to_device(state_output)
        d_obs_out = cuda.to_device(observables_output)
        d_state_sum = cuda.to_device(state_summary_output)
        d_obs_sum = cuda.to_device(observable_summary_output)
        d_status = cuda.to_device(status)

        base_shared = loop.buffer_indices.local_end
        algo_shared = getattr(loop.algorithm, "shared_memory_required", 0)
        controller_shared = getattr(
            loop.step_controller, "shared_memory_required", 0
        )
        shared_elements = base_shared + algo_shared + controller_shared
        shared_bytes = np.dtype(precision).itemsize * shared_elements

        local_req = max(1, loop.local_memory_elements)

        loop_fn = loop.device_function
        numba_precision = from_dtype(precision)

        @cuda.jit
        def kernel(
            init_vec,
            params_vec,
            drivers_vec,
            state_out_arr,
            obs_out_arr,
            state_sum_arr,
            obs_sum_arr,
            status_arr,
        ):
            idx = cuda.grid(1)
            if idx > 1:
                return
            
            shared = cuda.shared.array(0, dtype=numba_precision)
            local = cuda.local.array(local_req, dtype=numba_precision)
            status_arr[0] = loop_fn(
                init_vec,
                params_vec,
                drivers_vec,
                shared,
                local,
                state_out_arr,
                obs_out_arr,
                state_sum_arr,
                obs_sum_arr,
                precision(duration),
                precision(0.0),
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

        state_host = d_state_out.copy_to_host()
        observables_host = d_obs_out.copy_to_host()
        state_summary_host = d_state_sum.copy_to_host()
        observable_summary_host = d_obs_sum.copy_to_host()
        status_value = int(d_status.copy_to_host()[0])


        return {
            "state": state_host,
            "observables": observables_host,
            "state_summaries": state_summary_host,
            "observable_summaries": observable_summary_host,
            "status": status_value,
        }

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_instantiation_and_build(self, loop: IVPLoop) -> None:
        """Ensure the loop builds a valid device function."""

        loop_fn = loop.device_function
        assert is_devfunc(loop_fn)

    def test_numerical_kernel(
        self,
        loop: IVPLoop,
        expected_state: NDArray[np.floating],
        system,
        solver_settings: Dict[str, Any],
        initial_state: NDArray[np.floating],
        output_functions,
    ) -> None:
        """Run the loop and compare its state output."""
        simulation_duration = solver_settings['duration']
        results = self._run_loop(
            loop,
            system,
            initial_state,
            output_functions,
            simulation_duration,
        )
        assert results["status"] == 0, f"Solver status {results['status']}"
        state = results["state"]
        assert_allclose(state, expected_state, rtol=1e-4, atol=1e-4)

    def test_getters(
        self,
        loop: IVPLoop,
        buffer_sizes: LoopBufferSizes,
        precision: np.dtype,
    ) -> None:
        """Validate loop property accessors."""

        assert loop.dt_save == pytest.approx(loop.compile_settings.dt_save)
        assert loop.dt_summarise == pytest.approx(
            loop.compile_settings.dt_summarise
        )
        assert loop.precision is precision
        idx = loop.buffer_indices
        assert idx.state.stop - idx.state.start == buffer_sizes.state
        assert idx.dxdt.stop - idx.dxdt.start == buffer_sizes.dxdt
        if loop.is_adaptive:
            assert loop.dt_min == pytest.approx(loop.step_controller.dt_min)
            assert loop.dt_max == pytest.approx(loop.step_controller.dt_max)
        else:
            assert loop.dt_min == pytest.approx(loop.dt0)
            assert loop.dt_max == pytest.approx(loop.dt0)
        assert loop.buffer_indices is not None
        state_indices = loop.buffer_indices.state
        assert state_indices.stop - state_indices.start == buffer_sizes.state

    @pytest.fixture(scope="function")
    def algorithm_update_param(self) -> Dict[str, float]:
        """Algorithm-specific updates, overridden by subclasses."""

        return {}

    def test_update(
        self,
        loop: IVPLoop,
        algorithm_update_param: Dict[str, float],
    ) -> None:
        """Ensure the loop recognises configuration updates."""

        params = {"dt_save": 0.05, "dt_summarise": 0.25}
        params.update(algorithm_update_param)
        _ = loop.build()
        recognised = loop.update(params)
        expected_keys = set(params.keys())
        assert expected_keys.issubset(recognised)
        assert loop.dt_save == pytest.approx(0.05)
