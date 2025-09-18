"""Tests for IVPLoop with Euler-family integration loops."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
from numpy.typing import NDArray

from cubie.integrators.algorithms_ import (
    BackwardsEulerStep,
    CrankNicolsonStep,
    ExplicitEulerStep,
)

from tests.integrators.loops.ODELoopTester import ODELoopTester


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "loop_compile_settings_overrides",
    [
        {
            "dt_min": 0.01,
            "dt_max": 0.1,
            "dt_save": 0.01,
            "dt_summarise": 0.5,
            "saved_state_indices": [0, 1],
            "saved_observable_indices": [],
            "summarised_state_indices": [0, 1],
            "summarised_observable_indices": [],
            "output_functions": ["state"],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "step_controller_settings_override",
    [{"kind": "fixed"}],
    indirect=True,
)
class TestExplicitEulerLoop(ODELoopTester):
    """Tests for IVPLoop with explicit Euler steps."""

    @pytest.fixture(scope="function")
    def step_object(
        self,
        system,
        loop_compile_settings: Dict[str, float],
        precision: np.dtype,
    ):
        """Instantiate the explicit Euler step."""

        step_size = loop_compile_settings["dt_min"]
        return ExplicitEulerStep(
            system.dxdt_function,
            precision,
            system.sizes.states,
            step_size,
        )

    @pytest.fixture(scope="function")
    def expected_state(
        self,
        system,
        initial_state: NDArray[np.floating],
        solver_settings: Dict[str, float],
        loop_compile_settings: Dict[str, float],
        output_functions,
        step_controller_settings: Dict[str, float | int | str],
    ) -> NDArray[np.floating]:
        """Expected discrete solution from explicit Euler integration."""

        return self._reference_state_output(
            system=system,
            initial_state=initial_state,
            loop_compile_settings=loop_compile_settings,
            solver_settings=solver_settings,
            output_functions=output_functions,
            stepper="explicit_euler",
            step_controller_settings=step_controller_settings,
        )

    @pytest.fixture(scope="function")
    def algorithm_update_param(
        self,
        loop_compile_settings: Dict[str, float],
    ) -> Dict[str, float]:
        """Request a smaller fixed step size during updates."""

        new_dt = loop_compile_settings["dt_min"] / 2
        return {"fixed_step_size": new_dt, "dt": new_dt}


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "loop_compile_settings_overrides",
    [
        {
            "dt_min": 0.01,
            "dt_max": 0.1,
            "dt_save": 0.01,
            "dt_summarise": 0.5,
            "saved_state_indices": [0, 1],
            "saved_observable_indices": [],
            "summarised_state_indices": [0, 1],
            "summarised_observable_indices": [],
            "output_functions": ["state"],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "step_controller_settings_override",
    [{"kind": "fixed"}],
    indirect=True,
)
class TestBackwardsEulerLoop(ODELoopTester):
    """Tests for IVPLoop with backward Euler steps."""

    @pytest.fixture(scope="function")
    def step_object(
        self,
        system,
        loop_compile_settings: Dict[str, float],
        precision: np.dtype,
    ):
        """Instantiate the backward Euler step."""

        return BackwardsEulerStep(
            precision=precision,
            n=system.sizes.states,
            dxdt_function=system.dxdt_function,
            get_solver_helper_fn=system.get_solver_helper,
            atol=loop_compile_settings["atol"],
            rtol=loop_compile_settings["rtol"],
        )

    @pytest.fixture(scope="function")
    def expected_state(
        self,
        system,
        initial_state: NDArray[np.floating],
        solver_settings: Dict[str, float],
        loop_compile_settings: Dict[str, float],
        output_functions,
        step_controller_settings: Dict[str, float | int | str],
    ) -> NDArray[np.floating]:
        """Expected discrete solution from backward Euler integration."""

        return self._reference_state_output(
            system=system,
            initial_state=initial_state,
            loop_compile_settings=loop_compile_settings,
            solver_settings=solver_settings,
            output_functions=output_functions,
            stepper="backward_euler",
            step_controller_settings=step_controller_settings,
        )

    @pytest.fixture(scope="function")
    def algorithm_update_param(
        self,
        loop_compile_settings: Dict[str, float],
    ) -> Dict[str, float]:
        """Update tolerances and step size for the implicit solver."""

        new_dt = loop_compile_settings["dt_min"] / 2
        return {"dt": new_dt}


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "loop_compile_settings_overrides",
    [
        {
            "dt_min": 0.001,
            "dt_max": 0.2,
            "dt_save": 0.1,
            "dt_summarise": 0.5,
            "saved_state_indices": [0, 1],
            "saved_observable_indices": [],
            "summarised_state_indices": [0, 1],
            "summarised_observable_indices": [],
            "output_functions": ["state"],
            "atol": 1e-3,
            "rtol": 1e-3,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "step_controller_settings_override",
    [
        {"kind": "I", "order": 2},
        {"kind": "PI", "order": 2},
        {"kind": "PID", "order": 2},
        {"kind": "gustafsson", "order": 2},
    ],
    indirect=True,
)
class TestCrankNicolsonLoop(ODELoopTester):
    """Tests for IVPLoop with Crank-Nicolson steps."""

    @pytest.fixture(scope="function")
    def step_object(
        self,
        system,
        loop_compile_settings: Dict[str, float],
        precision: np.dtype,
    ):
        """Instantiate the Crank-Nicolson step."""

        return CrankNicolsonStep(
            precision=precision,
            n=system.sizes.states,
            dxdt_function=system.dxdt_function,
            get_solver_helper_fn=system.get_solver_helper,
            atol=loop_compile_settings["atol"],
            rtol=loop_compile_settings["rtol"],
        )

    @pytest.fixture(scope="function")
    def expected_state(
        self,
        system,
        initial_state: NDArray[np.floating],
        solver_settings: Dict[str, float],
        loop_compile_settings: Dict[str, float],
        output_functions,
        step_controller_settings: Dict[str, float | int | str],
    ) -> NDArray[np.floating]:
        """Expected solution from the CPU reference Crank-Nicolson loop."""

        return self._reference_state_output(
            system=system,
            initial_state=initial_state,
            loop_compile_settings=loop_compile_settings,
            solver_settings=solver_settings,
            output_functions=output_functions,
            stepper="crank_nicolson",
            step_controller_settings=step_controller_settings,
        )

    @pytest.fixture(scope="function")
    def algorithm_update_param(
        self,
        loop_compile_settings: Dict[str, float],
    ) -> Dict[str, float]:
        """Adaptive algorithms rely on controller updates; no loop changes."""

        return {'dt_min': 1e-4}
