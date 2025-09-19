"""Integration loop tests comparing device output with CPU references."""

from __future__ import annotations

from typing import Mapping

import pytest
from numpy.testing import assert_allclose

from cubie.integrators.algorithms_ import (
    BackwardsEulerStep,
    CrankNicolsonStep,
    ExplicitEulerStep,
)
from tests.integrators.loops.ODELoopTester import (
    build_reference_controller_settings,
    build_solver_config,
    cpu_reference_outputs,
    extract_state_and_time,
    run_device_loop,
)


pytestmark = pytest.mark.nocudasim


def _execute_reference_and_device(
    *,
    loop,
    system,
    initial_state,
    solver_settings,
    loop_compile_settings,
    output_functions,
    stepper: str,
    step_controller_settings: Mapping[str, float | int | str],
):
    """Run CPU reference and device loop using the provided configuration."""

    solver_config = build_solver_config(solver_settings)
    controller_config = build_reference_controller_settings(
        step_controller_settings, solver_config
    )
    reference = cpu_reference_outputs(
        system=system,
        initial_state=initial_state,
        solver_config=solver_config,
        loop_compile_settings=loop_compile_settings,
        output_functions=output_functions,
        stepper=stepper,
        controller_settings=controller_config,
    )
    device_raw = run_device_loop(
        loop=loop,
        system=system,
        initial_state=initial_state,
        output_functions=output_functions,
        solver_config=solver_config,
    )
    device = device_raw.trimmed_to(reference)
    return reference, device


def _assert_common_outputs(
    reference,
    device,
    output_functions,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare state, summary, and time outputs between CPU and device."""

    state_ref, time_ref = extract_state_and_time(
        reference["state"], output_functions
    )
    state_dev, time_dev = extract_state_and_time(
        device.state,
        output_functions,
    )
    assert_allclose(state_dev, state_ref, rtol=rtol, atol=atol)
    if time_ref is not None:
        assert_allclose(time_dev, time_ref, rtol=rtol, atol=atol)
    assert_allclose(
        device.state_summaries,
        reference["state_summaries"],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        device.observable_summaries,
        reference["observable_summaries"],
        rtol=rtol,
        atol=atol,
    )


class TestExplicitEulerLoop:
    """Tests for :class:`ExplicitEulerStep` with fixed and adaptive control."""

    @pytest.fixture(scope="function", name="step_object")
    def fixture_step_object(
        self, system, loop_compile_settings, precision
    ) -> ExplicitEulerStep:
        step_size = loop_compile_settings["dt_min"]
        return ExplicitEulerStep(
            system.dxdt_function,
            precision,
            system.sizes.states,
            step_size,
        )

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {
                "dt_min": 0.02,
                "dt_max": 0.02,
                "dt_save": 0.04,
                "dt_summarise": 0.08,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 1, 2],
                "summarised_state_indices": [0, 1, 2],
                "summarised_observable_indices": [0, 1, 2],
                "output_functions": [
                    "state",
                    "observables",
                    "time",
                    "mean",
                    "max",
                ],
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "solver_settings_override",
        [
            {
                "duration": 0.24,
                "warmup": 0.04,
                "dt_save": 0.04,
                "dt_summarise": 0.08,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "step_controller_settings_override",
        [{"kind": "fixed"}],
        indirect=True,
    )
    def test_fixed_controller_matches_reference(
        self,
        loop,
        system,
        initial_state,
        solver_settings,
        loop_compile_settings,
        output_functions,
        step_controller_settings,
    ) -> None:
        """Device loop should match CPU reference for fixed-step control."""

        reference, device = _execute_reference_and_device(
            loop=loop,
            system=system,
            initial_state=initial_state,
            solver_settings=solver_settings,
            loop_compile_settings=loop_compile_settings,
            output_functions=output_functions,
            stepper="explicit_euler",
            step_controller_settings=step_controller_settings,
        )
        assert device.status == 0
        _assert_common_outputs(
            reference,
            device,
            output_functions,
            rtol=1e-5,
            atol=1e-6,
        )
        assert_allclose(
            device.observables, reference["observables"], rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {
                "dt_min": 0.01,
                "dt_max": 0.1,
                "dt_save": 0.05,
                "dt_summarise": 0.15,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 1, 2],
                "summarised_state_indices": [0, 1, 2],
                "summarised_observable_indices": [0, 1, 2],
                "output_functions": [
                    "state",
                    "observables",
                    "time",
                    "mean",
                ],
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "solver_settings_override",
        [
            {
                "duration": 0.2,
                "warmup": 0.0,
                "dt_save": 0.05,
                "dt_summarise": 0.15,
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "step_controller_settings_override",
        [
            {
                "kind": "PI",
                "order": 2,
                "dt_min": 0.01,
                "dt_max": 0.1,
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        ],
        indirect=True,
    )
    def test_adaptive_time_vector_matches_reference(
        self,
        loop,
        system,
        initial_state,
        solver_settings,
        loop_compile_settings,
        output_functions,
        step_controller_settings,
    ) -> None:
        """Adaptive controller should match the CPU save times."""

        reference, device = _execute_reference_and_device(
            loop=loop,
            system=system,
            initial_state=initial_state,
            solver_settings=solver_settings,
            loop_compile_settings=loop_compile_settings,
            output_functions=output_functions,
            stepper="explicit_euler",
            step_controller_settings=step_controller_settings,
        )
        assert device.status == 0
        _assert_common_outputs(
            reference,
            device,
            output_functions,
            rtol=5e-4,
            atol=1e-6,
        )
        assert_allclose(
            device.observables,
            reference["observables"],
            rtol=5e-4,
            atol=1e-6,
        )


class TestBackwardEulerLoop:
    """Tests for :class:`BackwardsEulerStep` using adaptive control."""

    @pytest.fixture(scope="function", name="step_object")
    def fixture_step_object(
        self,
        system,
        loop_compile_settings,
        precision,
    ) -> BackwardsEulerStep:
        return BackwardsEulerStep(
            precision=precision,
            n=system.sizes.states,
            dxdt_function=system.dxdt_function,
            get_solver_helper_fn=system.get_solver_helper,
            atol=loop_compile_settings["atol"],
            rtol=loop_compile_settings["rtol"],
        )

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {
                "dt_min": 0.002,
                "dt_max": 0.05,
                "dt_save": 0.05,
                "dt_summarise": 0.1,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 1, 2],
                "summarised_state_indices": [0, 1, 2],
                "summarised_observable_indices": [0, 1, 2],
                "output_functions": [
                    "state",
                    "observables",
                    "time",
                    "mean",
                    "max",
                ],
                "atol": 5e-6,
                "rtol": 5e-5,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "solver_settings_override",
        [
            {
                "duration": 0.25,
                "warmup": 0.05,
                "dt_save": 0.05,
                "dt_summarise": 0.1,
                "atol": 5e-6,
                "rtol": 5e-5,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "step_controller_settings_override",
        [
            {
                "kind": "PID",
                "order": 2,
                "dt_min": 0.002,
                "dt_max": 0.05,
                "atol": 5e-6,
                "rtol": 5e-5,
            }
        ],
        indirect=True,
    )
    def test_adaptive_loop_matches_reference(
        self,
        loop,
        system,
        initial_state,
        solver_settings,
        loop_compile_settings,
        output_functions,
        step_controller_settings,
    ) -> None:
        """Backward Euler device loop should reproduce the CPU simulation."""

        reference, device = _execute_reference_and_device(
            loop=loop,
            system=system,
            initial_state=initial_state,
            solver_settings=solver_settings,
            loop_compile_settings=loop_compile_settings,
            output_functions=output_functions,
            stepper="backward_euler",
            step_controller_settings=step_controller_settings,
        )
        assert device.status == 0
        _assert_common_outputs(
            reference,
            device,
            output_functions,
            rtol=5e-4,
            atol=1e-6,
        )
        assert_allclose(
            device.observables,
            reference["observables"],
            rtol=5e-4,
            atol=1e-6,
        )


class TestCrankNicolsonLoop:
    """Crank–Nicolson loop tests covering multiple tolerance levels."""

    @pytest.fixture(scope="function", name="step_object")
    def fixture_step_object(
        self,
        system,
        loop_compile_settings,
        precision,
    ) -> CrankNicolsonStep:
        return CrankNicolsonStep(
            precision=precision,
            n=system.sizes.states,
            dxdt_function=system.dxdt_function,
            get_solver_helper_fn=system.get_solver_helper,
            atol=loop_compile_settings["atol"],
            rtol=loop_compile_settings["rtol"],
        )

    @pytest.mark.parametrize(
        "loop_compile_settings_overrides",
        [
            {
                "dt_min": 0.001,
                "dt_max": 0.05,
                "dt_save": 0.04,
                "dt_summarise": 0.12,
                "saved_state_indices": [0, 1, 2],
                "saved_observable_indices": [0, 1, 2],
                "summarised_state_indices": [0, 1, 2],
                "summarised_observable_indices": [0, 1, 2],
                "output_functions": [
                    "state",
                    "observables",
                    "time",
                    "mean",
                    "max",
                ],
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "solver_settings_override",
        [
            {
                "duration": 0.24,
                "warmup": 0.02,
                "dt_save": 0.04,
                "dt_summarise": 0.12,
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "step_controller_settings_override",
        [
            {
                "kind": "gustafsson",
                "order": 2,
                "dt_min": 0.001,
                "dt_max": 0.05,
                "atol": 1e-5,
                "rtol": 1e-4,
            },
            {
                "kind": "PI",
                "order": 2,
                "dt_min": 0.001,
                "dt_max": 0.05,
                "atol": 5e-6,
                "rtol": 5e-5,
            },
        ],
        indirect=True,
    )
    @pytest.mark.xfail(
        reason=(
            "Adaptive observable output is not yet implemented."
        ),
        strict=True,
    )
    def test_adaptive_outputs_match_reference(
        self,
        loop,
        system,
        initial_state,
        solver_settings,
        loop_compile_settings,
        output_functions,
        step_controller_settings,
    ) -> None:
        """All outputs should match the CPU reference for adaptive control."""

        reference, device = _execute_reference_and_device(
            loop=loop,
            system=system,
            initial_state=initial_state,
            solver_settings=solver_settings,
            loop_compile_settings=loop_compile_settings,
            output_functions=output_functions,
            stepper="crank_nicolson",
            step_controller_settings=step_controller_settings,
        )
        assert device.status == 0
        _assert_common_outputs(
            reference,
            device,
            output_functions,
            rtol=1e-3,
            atol=1e-6,
        )
        assert_allclose(
            device.observables,
            reference["observables"],
            rtol=1e-3,
            atol=1e-6,
        )

