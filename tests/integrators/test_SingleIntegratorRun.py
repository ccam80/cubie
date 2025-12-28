"""Integration tests for :mod:`cubie.integrators.SingleIntegratorRun`."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from cubie import SingleIntegratorRun
from cubie.integrators.SingleIntegratorRunCore import SingleIntegratorRunCore
from tests._utils import assert_integration_outputs


def _compare_scalar(actual, expected, tolerance):
    if expected is None:
        assert actual is None
        return
    if isinstance(expected, (float, np.floating)):
        assert actual == pytest.approx(
            float(expected),
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        return
    if isinstance(expected, (int, np.integer)):
        assert int(actual) == int(expected)
        return
    assert actual == expected


def _compare_array(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _compare_generic(actual, expected, tolerance):
    if isinstance(expected, np.ndarray):
        _compare_array(actual, expected)
    elif isinstance(expected, (float, np.floating, int, np.integer)):
        _compare_scalar(actual, expected, tolerance)
    elif expected is None:
        assert actual is None
    elif callable(expected):
        assert actual is expected
    else:
        assert actual == expected


def _settings_to_dict(settings_source):
    settings = settings_source
    if callable(settings_source):
        settings = settings_source()
    return dict(settings)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {},
        {
            "algorithm": "bogacki-shampine-32",
            "step_controller": "pid",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-7,
            "dt_max": 0.1,
            "dt_save": 0.1,
            "dt_summarise": 0.3,
            "duration": 0.3,
            "output_types": ["state","time", "observables","mean"],
            "saved_state_indices": [0],
            "saved_observable_indices": [0],
            "summarised_state_indices": [0],
            "summarised_observable_indices": [0],
        },
    ],
    indirect=True,
)
class TestSingleIntegratorRun:
    def test_build_getters_and_equivalence(
        self,
        single_integrator_run,
        system,
        solver_settings,
        precision,
        initial_state,
        cpu_loop_outputs,
        device_loop_outputs,
        driver_array,
        tolerance,
    ):
        """Requesting the device loop compiles children and preserves getters."""

        run = single_integrator_run
        device_fn = run.device_function
        assert callable(device_fn)
        assert run.cache_valid is True
        assert run._loop.cache_valid is True
        assert run._algo_step.cache_valid is True
        assert run._step_controller.cache_valid is True
        assert run._output_functions.cache_valid is True

        # Compile settings echo the configuration used during setup.
        assert run.precision is precision
        expected_algorithm = run.compile_settings.algorithm
        assert run.algorithm == expected_algorithm
        assert run.algorithm_key == expected_algorithm
        assert solver_settings["algorithm"].lower() in expected_algorithm
        expected_controller = solver_settings["step_controller"].lower()
        assert run.step_controller == expected_controller

        assert run.dt_save == pytest.approx(
            solver_settings["dt_save"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert run.dt_summarise == pytest.approx(
            solver_settings["dt_summarise"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )

        if run.is_adaptive:
            dt_min = solver_settings["dt_min"]
            assert run.dt_min == pytest.approx(
                dt_min,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert run.dt_max == pytest.approx(
                solver_settings["dt_max"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
        else:
            assert run.dt_max == pytest.approx(
                run.dt_min,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert run.dt == pytest.approx(
                solver_settings["dt"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )

        # Controller tolerances are only defined for adaptive controllers.
        if run.atol is not None:
            target = np.asarray(solver_settings["atol"], dtype=precision)
            np.testing.assert_allclose(
                run.atol,
                target,
                rtol=tolerance.rel_tight,
                atol=tolerance.abs_tight,
            )
        if run.rtol is not None:
            target = np.asarray(solver_settings["rtol"], dtype=precision)
            np.testing.assert_allclose(
                run.rtol,
                target,
                rtol=tolerance.rel_tight,
                atol=tolerance.abs_tight,
            )

        # Saved indices and counts mirror the configured output selections.


        assert list(run.output_types) == list(solver_settings["output_types"])

        assert run.compiled_loop_function is device_fn
        assert run.threads_per_loop == run._algo_step.threads_per_step

        # Properties that simply forward underlying objects.
        loop_props: Dict[str, str] = {
            "compile_flags": "compile_flags",
            "save_state_fn": "save_state_fn",
            "update_summaries_fn": "update_summaries_fn",
            "save_summaries_fn": "save_summaries_fn",
            "control_device_function": "step_controller_fn",
            "compiled_loop_step_function": "step_function",
        }
        controller_props: Dict[str, str] = {
            "min_gain": "min_gain",
            "max_gain": "max_gain",
            "safety": "safety",
            "algorithm_order": "algorithm_order",
            "atol": "atol",
            "rtol": "rtol",
            "kp": "kp",
            "ki": "ki",
            "kd": "kd",
            "gamma": "gamma",
            "max_newton_iters": "max_newton_iters",
        }
        algo_props: Dict[str, str] = {
            "threads_per_step": "threads_per_step",
            "uses_multiple_stages": "is_multistage",
            "adapts_step": "is_adaptive",
            "implicit_step": "is_implicit",
            "order": "order",
            "integration_step_function": "step_function",
            "state_count": "n",
            "solver_helper": "get_solver_helper_fn",
            "beta_coefficient": "beta",
            "gamma_coefficient": "gamma",
            "mass_matrix": "mass_matrix",
            "preconditioner_order": "preconditioner_order",
            "linear_solver_tolerance": "krylov_tolerance",
            "max_linear_iterations": "max_linear_iters",
            "linear_correction_type": "linear_correction_type",
            "newton_tolerance": "newton_tolerance",
            "newton_iterations_limit": "max_newton_iters",
            "newton_damping": "newton_damping",
            "newton_max_backtracks": "newton_max_backtracks",
            "integration_step_size": "dt",
        }
        output_props: Dict[str, str] = {
            "save_state_func": "save_state_func",
            "update_summaries_func": "update_summaries_func",
            "save_summary_metrics_func": "save_summary_metrics_func",
            "output_types": "output_types",
            "output_compile_flags": "compile_flags",
            "save_time": "save_time",
            "saved_state_indices": "saved_state_indices",
            "saved_observable_indices": "saved_observable_indices",
            "summarised_state_indices": "summarised_state_indices",
            "summarised_observable_indices": "summarised_observable_indices",
            "n_saved_states": "n_saved_states",
            "n_saved_observables": "n_saved_observables",
            "state_summaries_output_height": "state_summaries_output_height",
            "observable_summaries_output_height": "observable_summaries_output_height",
            "summary_buffer_height_per_variable": "summaries_buffer_height_per_var",
            "state_summaries_buffer_height": "state_summaries_buffer_height",
            "observable_summaries_buffer_height": "observable_summaries_buffer_height",
            "total_summary_buffer_size": "total_summary_buffer_size",
            "summary_output_height_per_variable": "summaries_output_height_per_var",
            "n_summarised_states": "n_summarised_states",
            "n_summarised_observables": "n_summarised_observables",
            "output_array_heights": "output_array_heights",
            "summary_legend_per_variable": "summary_legend_per_variable",
        }

        for prop_name, attr_name in loop_props.items():
            actual = getattr(run, prop_name)
            expected = getattr(run._loop, attr_name)
            _compare_generic(actual, expected, tolerance)

        for prop_name, attr_name in controller_props.items():
            actual = getattr(run, prop_name)
            expected = getattr(run._step_controller, attr_name, None)
            _compare_generic(actual, expected, tolerance)

        for prop_name, attr_name in algo_props.items():
            actual = getattr(run, prop_name)
            expected = getattr(run._algo_step, attr_name, None)
            _compare_generic(actual, expected, tolerance)

        for prop_name, attr_name in output_props.items():
            actual = getattr(run, prop_name)
            expected = getattr(run._output_functions, attr_name)
            if prop_name in {
                "saved_state_indices",
                "saved_observable_indices",
                "summarised_state_indices",
                "summarised_observable_indices",
            }:
                _compare_array(actual, expected)
            else:
                _compare_generic(actual, expected, tolerance)

        # Numerical equivalence with the CPU reference loop.
        cpu_reference = cpu_loop_outputs
        device_outputs = device_loop_outputs

        assert device_outputs.status == cpu_reference["status"]
        assert_integration_outputs(
            reference=cpu_reference,
            device=device_outputs,
            output_functions=run._output_functions,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
        )


def test_update_routes_to_children(
    single_integrator_run_mutable,
    solver_settings,
    system,
    tolerance,
):
    """All components receive updates and report the new configuration."""

    run = single_integrator_run_mutable
    new_dt = solver_settings["dt_min"] * 0.5
    new_saved_states = [0]
    new_saved_observables = [0]
    new_constant = system.constants.values_array[0] * 1.2

    updates = {
        "dt": new_dt,
        "output_types": ['state', 'observables', 'mean'],
        "saved_state_indices": new_saved_states,
        "saved_observable_indices": new_saved_observables,
        "summarised_state_indices": new_saved_states,
        "summarised_observable_indices": new_saved_observables,
        "c0": new_constant,
    }

    recognized = run.update(updates)
    expected_keys = {
        "dt",
        "saved_state_indices",
        "saved_observable_indices",
        "summarised_state_indices",
        "summarised_observable_indices",
        "c0",
    }
    assert expected_keys.issubset(recognized)
    assert run.cache_valid is False

    assert run.dt == pytest.approx(
        new_dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert run.dt_min == pytest.approx(
        new_dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert run.dt_max == pytest.approx(
        new_dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    flags = run.output_compile_flags
    expected_saved_states = (
        np.asarray(new_saved_states)
        if flags.save_state
        else np.empty(0, dtype=np.int64)
    )
    expected_saved_obs = (
        np.asarray(new_saved_observables)
        if flags.save_observables
        else np.empty(0, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        run.saved_state_indices, expected_saved_states
    )
    np.testing.assert_array_equal(
        run.saved_observable_indices, expected_saved_obs
    )
    np.testing.assert_array_equal(
        run.summarised_state_indices, np.asarray(new_saved_states)
    )
    np.testing.assert_array_equal(
        run.summarised_observable_indices,
        np.asarray(new_saved_observables),
    )

    assert run.n_saved_states == int(expected_saved_states.size)
    assert run.n_saved_observables == int(expected_saved_obs.size)
    expected_summary_count = (
        len(new_saved_states) if flags.summarise_state else 0
    )
    expected_summary_obs = (
        len(new_saved_observables) if flags.summarise_observables else 0
    )
    assert run.n_summarised_states == expected_summary_count
    assert run.n_summarised_observables == expected_summary_obs

    controller_settings = _settings_to_dict(run._step_controller.settings_dict)
    algo_settings = _settings_to_dict(run._algo_step.settings_dict)
    assert controller_settings["dt"] == pytest.approx(
        new_dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    assert float(system.constants.values_array[0]) == pytest.approx(
        new_constant,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )



def test_default_step_controller_settings_applied(
    system,
    solver_settings,
    driver_array,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """When no overrides are supplied algorithm defaults are applied."""

    driver_fn = driver_array.evaluation_function if driver_array else None
    run = SingleIntegratorRun(
        system=system,
        loop_settings=loop_settings,
        driver_function=driver_fn,
        step_control_settings=None,
        algorithm_settings=algorithm_settings,
        output_settings=output_settings,
    )

    defaults = run._algo_step.controller_defaults.copy()
    assert run.step_controller == defaults.step_controller['step_controller']
    controller_settings = run._step_controller.settings_dict
    defaults.step_controller.pop('step_controller')
    for key, expected in defaults.step_controller.items():

        assert key in controller_settings
        actual = controller_settings[key]
        if isinstance(expected, (float, np.floating)):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected
    assert run._step_controller.n == system.sizes.states
    if run.algorithm_order is not None:
        assert run.algorithm_order == run._algo_step.order


@pytest.mark.parametrize(
    "algorithm, overrides",
    [
        (
            "crank_nicolson",
            {
                "dt_min": 5e-5,
                "dt_max": 5e-2,
                "min_gain": 0.3,
            },
        )
    ],
)

def test_step_controller_overrides_take_precedence(
    system,
    solver_settings,
    output_settings,

    driver_array,
    algorithm,
    overrides,
    algorithm_settings,
    loop_settings,
):
    """User supplied settings override algorithm defaults."""
    algorithm_settings["algorithm"] = algorithm
    precision = system.precision
    driver_fn = driver_array.evaluation_function if driver_array else None
    override_settings = {
        key: precision(value) if isinstance(value, float) else value
        for key, value in overrides.items()
    }
    run = SingleIntegratorRun(
        system=system,
        loop_settings=loop_settings,
        output_settings=output_settings,
        driver_function=driver_fn,
        step_control_settings=dict(override_settings),
        algorithm_settings=algorithm_settings,
    )

    assert run.step_controller == "pid"
    assert run.dt_min == pytest.approx(override_settings["dt_min"])
    assert run.dt_max == pytest.approx(override_settings["dt_max"])
    controller_settings = run._step_controller.settings_dict
    assert controller_settings["min_gain"] == pytest.approx(
        override_settings["min_gain"]
    )
    assert controller_settings["algorithm_order"] == run._algo_step.order


# Algorithm-controller compatibility tests
def test_errorless_euler_with_adaptive_warns_and_replaces(system):
    """Errorless Euler with adaptive PI warns and replaces with fixed."""
    import warnings
    
    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
        
        # Should issue our compatibility warning (may have other warnings too)
        compat_warnings = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat_warnings) >= 1
        assert issubclass(compat_warnings[0].category, UserWarning)
        warn_msg = str(compat_warnings[0].message).lower()
        assert "euler" in warn_msg
        assert "pid" in warn_msg
        assert "fixed" in warn_msg
        
        # Controller should be replaced with fixed
        assert not core._step_controller.is_adaptive
        assert core._algo_step.is_controller_fixed


def test_errorless_rk4_tableau_with_adaptive_warns(system):
    """Errorless RK4 tableau with adaptive PI warns and replaces."""
    import warnings
    from cubie.integrators.algorithms.generic_erk_tableaus import (
        CLASSICAL_RK4_TABLEAU,
    )
    
    algorithm_settings = {
        "algorithm": "erk",
        "tableau": CLASSICAL_RK4_TABLEAU,
    }
    step_control_settings = {
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
        
        compat_warnings = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat_warnings) >= 1
        assert not core._step_controller.is_adaptive
        assert core._algo_step.is_controller_fixed


def test_adaptive_tableau_with_adaptive_succeeds(system):
    """Adaptive Dormand-Prince with PI controller succeeds without warning."""
    import warnings
    from cubie.integrators.algorithms.generic_erk_tableaus import (
        DORMAND_PRINCE_54_TABLEAU,
    )
    
    algorithm_settings = {
        "algorithm": "erk",
        "tableau": DORMAND_PRINCE_54_TABLEAU,
    }
    step_control_settings = {
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
        
        # Should not issue our compatibility warning
        compat_warnings = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat_warnings) == 0
        assert core._algo_step.is_adaptive
        assert core._step_controller.is_adaptive
        assert not core._algo_step.is_controller_fixed


def test_errorless_euler_with_fixed_succeeds(system):
    """Errorless Euler with fixed controller succeeds without warning."""
    import warnings
    
    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "fixed",
        "dt": 1e-3,
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
        
        # Should not issue our compatibility warning
        compat_warnings = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat_warnings) == 0
        assert not core._algo_step.is_adaptive
        assert not core._step_controller.is_adaptive
        assert core._algo_step.is_controller_fixed


def test_warning_message_contains_algorithm_and_controller(system):
    """Warning message includes algorithm and controller names."""
    import warnings
    
    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
        
        compat_warnings = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat_warnings) >= 1
        warn_msg = str(compat_warnings[0].message)
        assert "euler" in warn_msg
        assert "pid" in warn_msg
        assert "error estimate" in warn_msg.lower()
        assert "fixed" in warn_msg.lower()
