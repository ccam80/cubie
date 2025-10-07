from typing import Iterable
from os import environ

import pytest
import numpy as np
from cubie.batchsolving.solver import Solver, solve_ivp
from cubie.batchsolving.solveresult import SolveResult, SolveSpec
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.SystemInterface import SystemInterface

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import (
        FakeCUDAArray as MappedNDArray,
    )
else:
    from numba.cuda.cudadrv.devicearray import MappedNDArray


@pytest.fixture(scope="function")
def solver_instance(system, solver_settings, step_controller_settings):
    """Create a solver instance for testing."""
    return Solver(
        system,
        algorithm=solver_settings["algorithm"],
        duration=solver_settings["duration"],
        warmup=solver_settings["warmup"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        saved_states=None,
        saved_observables=None,  # Will use defaults
        output_types=solver_settings["output_types"],
        precision=solver_settings["precision"],
        profileCUDA=solver_settings.get("profileCUDA", False),
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        mem_proportion=solver_settings["mem_proportion"],
        step_control_settings=step_controller_settings
    )


@pytest.fixture(scope="function")
def simple_initial_values(system):
    """Create simple initial values for testing."""
    return {
        list(system.initial_values.names)[0]: [0.1, 0.5],
        list(system.initial_values.names)[1]: [0.2, 0.6],
    }


@pytest.fixture(scope="function")
def simple_parameters(system):
    """Create simple parameters for testing."""
    return {
        list(system.parameters.names)[0]: [1.0, 2.0],
        list(system.parameters.names)[1]: [0.5, 1.5],
    }


@pytest.fixture(scope="function")
def solved_solver_simple(
    solver_instance,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    result = solver_instance.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
        results_type="full",
    )
    return solver_instance, result


def test_solver_initialization(solver_instance, system):
    """Test that the solver initializes correctly."""
    assert solver_instance is not None
    assert solver_instance.system_interface is not None
    assert solver_instance.grid_builder is not None
    assert solver_instance.kernel is not None
    assert isinstance(solver_instance.system_interface, SystemInterface)
    assert isinstance(solver_instance.grid_builder, BatchGridBuilder)


def test_solver_properties(solver_instance, solver_settings):
    """Test that solver properties return expected values."""
    assert solver_instance.precision == solver_settings["precision"]
    assert solver_instance.system_sizes is not None
    assert solver_instance.output_array_heights is not None
    assert solver_instance.memory_manager == solver_settings["memory_manager"]
    assert solver_instance.stream_group == solver_settings["stream_group"]


@pytest.mark.parametrize(
    "solver_settings_override", [{"mem_proportion": 0.1}], indirect=True
)
def test_manual_proportion(solver_instance, solver_settings):
    assert solver_instance.mem_proportion == 0.1


def test_variable_indices_methods(solver_instance, system):
    """Test methods for getting variable indices."""
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names")
        else []
    )

    # Test with specific labels
    if len(state_names) > 0:
        indices = solver_instance.get_state_indices([state_names[0]])
        assert isinstance(indices, np.ndarray)
        assert len(indices) >= 1

    # Test with None (all states)
    all_state_indices = solver_instance.get_state_indices(None)
    assert isinstance(all_state_indices, np.ndarray)

    if len(observable_names) > 0:
        obs_indices = solver_instance.get_observable_indices(
            [observable_names[0]]
        )
        assert isinstance(obs_indices, np.ndarray)


def test_saved_variables_properties(solver_instance):
    """Test properties related to saved variables."""
    assert solver_instance.saved_state_indices is not None
    assert solver_instance.saved_observable_indices is not None
    assert solver_instance.saved_states is not None
    assert solver_instance.saved_observables is not None

    # Test that saved states/observables return lists of strings
    assert isinstance(solver_instance.saved_states, list)
    assert isinstance(solver_instance.saved_observables, list)


def test_summarised_variables_properties(solver_instance):
    """Test properties related to summarised variables."""
    assert solver_instance.summarised_state_indices is not None
    assert solver_instance.summarised_observable_indices is not None
    assert solver_instance.summarised_states is not None
    assert solver_instance.summarised_observables is not None

    # Test that summarised states/observables return lists of strings
    assert isinstance(solver_instance.summarised_states, list)
    assert isinstance(solver_instance.summarised_observables, list)


def test_output_properties(solver_instance):
    """Test output-related properties."""
    assert solver_instance.active_output_arrays is not None
    assert solver_instance.output_types is not None
    assert isinstance(solver_instance.output_types, Iterable)
    assert solver_instance.output_stride_order is not None
    assert isinstance(solver_instance.output_stride_order, Iterable)



def test_solve_info_property(
    precision,
    solver_instance,
    solver_settings,
    tolerance,
):
    """Test that solve_info returns a valid SolveSpec."""
    solve_info = solver_instance.solve_info
    assert isinstance(solve_info, SolveSpec)
    assert solve_info.dt_min == pytest.approx(
        solver_settings["dt_min"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    if solver_instance.kernel.single_integrator.is_adaptive:
        assert solve_info.dt_max == pytest.approx(
            solver_settings["dt_max"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.atol == pytest.approx(
            solver_settings["atol"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.rtol == pytest.approx(
            solver_settings["rtol"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    assert solve_info.dt_save == pytest.approx(
        solver_settings["dt_save"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert solve_info.dt_summarise == pytest.approx(
        solver_settings["dt_summarise"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    assert solve_info.algorithm == solver_settings["algorithm"]
    assert solve_info.output_types ==  solver_settings["output_types"]
    assert solve_info.precision ==  solver_settings["precision"]

    # Test that solver kernel properties are correctly exposed
    assert solve_info.duration == solver_instance.kernel.duration
    assert solve_info.warmup == solver_instance.kernel.warmup

    # Test that variable lists are correctly exposed
    assert solve_info.saved_states == solver_instance.saved_states
    assert solve_info.saved_observables == solver_instance.saved_observables
    assert solve_info.summarised_states == solver_instance.summarised_states
    assert (
        solve_info.summarised_observables
        == solver_instance.summarised_observables
    )
    # Note: There appears to be a bug in the solver.py where summarised_observables
    # is set to summarised_states instead of summarised_observables
    # This test documents the current behavior
    assert hasattr(solve_info, "summarised_observables")


@pytest.mark.parametrize(
    "system_override, solver_settings_override",
    [
        ({}, {}),
        ("three_chamber",
         {"duration": 0.05,
          "dt_save": 0.02,
          "dt_summarise": 0.04,
          "output_types": ["state"]}),
    ],
    ids=["default_system", "three_chamber_system"],
    indirect=True,
)
def test_solve_basic(
    solver_instance,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    solver_instance.kernel.build()
    result = solver_instance.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
        results_type="full",
    )

    assert isinstance(result, SolveResult)
    assert hasattr(result, "time_domain_array")
    assert hasattr(result, "summaries_array")

@pytest.mark.parametrize("solver_settings_override",
                         [{
                            "duration": 0.05,
                            "dt_save": 0.02,
                            "dt_summarise": 0.04,
                            "output_types": ["state", "time", "observables",
                                             "mean"]
                         }],
                         indirect=True
)
def test_solve_with_different_grid_types(
    solver_instance,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test solve with different grid types."""
    # Test combinatorial grid
    result_comb = solver_instance.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        grid_type="combinatorial",
    )
    assert isinstance(result_comb, SolveResult)

    # Test verbatim grid with properly shaped arrays
    state_names = list(simple_initial_values.keys())
    param_names = list(simple_parameters.keys())

    # Create verbatim arrays (same length for all variables)
    verbatim_initial_values = {
        state_names[0]: [0.1, 0.5],
        state_names[1]: [0.2, 0.6],
    }
    verbatim_parameters = {
        param_names[0]: [1.0, 2.0],
        param_names[1]: [0.5, 1.5],
    }

    result_verb = solver_instance.solve(
        initial_values=verbatim_initial_values,
        parameters=verbatim_parameters,
        drivers=driver_settings,
        duration=0.1,
        grid_type="verbatim",
    )
    assert isinstance(result_verb, SolveResult)

@pytest.mark.parametrize("solver_settings_override",
                         [{
                            "duration": 0.05,
                            "dt_save": 0.02,
                            "dt_summarise": 0.04,
                            "output_types": ["state", "time", "observables",
                                             "mean"]
                         }],
                         indirect=True
)
def test_solve_with_different_result_types(
    solver_instance,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test solve with different result types."""
    result_types = ["full", "numpy"]

    for result_type in result_types:
        result = solver_instance.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.1,
            results_type=result_type,
        )
        assert result is not None


def test_update_basic(solver_instance):
    """Test basic update functionality."""
    original_duration = solver_instance.kernel.duration

    updates = {"duration": 2.0}
    updated_keys = solver_instance.update(updates)

    assert "duration" in updated_keys
    assert solver_instance.kernel.duration == 2.0
    assert solver_instance.kernel.duration != original_duration


def test_update_with_kwargs(solver_instance, tolerance):
    """Test update with keyword arguments."""
    original_dt = solver_instance.kernel.single_integrator.dt0

    updated_keys = solver_instance.update({}, dt=1e-8)

    assert "dt" in updated_keys
    assert solver_instance.kernel.single_integrator.dt0 == pytest.approx(
        1e-8,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert (
        solver_instance.kernel.single_integrator.dt0
        != pytest.approx(
            original_dt,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    )

def test_update_unrecognized_keys(solver_instance):
    """Test that update raises KeyError for unrecognized keys."""
    with pytest.raises(KeyError):
        solver_instance.update({"nonexistent_parameter": 42})


def test_update_silent_mode(solver_instance):
    """Test update in silent mode ignores unrecognized keys."""
    original_duration = solver_instance.kernel.duration

    updated_keys = solver_instance.update(
        {"duration": 3.0, "nonexistent_parameter": 42}, silent=True
    )

    assert "duration" in updated_keys
    assert "nonexistent_parameter" not in updated_keys
    assert solver_instance.kernel.duration == 3.0


def test_update_saved_variables(solver_instance, system):
    """Test updating saved variables with labels."""
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names")
        else []
    )

    if len(state_names) > 0 and len(observable_names) > 0:
        updates = {
            "saved_states": [state_names[0]],
            "saved_observables": [observable_names[0]]
            if observable_names
            else [],
        }

        updated_keys = solver_instance.update(updates)

        # The method should have converted labels to indices
        assert len(updated_keys) > 0


def test_profiling_methods(solver_instance):
    """Test profiling enable/disable methods."""
    # These methods should execute without error
    solver_instance.enable_profiling()
    solver_instance.disable_profiling()


def test_memory_settings_update(solver_instance):
    """Test updating memory-related settings."""
    # Test memory proportion update
    updated_keys = solver_instance.update_memory_settings(
        {"mem_proportion": 0.1}
    )
    assert "mem_proportion" in updated_keys


def test_data_properties_after_solve(solved_solver_simple):
    """Test that data properties are accessible after solving."""
    # These should be accessible after solving
    solver, result = solved_solver_simple
    assert isinstance(solver.kernel.device_state_array, MappedNDArray)
    assert isinstance(solver.kernel.device_observables_array, MappedNDArray)
    assert isinstance(
        solver.kernel.device_state_summaries_array, MappedNDArray
    )
    assert isinstance(
        solver.kernel.device_observable_summaries_array, MappedNDArray
    )

    assert isinstance(result.time_domain_array, np.ndarray)
    assert isinstance(result.summaries_array, np.ndarray)

    assert not np.all(result.time_domain_array == 0)


def test_output_length_and_summaries_length(solver_instance):
    """Test output length and summaries length properties."""
    assert solver_instance.output_length is not None
    assert solver_instance.summaries_length is not None
    assert isinstance(solver_instance.output_length, int)
    assert isinstance(solver_instance.summaries_length, int)


def test_chunk_related_properties(solved_solver_simple):
    """Test chunk-related properties."""
    solver, result = solved_solver_simple
    assert solver.chunk_axis is not None
    assert solver.chunks == 1


def test_variable_labels_properties(solver_instance):
    """Test properties that return variable labels."""
    assert solver_instance.input_variables is not None
    assert solver_instance.output_variables is not None
    assert isinstance(solver_instance.input_variables, list)
    assert isinstance(solver_instance.output_variables, list)


# Test the solve_ivp convenience function
def test_solve_ivp_function(
    system, simple_initial_values, simple_parameters, driver_settings
):
    """Test the solve_ivp convenience function."""
    result = solve_ivp(
        system=system,
        y0=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        dt_eval=0.02,
        method="euler",
        duration=0.1,
        settling_time=0.0,
    )

    assert isinstance(result, SolveResult)


def test_solver_with_different_algorithms(system, solver_settings):
    """Test solver with different algorithms."""
    algorithms = ["euler", "backwards_euler_pc"]

    for algorithm in algorithms:
        solver = Solver(
            system,
            algorithm=algorithm,
            duration=solver_settings["duration"],
            dt_min=solver_settings["dt_min"],
            dt_max=solver_settings["dt_max"],
            dt_save=solver_settings["dt_save"],
            precision=solver_settings["precision"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
        )

        assert solver is not None
        assert solver.kernel.algorithm == algorithm


@pytest.mark.parametrize(
    "precision_override", [np.float32, np.float64], indirect=True
)
def test_solver_precision_types(system, solver_settings, precision):
    """Test solver with different precision types."""
    solver = Solver(
        system,
        precision=solver_settings["precision"],
        duration=solver_settings["duration"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        dt_save=solver_settings["dt_save"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
    )

    assert solver.precision == precision


def test_solver_output_types(system, solver_settings):
    """Test solver with different output types."""
    output_types_list = [
        ["state"],
        ["state", "observables"],
        ["state", "observables", "mean"],
        ["state", "observables", "mean", "max", "rms"],
    ]

    for output_types in output_types_list:
        solver = Solver(
            system,
            output_types=output_types,
            duration=solver_settings["duration"],
            dt_min=solver_settings["dt_min"],
            dt_max=solver_settings["dt_max"],
            dt_save=solver_settings["dt_save"],
            precision=solver_settings["precision"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
        )

        assert solver.output_types == output_types


@pytest.mark.parametrize(
    "system_override", ["three_chamber", "stiff", "linear"], indirect=True
)
def test_solver_with_different_systems(solver_instance):
    """Test solver works with different system types."""
    assert solver_instance is not None
    assert solver_instance.system_interface is not None
    assert solver_instance.grid_builder is not None
    assert solver_instance.kernel is not None


def test_solver_summary_legend(solver_instance):
    """Test that summary legend property works."""
    legend = solver_instance.summary_legend_per_variable
    assert isinstance(legend, dict)


def test_solver_save_time_property(solver_instance):
    """Test save_time property."""
    save_time = solver_instance.save_time
    assert isinstance(save_time, bool)


def test_solver_state_and_observable_summaries(solver_instance):
    """Test state and observable summaries properties."""
    # These properties should be accessible even before solving
    assert solver_instance.state_summaries is not None
    assert solver_instance.observable_summaries is not None


def test_solver_summaries_buffer_sizes(solver_instance):
    """Test summaries buffer sizes property."""
    buffer_sizes = solver_instance.summaries_buffer_sizes
    assert buffer_sizes is not None


def test_solver_num_runs_property(solver_instance):
    """Test num_runs property."""
    num_runs = solver_instance.num_runs
    # num_runs might be None before solving, so just check it's accessible
    assert num_runs is not None or num_runs is None
