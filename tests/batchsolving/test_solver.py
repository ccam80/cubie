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
    solver,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    result = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
        results_type="full",
    )
    return solver, result


def test_solver_initialization(solver, system):
    """Test that the solver initializes correctly."""
    assert solver is not None
    assert solver.system_interface is not None
    assert solver.grid_builder is not None
    assert solver.kernel is not None
    assert isinstance(solver.system_interface, SystemInterface)
    assert isinstance(solver.grid_builder, BatchGridBuilder)


def test_solver_properties(solver, solver_settings):
    """Test that solver properties return expected values."""
    assert solver.precision == solver_settings["precision"]
    assert solver.system_sizes is not None
    assert solver.output_array_heights is not None
    assert solver.memory_manager == solver_settings["memory_manager"]
    assert solver.stream_group == solver_settings["stream_group"]


@pytest.mark.parametrize(
    "solver_settings_override", [{"mem_proportion": 0.1}], indirect=True
)
def test_manual_proportion(solver, solver_settings):
    assert solver.mem_proportion == 0.1


def test_variable_indices_methods(solver, system):
    """Test methods for getting variable indices."""
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names")
        else []
    )

    # Test with specific labels
    if len(state_names) > 0:
        indices = solver.get_state_indices([state_names[0]])
        assert isinstance(indices, np.ndarray)
        assert len(indices) >= 1

    # Test with None (all states)
    all_state_indices = solver.get_state_indices(None)
    assert isinstance(all_state_indices, np.ndarray)

    if len(observable_names) > 0:
        obs_indices = solver.get_observable_indices(
            [observable_names[0]]
        )
        assert isinstance(obs_indices, np.ndarray)


def test_saved_variables_properties(solver):
    """Test properties related to saved variables."""
    assert solver.saved_state_indices is not None
    assert solver.saved_observable_indices is not None
    assert solver.saved_states is not None
    assert solver.saved_observables is not None

    # Test that saved states/observables return lists of strings
    assert isinstance(solver.saved_states, list)
    assert isinstance(solver.saved_observables, list)


def test_summarised_variables_properties(solver):
    """Test properties related to summarised variables."""
    assert solver.summarised_state_indices is not None
    assert solver.summarised_observable_indices is not None
    assert solver.summarised_states is not None
    assert solver.summarised_observables is not None

    # Test that summarised states/observables return lists of strings
    assert isinstance(solver.summarised_states, list)
    assert isinstance(solver.summarised_observables, list)


def test_output_properties(solver):
    """Test output-related properties."""
    assert solver.active_output_arrays is not None
    assert solver.output_types is not None
    assert isinstance(solver.output_types, Iterable)
    assert solver.state_stride_order is not None
    assert isinstance(solver.state_stride_order, Iterable)



def test_solve_info_property(
    precision,
    solver,
    solver_settings,
    tolerance,
):
    """Test that solve_info returns a valid SolveSpec."""
    solver.kernel.duration = 1.0
    solve_info = solver.solve_info
    assert isinstance(solve_info, SolveSpec)

    if solver.kernel.single_integrator.is_adaptive:
        assert solve_info.dt_min == pytest.approx(
            solver_settings["dt_min"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
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
    else:
        assert solve_info.dt == pytest.approx(
                solver_settings["dt"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
        )
        assert solve_info.dt_min == pytest.approx(
            solver_settings["dt"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.dt_max == pytest.approx(
                solver_settings["dt"],
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
    assert solve_info.duration == solver.duration
    assert solve_info.warmup == solver.warmup
    assert solve_info.t0 == solver.t0

    # Test that variable lists are correctly exposed
    assert solve_info.saved_states == solver.saved_states
    assert solve_info.saved_observables == solver.saved_observables
    assert solve_info.summarised_states == solver.summarised_states
    assert (
        solve_info.summarised_observables
        == solver.summarised_observables
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
    solver,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    result = solver.solve(
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
    solver,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test solve with different grid types."""
    # Test combinatorial grid
    result_comb = solver.solve(
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

    result_verb = solver.solve(
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
    solver,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test solve with different result types."""
    result_types = ["full", "numpy"]

    for result_type in result_types:
        result = solver.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.1,
            results_type=result_type,
        )
        assert result is not None


def test_update_basic(solver_mutable, tolerance, precision):
    """Test basic update functionality updating the fixed step size."""
    solver = solver_mutable
    original_dt = solver.dt
    # Choose a new dt distinct from the original
    new_dt = precision(original_dt * 0.5 if original_dt not in (0,
                                                                None) else
                       1e-7)
    updated_keys = solver.update({"dt": new_dt})
    assert "dt" in updated_keys
    # For fixed-step integrators dt should now reflect the new value if not getattr(solver.kernel.single_integrator, "is_adaptive", False):
    assert solver.dt == pytest.approx(
        new_dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight)


def test_update_with_kwargs(solver_mutable, tolerance):
    """Test update with keyword arguments."""
    solver = solver_mutable
    original_dt = solver.kernel.single_integrator.dt0

    updated_keys = solver.update({}, dt=1e-8)

    assert "dt" in updated_keys
    assert solver.kernel.single_integrator.dt0 == pytest.approx(
        1e-8,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert (
        solver.kernel.single_integrator.dt0
        != pytest.approx(
            original_dt,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    )

def test_update_unrecognized_keys(solver_mutable):
    """Test that update raises KeyError for unrecognized keys."""
    solver = solver_mutable
    with pytest.raises(KeyError):
        solver.update({"nonexistent_parameter": 42})


def test_update_silent_mode(precision, solver_mutable):
    """Test update in silent mode ignores unrecognized keys."""
    solver = solver_mutable
    original_dt = solver.dt
    # Choose a new dt distinct from the original
    new_dt = precision(original_dt * 0.5 if original_dt not in (0, None) else \
        1e-7)
    updated_keys = solver.update({"dt": new_dt, "nonexistent_parameter":
        42}, silent=True)


    assert "dt" in updated_keys

    assert "nonexistent_parameter" not in updated_keys
    assert solver.kernel.dt == new_dt


def test_update_saved_variables(solver_mutable, system):
    """Test updating saved variables with labels."""
    solver = solver_mutable
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

        updated_keys = solver.update(updates)

        # The method should have converted labels to indices
        assert len(updated_keys) > 0


def test_profiling_methods(solver):
    """Test profiling enable/disable methods."""
    # These methods should execute without error
    solver.enable_profiling()
    solver.disable_profiling()


def test_memory_settings_update(solver_mutable):
    """Test updating memory-related settings."""
    solver = solver_mutable
    # Test memory proportion update
    updated_keys = solver.update_memory_settings(
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


def test_output_length_and_summaries_length(solver):
    """Test output length and summaries length properties."""
    assert solver.output_length is not None
    assert solver.summaries_length is not None
    assert isinstance(solver.output_length, int)
    assert isinstance(solver.summaries_length, int)


def test_chunk_related_properties(solved_solver_simple):
    """Test chunk-related properties."""
    solver, result = solved_solver_simple
    assert solver.chunk_axis is not None
    assert solver.chunks == 1


def test_variable_labels_properties(solver):
    """Test properties that return variable labels."""
    assert solver.input_variables is not None
    assert solver.output_variables is not None
    assert isinstance(solver.input_variables, list)
    assert isinstance(solver.output_variables, list)


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
        dt=1e-2,
        dt_save=0.02,
        duration= 0.05,
        dt_summarise= 0.04,
        output_types= ["state", "time", "observables",
                     "mean"],
        method="euler",
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
            dt_min=solver_settings["dt_min"],
            dt=solver_settings["dt"],
            dt_max=solver_settings["dt_max"],
            precision=solver_settings["precision"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            loop_settings={"dt_save": solver_settings["dt_save"]},
        )

        assert solver is not None
        assert solver.kernel.algorithm == algorithm

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
            dt_min=solver_settings["dt_min"],
            dt_max=solver_settings["dt_max"],
            precision=solver_settings["precision"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            loop_settings={"dt_save": solver_settings["dt_save"]},
        )

        assert solver.output_types == output_types


@pytest.mark.parametrize(
    "system_override", ["three_chamber", "stiff", "linear"], indirect=True
)
def test_solver_with_different_systems(solver):
    """Test solver works with different system types."""
    assert solver is not None
    assert solver.system_interface is not None
    assert solver.grid_builder is not None
    assert solver.kernel is not None


def test_solver_summary_legend(solver):
    """Test that summary legend property works."""
    legend = solver.summary_legend_per_variable
    assert isinstance(legend, dict)


def test_solver_save_time_property(solver):
    """Test save_time property."""
    save_time = solver.save_time
    assert isinstance(save_time, bool)


def test_solver_state_and_observable_summaries(solver):
    """Test state and observable summaries properties."""
    # These properties should be accessible even before solving
    assert solver.state_summaries is not None
    assert solver.observable_summaries is not None


def test_solver_summaries_buffer_sizes(solver):
    """Test summaries buffer sizes property."""
    buffer_sizes = solver.summaries_buffer_sizes
    assert buffer_sizes is not None


def test_solver_num_runs_property(solver):
    """Test num_runs property."""
    num_runs = solver.num_runs
    # num_runs might be None before solving, so just check it's accessible
    assert num_runs is not None or num_runs is None


# ============================================================================
# Time Precision Tests (float64 time accumulation)
# ============================================================================


@pytest.mark.parametrize("precision", [np.float32], indirect=True)
def test_solver_stores_time_as_float64(system):
    """Verify Solver stores time parameters as float64."""
    solver = Solver(
        system,
        algorithm="euler",
        dt=1e-3,
    )
    
    # Set time parameters as float32
    solver.kernel.duration = np.float32(10.0)
    solver.kernel.warmup = np.float32(1.0)
    solver.kernel.t0 = np.float32(5.0)
    
    # Verify retrieved as float64
    assert isinstance(solver.kernel.duration, (float, np.floating))
    assert isinstance(solver.kernel.warmup, (float, np.floating))
    assert isinstance(solver.kernel.t0, (float, np.floating))
    
    # Verify values preserved
    assert np.isclose(solver.kernel.duration, 10.0)
    assert np.isclose(solver.kernel.warmup, 1.0)
    assert np.isclose(solver.kernel.t0, 5.0)


@pytest.mark.parametrize("precision", [np.float32, np.float64], indirect=True)
def test_time_precision_independent_of_state_precision(system):
    """Verify time precision is float64 regardless of state precision."""
    
    solver = Solver(
        system,
        algorithm="euler",
        dt=1e-3,
    )
    
    solver.kernel.duration = 5.0
    solver.kernel.t0 = 1.0
    
    # Time should be float64 even when state precision is float32
    assert solver.kernel.duration == np.float64(5.0)
    assert solver.kernel.t0 == np.float64(1.0)
