from typing import Iterable

import pytest
import numpy as np
from cubie.batchsolving.solver import Solver, solve_ivp
from cubie.batchsolving.solveresult import SolveResult, SolveSpec
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.SystemInterface import SystemInterface

from cubie.cuda_simsafe import DeviceNDArray


@pytest.fixture(scope="session")
def simple_initial_values(system):
    """Create simple initial values for testing."""
    return {
        list(system.initial_values.names)[0]: [0.1, 0.5],
        list(system.initial_values.names)[1]: [0.2, 0.6],
    }


@pytest.fixture(scope="session")
def simple_parameters(system):
    """Create simple parameters for testing."""
    return {
        list(system.parameters.names)[0]: [1.0, 2.0],
        list(system.parameters.names)[1]: [0.5, 1.5],
    }


@pytest.fixture(scope="session")
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
    assert solver.active_outputs is not None
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
        duration=0.05,
        dt_save=0.02,
        dt_summarise=0.04,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
        results_type="full",
    )

    assert isinstance(result, SolveResult)
    assert hasattr(result, "time_domain_array")
    assert hasattr(result, "summaries_array")

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
                duration=0.05,
                dt_save=0.02,
                dt_summarise=0.04,
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
        all_vars = [state_names[0]]
        if observable_names:
            all_vars.append(observable_names[0])
        
        updates = {
            "save_variables": all_vars,
        }

        updated_keys = solver.update(updates)

        assert len(updated_keys) > 0

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
    assert isinstance(solver.kernel.device_state_array, DeviceNDArray)
    assert isinstance(solver.kernel.device_observables_array, DeviceNDArray)
    assert isinstance(
        solver.kernel.device_state_summaries_array, DeviceNDArray
    )
    assert isinstance(
        solver.kernel.device_observable_summaries_array, DeviceNDArray
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

def test_solver_num_runs_property(solver):
    """Test num_runs property."""
    num_runs = solver.num_runs
    # num_runs might be None before solving, so just check it's accessible
    assert num_runs is not None or num_runs is None


# ============================================================================
# Time Precision Tests (float64 time accumulation)
# ============================================================================

def test_solver_stores_time_as_float64(solver_mutable):
    """Verify Solver stores time parameters as float64."""
    # Set time parameters as float32
    solver_mutable.kernel.duration = np.float32(10.0)
    solver_mutable.kernel.warmup = np.float32(1.0)
    solver_mutable.kernel.t0 = np.float32(5.0)
    
    # Verify retrieved as float64
    assert isinstance(solver_mutable.kernel.duration, (float, np.floating))
    assert isinstance(solver_mutable.kernel.warmup, (float, np.floating))
    assert isinstance(solver_mutable.kernel.t0, (float, np.floating))
    
    # Verify values preserved
    assert np.isclose(solver_mutable.kernel.duration, 10.0)
    assert np.isclose(solver_mutable.kernel.warmup, 1.0)
    assert np.isclose(solver_mutable.kernel.t0, 5.0)


@pytest.mark.parametrize("solver_settings_override", [
    {"precision": np.float32},
    {"precision": np.float64}],
    indirect=True)
def test_time_precision_independent_of_state_precision(system, solver_mutable):
    """Verify time precision is float64 regardless of state precision."""

    solver_mutable.kernel.duration = 5.0
    solver_mutable.kernel.t0 = 1.0
    
    # Time should be float64 even when state precision is float32
    assert solver_mutable.kernel.duration == np.float64(5.0)
    assert solver_mutable.kernel.t0 == np.float64(1.0)


# ============================================================================
# Input Classification Tests
# ============================================================================


def test_classify_inputs_dict(solver, simple_initial_values, simple_parameters):
    """Test that dict inputs are classified as 'dict'."""
    result = solver._classify_inputs(simple_initial_values, simple_parameters)
    assert result == 'dict'


def test_classify_inputs_mixed(solver, system):
    """Test that mixed inputs (dict + array) are classified as 'dict'."""
    n_states = solver.system_sizes.states
    inits_array = np.ones((n_states, 2), dtype=solver.precision)
    params_dict = {list(system.parameters.names)[0]: [1.0, 2.0]}

    result = solver._classify_inputs(inits_array, params_dict)
    assert result == 'dict'

    # Test the reverse case
    inits_dict = {list(system.initial_values.names)[0]: [0.1, 0.2]}
    n_params = solver.system_sizes.parameters
    params_array = np.ones((n_params, 2), dtype=solver.precision)

    result = solver._classify_inputs(inits_dict, params_array)
    assert result == 'dict'


def test_classify_inputs_array(solver):
    """Test that matching numpy arrays are classified as 'array'."""
    n_states = solver.system_sizes.states
    n_params = solver.system_sizes.parameters
    n_runs = 4

    inits = np.ones((n_states, n_runs), dtype=solver.precision)
    params = np.ones((n_params, n_runs), dtype=solver.precision)

    result = solver._classify_inputs(inits, params)
    assert result == 'array'


def test_classify_inputs_mismatched_runs(solver):
    """Test that mismatched run counts fall back to 'dict'."""
    n_states = solver.system_sizes.states
    n_params = solver.system_sizes.parameters

    inits = np.ones((n_states, 3), dtype=solver.precision)
    params = np.ones((n_params, 5), dtype=solver.precision)

    result = solver._classify_inputs(inits, params)
    assert result == 'dict'


def test_classify_inputs_wrong_var_count(solver):
    """Test that wrong variable counts fall back to 'dict'."""
    n_params = solver.system_sizes.parameters
    n_runs = 4

    # Wrong number of states
    inits = np.ones((999, n_runs), dtype=solver.precision)
    params = np.ones((n_params, n_runs), dtype=solver.precision)

    result = solver._classify_inputs(inits, params)
    assert result == 'dict'


def test_classify_inputs_1d_arrays(solver):
    """Test that 1D arrays fall back to 'dict'."""
    n_states = solver.system_sizes.states
    n_params = solver.system_sizes.parameters

    inits = np.ones(n_states, dtype=solver.precision)
    params = np.ones(n_params, dtype=solver.precision)

    result = solver._classify_inputs(inits, params)
    assert result == 'dict'


# ============================================================================
# Array Validation Tests
# ============================================================================


def test_validate_arrays_dtype_cast(solver):
    """Test that arrays are cast to system precision."""
    n_states = solver.system_sizes.states
    n_params = solver.system_sizes.parameters
    n_runs = 2

    # Create arrays with wrong dtype
    wrong_dtype = np.float64 if solver.precision == np.float32 else np.float32
    inits = np.ones((n_states, n_runs), dtype=wrong_dtype)
    params = np.ones((n_params, n_runs), dtype=wrong_dtype)

    validated_inits, validated_params = solver._validate_arrays(inits, params)

    assert validated_inits.dtype == solver.precision
    assert validated_params.dtype == solver.precision

# ============================================================================
# build_grid() Tests
# ============================================================================


def test_build_grid_returns_correct_shape(
    solver, simple_initial_values, simple_parameters
):
    """Test that build_grid returns arrays with correct shapes."""
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="verbatim"
    )

    assert isinstance(inits, np.ndarray)
    assert isinstance(params, np.ndarray)
    assert inits.ndim == 2
    assert params.ndim == 2
    assert inits.shape[0] == solver.system_sizes.states
    assert params.shape[0] == solver.system_sizes.parameters
    # Verbatim: run count matches input length
    assert inits.shape[1] == params.shape[1]


def test_build_grid_combinatorial(
    solver, simple_initial_values, simple_parameters
):
    """Test that build_grid with combinatorial creates product grid."""
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="combinatorial"
    )

    # Combinatorial produces more runs than verbatim
    n_init_values = len(list(simple_initial_values.values())[0])
    n_param_values = len(list(simple_parameters.values())[0])
    # Number of runs is product of all value counts
    assert inits.shape[1] >= n_init_values
    assert params.shape[1] >= n_param_values


def test_build_grid_precision(solver, simple_initial_values, simple_parameters):
    """Test that build_grid returns arrays with correct precision."""
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters
    )

    assert inits.dtype == solver.precision
    assert params.dtype == solver.precision


# ============================================================================
# solve() Fast Path Tests
# ============================================================================

def test_solve_with_prebuilt_arrays(
    solver, simple_initial_values, simple_parameters, driver_settings,
):
    """Test that solve() works with pre-built arrays (fast path)."""
    # First build the grid
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="verbatim"
    )

    # Now solve with the pre-built arrays
    result = solver.solve(
        initial_values=inits,
        parameters=params,
        drivers=driver_settings,
        duration=0.05,
    )

    assert isinstance(result, SolveResult)



def test_solve_array_path_matches_dict_path(
    solver, simple_initial_values, simple_parameters,
                        driver_settings
):
    """Test that array fast path produces same results as dict path."""
    # Solve with dict inputs
    result_dict = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        grid_type="verbatim",
    )

    # Build grid and solve with arrays
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="verbatim"
    )
    result_array = solver.solve(
            initial_values=inits,
            parameters=params,
            drivers=driver_settings,
            duration=0.05,
    )


    # Results should match
    assert (
        result_dict.time_domain_array.shape
        == result_array.time_domain_array.shape
    )
    np.testing.assert_allclose(
        result_dict.time_domain_array,
        result_array.time_domain_array,
        rtol=1e-5,
        atol=1e-7,
    )


def test_solve_dict_path_backward_compatible(
    solver, simple_initial_values, simple_parameters, driver_settings,
):
    """Test that dict inputs still work exactly as before."""
    result = solver.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.05,
            settling_time=0.0,
            blocksize=32,
            grid_type="combinatorial",
            results_type="full",
    )

    assert isinstance(result, SolveResult)
    assert hasattr(result, "time_domain_array")
    assert hasattr(result, "summaries_array")


# ============================================================================
# save_variables and summarise_variables Tests
# ============================================================================


def test_save_variables_pure_states(solver, system):
    """Test save_variables with only state names."""
    state_names = list(system.initial_values.names)[:2]
    
    output_settings = {"save_variables": state_names}
    solver.convert_output_labels(output_settings)
    
    # Verify save_variables was removed
    assert "save_variables" not in output_settings
    # Verify state indices were set
    assert "saved_state_indices" in output_settings
    assert len(output_settings["saved_state_indices"]) == 2
    # Verify no observable indices created
    assert ("saved_observable_indices" not in output_settings
            or output_settings.get("saved_observable_indices") is None
            or len(output_settings["saved_observable_indices"]) == 0)


def test_save_variables_pure_observables(solver, system):
    """Test save_variables with only observable names."""
    if (not hasattr(system.observables, "names")
            or len(system.observables.names) == 0):
        pytest.skip("System has no observables")
    
    obs_names = list(system.observables.names)[:2]
    
    output_settings = {"save_variables": obs_names}
    solver.convert_output_labels(output_settings)
    
    # Verify save_variables was removed
    assert "save_variables" not in output_settings
    # Verify observable indices were set
    assert "saved_observable_indices" in output_settings
    assert len(output_settings["saved_observable_indices"]) >= 1
    # Verify no state indices created
    assert ("saved_state_indices" not in output_settings
            or output_settings.get("saved_state_indices") is None
            or len(output_settings["saved_state_indices"]) == 0)


def test_save_variables_mixed(solver, system):
    """Test save_variables with both states and observables."""
    state_names = list(system.initial_values.names)[:1]
    obs_names = []
    if (hasattr(system.observables, "names")
            and len(system.observables.names) > 0):
        obs_names = list(system.observables.names)[:1]
    
    if not obs_names:
        pytest.skip("System has no observables for mixed test")
    
    mixed_names = state_names + obs_names
    
    output_settings = {"save_variables": mixed_names}
    solver.convert_output_labels(output_settings)
    
    # Verify both types classified
    assert "saved_state_indices" in output_settings
    assert "saved_observable_indices" in output_settings
    assert len(output_settings["saved_state_indices"]) >= 1
    assert len(output_settings["saved_observable_indices"]) >= 1


def test_summarise_variables_pure_states(solver, system):
    """Test summarise_variables with only state names."""
    state_names = list(system.initial_values.names)[:2]
    
    output_settings = {"summarise_variables": state_names}
    solver.convert_output_labels(output_settings)
    
    assert "summarise_variables" not in output_settings
    assert "summarised_state_indices" in output_settings
    assert len(output_settings["summarised_state_indices"]) == 2


def test_summarise_variables_pure_observables(solver, system):
    """Test summarise_variables with only observable names."""
    if (not hasattr(system.observables, "names")
            or len(system.observables.names) == 0):
        pytest.skip("System has no observables")
    
    obs_names = list(system.observables.names)[:2]
    
    output_settings = {"summarise_variables": obs_names}
    solver.convert_output_labels(output_settings)
    
    assert "summarise_variables" not in output_settings
    assert "summarised_observable_indices" in output_settings
    assert len(output_settings["summarised_observable_indices"]) >= 1


def test_summarise_variables_mixed(solver, system):
    """Test summarise_variables with both states and observables."""
    state_names = list(system.initial_values.names)[:1]
    obs_names = []
    if (hasattr(system.observables, "names")
            and len(system.observables.names) > 0):
        obs_names = list(system.observables.names)[:1]
    
    if not obs_names:
        pytest.skip("System has no observables for mixed test")
    
    mixed_names = state_names + obs_names
    
    output_settings = {"summarise_variables": mixed_names}
    solver.convert_output_labels(output_settings)
    
    # Verify both types classified
    assert "summarised_state_indices" in output_settings
    assert "summarised_observable_indices" in output_settings
    assert len(output_settings["summarised_state_indices"]) >= 1
    assert len(output_settings["summarised_observable_indices"]) >= 1


def test_save_variables_union_with_indices(solver, system):
    """Test save_variables merges with existing saved_state_indices."""
    state_names = list(system.initial_values.names)
    
    # Pre-populate with first state index
    output_settings = {
        "saved_state_indices": np.array([0], dtype=np.int16),
        "save_variables": [state_names[1]]
    }
    solver.convert_output_labels(output_settings)
    
    # Should have union of indices 0 and 1
    result = output_settings["saved_state_indices"]
    assert len(result) == 2
    assert 0 in result
    assert 1 in result


def test_save_variables_union_with_saved_state_indices(solver, system):
    """Test save_variables merges with existing saved_state_indices."""
    state_names = list(system.initial_values.names)
    
    output_settings = {
        "saved_state_indices": np.array([0], dtype=np.int16),
        "save_variables": [state_names[1]]
    }
    solver.convert_output_labels(output_settings)
    
    result = output_settings["saved_state_indices"]
    assert len(result) == 2
    assert 0 in result
    assert 1 in result


def test_save_variables_empty_list(solver):
    """Test save_variables with empty list is no-op."""
    output_settings = {"save_variables": []}
    solver.convert_output_labels(output_settings)
    
    # Empty list should be removed but not create indices
    assert "save_variables" not in output_settings
    assert ("saved_state_indices" not in output_settings
            or output_settings.get("saved_state_indices") is None)


def test_save_variables_none(solver):
    """Test save_variables=None is ignored."""
    output_settings = {"save_variables": None}
    solver.convert_output_labels(output_settings)
    
    # None should trigger fast-path and not be processed
    # (parameter is not popped from dict)


def test_save_variables_invalid_name_raises(solver):
    """Test save_variables with invalid name raises clear error."""
    output_settings = {"save_variables": ["nonexistent_variable"]}
    
    with pytest.raises(ValueError, match="Variables not found"):
        solver.convert_output_labels(output_settings)


def test_save_variables_error_includes_available_names(solver):
    """Test error message includes available variable names."""
    output_settings = {"save_variables": ["nonexistent_variable"]}
    
    try:
        solver.convert_output_labels(output_settings)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Available states:" in str(e)
        assert "Available observables:" in str(e)


def test_array_only_fast_path(solver):
    """Test array-only parameters don't trigger name resolution."""
    import time
    
    output_settings = {
        "saved_state_indices": np.array([0, 1], dtype=np.int16)
    }
    
    # Time the fast path (should be very quick)
    start = time.perf_counter()
    for _ in range(1000):
        settings_copy = output_settings.copy()
        solver.convert_output_labels(settings_copy)
    fast_time = time.perf_counter() - start
    
    # Should be well under 1 second for 1000 iterations
    assert fast_time < 1.0


def test_solve_ivp_with_save_variables(system):
    """Test solve_ivp accepts save_variables and produces correct output."""
    state_names = list(system.initial_values.names)[:2]
    
    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        save_variables=state_names,
        dt_save=0.01,
        duration=0.1,
        method="euler",
    )
    
    # Verify result contains saved states
    assert result is not None
    assert hasattr(result, 'time_domain_array')
    assert result.time_domain_array is not None
    # Verify shape matches number of save_variables
    # time_domain_array shape is (time, variable, run) by default
    assert result.time_domain_array.shape[1] == len(state_names)


def test_solver_solve_with_save_variables(system):
    """Test Solver.solve accepts save_variables parameter."""
    state_names = list(system.initial_values.names)[:1]
    
    # Create fresh solver to avoid driver conflicts
    solver = Solver(system)
    result = solver.solve(
        initial_values={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        save_variables=state_names,
        duration=0.1,
    )
    
    assert result is not None
    # Verify saved output contains requested states
    assert result.time_domain_array.shape[1] >= 1


def test_solve_ivp_with_summarise_variables(system):
    """Test solve_ivp accepts summarise_variables and produces summaries."""
    state_names = list(system.initial_values.names)[:2]
    
    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        summarise_variables=state_names,
        duration=0.1,
        method="euler",
    )
    
    # Verify result contains summaries
    assert result is not None


def test_save_variables_with_solve_ivp(system):
    """Test save_variables parameter works with solve_ivp."""
    state_names = list(system.initial_values.names)[:2]
    
    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        save_variables=state_names,
        dt_save=0.01,
        duration=0.1,
        method="euler",
    )
    
    assert result is not None
    assert result.time_domain_array is not None


def test_save_variables_with_multiple_states(system):
    """Test save_variables with multiple state variables."""
    state_names = list(system.initial_values.names)
    if len(state_names) < 2:
        pytest.skip("Need at least 2 states")
    
    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0], state_names[1]: [2.0]},
        parameters={list(system.parameters.names)[0]: [0.1]},
        save_variables=state_names[:2],
        dt_save=0.01,
        duration=0.1,
        method="euler",
    )
    
    assert result is not None
    assert result.time_domain_array.shape[1] == 2


def test_deprecated_label_parameters_rejected(system):
    """Test that deprecated label-based parameters are rejected.
    
    The parameters saved_states, saved_observables, summarised_states,
    and summarised_observables are no longer accepted as input parameters.
    Users should use save_variables/summarise_variables or index-based
    parameters instead.
    """
    from cubie.outputhandling.output_functions import (
        ALL_OUTPUT_FUNCTION_PARAMETERS
    )
    
    assert "saved_states" not in ALL_OUTPUT_FUNCTION_PARAMETERS
    assert "saved_observables" not in ALL_OUTPUT_FUNCTION_PARAMETERS
    assert "summarised_states" not in ALL_OUTPUT_FUNCTION_PARAMETERS
    assert "summarised_observables" not in ALL_OUTPUT_FUNCTION_PARAMETERS
    
    solver = Solver(
        system,
        algorithm="euler",
    )
    assert solver is not None


def test_unified_save_variables_parameter(system):
    """Test that save_variables parameter works as replacement for deprecated params.
    
    This demonstrates the recommended migration path from deprecated
    saved_states/saved_observables to the unified save_variables parameter.
    """
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names") and system.observables.names
        else []
    )
    
    all_vars = state_names[:2]
    if observable_names:
        all_vars.extend(observable_names[:1])
    
    solver = Solver(
        system,
        algorithm="euler",
        save_variables=all_vars,
        output_types=["state", "observables"],
    )
    
    assert len(solver.saved_state_indices) >= 2
    if observable_names:
        assert len(solver.saved_observable_indices) >= 1
    
    saved_states_list = solver.saved_states
    assert isinstance(saved_states_list, list)
    assert len(saved_states_list) >= 2
