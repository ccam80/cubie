import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from cubie.batchsolving.BatchGridBuilder import (
    BatchGridBuilder,
    combinatorial_grid,
    combine_grids,
    extend_grid_to_array,
    generate_array,
    generate_grid,
    unique_cartesian_product,
    verbatim_grid,
)
import itertools
# from cubie.odesystems.systems.decays import Decays


@pytest.fixture(scope="session")
def grid_builder(system):
    return BatchGridBuilder.from_system(system)


@pytest.fixture(scope="session")
def example_arrays():
    """Fixture providing three example arrays for states, parameters, observables."""
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    return a, b


@pytest.fixture(scope="session")
def example_grids(example_arrays):
    """Fixture providing example grids for combinatorial and verbatim.
    
    Grids are in (variable, run) format: rows are variables, columns are runs.
    """
    a, b = example_arrays
    # Build in (variable, run) format - 2 variables, 9 runs for combinatorial
    combinatorial_grid = np.zeros(shape=(2, a.shape[0] * b.shape[0]))
    for i, val_a in enumerate(a):
        for j, val_b in enumerate(b):
            run_idx = i * b.shape[0] + j
            combinatorial_grid[0, run_idx] = val_a
            combinatorial_grid[1, run_idx] = val_b
    verbatim_grid = np.vstack((a, b))  # (2, 3) format
    return combinatorial_grid, verbatim_grid


@pytest.fixture(scope="session")
def batch_settings(request):
    """Fixture providing default batch settings."""
    defaults = {
        "num_state_vals": 10,
        "num_param_vals": 10,
        "kind": "combinatorial",
    }
    if hasattr(request, "param"):
        for key in request.param:
            if key in defaults:
                # Update only if the key exists in defaults
                defaults[key] = request.param[key]

    return defaults


@pytest.fixture(scope="session")
def batch_request(system, batch_settings):
    """Fixture providing a requested_batch dict using state and parameter names from the system."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.arange(batch_settings["num_state_vals"]),
        param_names[1]: np.arange(batch_settings["num_param_vals"]),
    }


def test_combinatorial_and_verbatim_grid(system):
    param = system.parameters
    request = {
        param.names[0]: np.array([0, 1]),
        param.names[1]: np.array([10, 20]),
    }
    idx, grid = combinatorial_grid(request, param)
    assert_array_equal(idx, np.array([0, 1]))
    # Grid is (variable, run) format: 2 swept vars, 4 runs
    assert grid.shape == (2, 4)

    idx_v, grid_v = verbatim_grid(request, param)
    assert_array_equal(idx_v, np.array([0, 1]))
    # Verbatim grid: 2 swept vars, 2 runs (paired row-wise)
    assert_array_equal(grid_v, np.array([[0, 1], [10, 20]]))


def test_call_with_request(grid_builder, system):
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    params = {param_names[0]: [10, 20]}
    states = {state_names[0]: [0, 1]}
    inits, params = grid_builder(params=params, states=states, kind="combinatorial")
    # Arrays are in (variable, run) format - runs in shape[1]
    assert inits.shape[1] == params.shape[1] == 4


def test_unique_cartesian_product(example_arrays, example_grids):
    a, b = example_arrays
    cartesian_product = unique_cartesian_product([a, b])
    combinatorial_grid, _ = example_grids
    assert_array_equal(cartesian_product, combinatorial_grid)


def test_combinatorial_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals - 1) + numvals - 1
    request = {param.names[0]: a, param.names[1]: b}
    indices, grid = combinatorial_grid(request, param)
    # Build expected grid in (variable, run) format
    n_runs = numvals * (numvals - 1)
    test_result = np.zeros(shape=(2, n_runs))

    for i, val_a in enumerate(a):
        for j, val_b in enumerate(b):
            run_idx = i * b.shape[0] + j
            test_result[0, run_idx] = val_a
            test_result[1, run_idx] = val_b

    assert_array_equal(indices, np.array([0, 1]))
    # Grid is (variable, run) format: rows are swept variables, columns are runs
    assert grid.shape[0] == 2  # 2 variables swept
    assert grid.shape[1] == n_runs
    assert_array_equal(grid, test_result)


def test_verbatim_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {param.names[0]: a, param.names[1]: b}
    # Build expected grid in (variable, run) format
    test_result = np.zeros(shape=(2, numvals))
    test_result[0, :] = a
    test_result[1, :] = b

    indices, grid = verbatim_grid(request, param)
    # Grid is (variable, run) format
    assert grid.shape[0] == 2  # 2 variables swept
    assert grid.shape[1] == numvals  # numvals runs
    assert_array_equal(indices, np.array([0, 1]))
    assert_array_equal(grid, test_result)


def test_generate_grid(system, batch_settings):
    state = system.initial_values
    numvals = batch_settings["num_state_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {state.names[0]: a, state.names[1]: b}
    # Grid is (variable, run) format
    indices, grid = generate_grid(request, state, kind="combinatorial")
    assert grid.shape[0] == 2  # 2 variables swept
    assert grid.shape[1] == numvals * numvals  # n*n runs for combinatorial
    indices, grid = generate_grid(request, state, kind="verbatim")
    assert grid.shape[0] == 2  # 2 variables swept
    assert grid.shape[1] == numvals  # numvals runs for verbatim
    with pytest.raises(ValueError):
        generate_grid(request, state, kind="badkind")


def test_grid_size_errors(system):
    """Test that ValueError is raised for invalid grid sizes."""
    param = system.parameters
    with pytest.raises(ValueError):
        verbatim_grid(
            {param.names[0]: [1, 2, 3], param.names[1]: [1, 2]}, param
        )


def test_combine_grids(example_arrays):
    # Combinatorial test: create two grids in (variable, run) format
    # grid1: 2 variables, 2 runs
    grid1 = np.array([[1, 2], [3, 4]])  # var0=[1,2], var1=[3,4]
    # grid2: 2 variables, 3 runs
    grid2 = np.array([[10, 20, 30], [40, 50, 60]])  # var0=[10,20,30], var1=[40,50,60]

    # Expected combinatorial: 2 runs * 3 runs = 6 runs total
    expected_grid1 = np.array([[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]])
    expected_grid2 = np.array(
        [[10, 20, 30, 10, 20, 30], [40, 50, 60, 40, 50, 60]]
    )
    result_grid1, result_grid2 = combine_grids(
        grid1, grid2, kind="combinatorial"
    )
    assert np.array_equal(result_grid1, expected_grid1)
    assert np.array_equal(result_grid2, expected_grid2)

    # Verbatim test: grids with matching number of runs should be returned as is
    grid1_v = np.array([[1, 2], [3, 4]])  # 2 vars, 2 runs
    grid2_v = np.array([[10, 20], [30, 40]])  # 2 vars, 2 runs
    result_grid1_v, result_grid2_v = combine_grids(
        grid1_v, grid2_v, kind="verbatim"
    )
    assert np.array_equal(result_grid1_v, grid1_v)
    assert np.array_equal(result_grid2_v, grid2_v)


def test_extend_grid_to_array(system):
    param = system.parameters
    indices = np.array([0, 1])
    # Grid in (variable, run) format: 2 swept variables, 3 runs
    grid = np.array([[1, 2, 3], [4, 5, 6]])
    arr = extend_grid_to_array(grid, indices, param.values_array)
    # Result is (variable, run) format
    assert arr.shape[0] == param.values_array.shape[0]  # all variables
    assert arr.shape[1] == 3  # 3 runs
    assert_array_equal(arr[indices, :], grid)
    assert np.all(arr[2, :] == param.values_array[2])


def test_generate_array(system, batch_settings, batch_request):
    # Test the generate_array utility function
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    request = {
        param.names[0]: a,
    }
    arr = generate_array(request, param)
    # Result is (variable, run) format
    assert arr.ndim == 2
    assert arr.shape[0] == param.values_array.shape[0]  # all variables
    assert arr.shape[1] == numvals  # numvals runs
    assert_array_equal(arr[0, :], a)  # swept variable values
    assert np.all(
        arr[1:, :] == param.values_array[1:, np.newaxis]
    )  # Check other parameters are unchanged


@pytest.fixture(scope="session")
def param_dict(system, batch_settings):
    param_names = list(system.parameters.names)
    return {
        param_names[1]: np.arange(batch_settings["num_param_vals"]),
    }


@pytest.fixture(scope="session")
def state_dict(system, batch_settings):
    state_names = list(system.initial_values.names)
    return {
        state_names[0]: np.arange(batch_settings["num_state_vals"]),
    }


@pytest.fixture(scope="session")
def param_seq(system, batch_settings):
    return np.arange(system.parameters.n - 1).tolist()


@pytest.fixture(scope="session")
def state_seq(system, batch_settings):
    return tuple(np.arange(system.initial_values.n - 1))


@pytest.fixture(scope="session")
def state_array(system, batch_settings):
    """Return a partial state array to test default-filling behavior.

    Creates an array with fewer variables than n_states so that the
    _trim_or_extend logic fills missing variables with system defaults.
    The test_call_input_types test verifies the last variable equals
    the default value.
    """
    n_runs = batch_settings["num_state_vals"]
    n_vars = system.sizes.states - 1  # Partial: last variable uses defaults
    # Build (variable, run) format: rows are variables, columns are runs
    rows = [np.linspace(0.1 * (i + 1), 0.5 * (i + 1), n_runs) for i in range(n_vars)]
    return np.vstack(rows)


@pytest.fixture(scope="session")
def param_array(system, batch_settings):
    """Return a partial param array to test default-filling behavior.

    Creates an array with fewer variables than n_params so that the
    _trim_or_extend logic fills missing variables with system defaults.
    The test_call_input_types test verifies the last variable equals
    the default value.
    """
    n_runs = batch_settings["num_param_vals"]
    n_vars = system.sizes.parameters - 1  # Partial: last variable uses defaults
    # Build (variable, run) format: rows are variables, columns are runs
    rows = [np.linspace(10.0 * (i + 1), 50.0 * (i + 1), n_runs) for i in range(n_vars)]
    return np.vstack(rows)


@pytest.mark.parametrize(
    "params_type,states_type",
    list(itertools.product(["dict", "seq", "array", None], repeat=2)),
)
def test_call_input_types(
    grid_builder,
    system,
    params_type,
    states_type,
    param_dict,
    param_seq,
    param_array,
    state_dict,
    state_seq,
    state_array,
):
    # Prepare input for each type
    params = None
    states = None
    if params_type == "dict":
        params = param_dict
    elif params_type == "seq":
        params = param_seq
    elif params_type == "array":
        params = param_array

    if states_type == "dict":
        states = state_dict
    elif states_type == "seq":
        states = state_seq
    elif states_type == "array":
        states = state_array

    # All combinations of params and states are valid
    initial, param = grid_builder(params=params, states=states)

    sizes = system.sizes

    # Arrays are in (variable, run) format
    assert initial.shape[1] == param.shape[1]  # Same number of runs
    assert initial.shape[0] == sizes.states  # Variables in shape[0]
    assert param.shape[0] == sizes.parameters
    assert_array_equal(
        param[-1, :],
        np.full_like(param[-1, :], system.parameters.values_array[-1]),
    )
    assert_array_equal(
        initial[-1, :],
        np.full_like(initial[-1, :], system.initial_values.values_array[-1]),
    )

def test_call_outputs(system, grid_builder):
    # Input arrays are in (variable, run) format
    testarray1 = np.array([[1, 2], [3, 4]])  # 2 vars, 2 runs
    state_testarray1 = extend_test_array(testarray1, system.initial_values)
    param_testarray1 = extend_test_array(testarray1, system.parameters)
    teststatedict = {"x0": [1, 3], "x1": [2, 4]}
    testparamdict = {"p0": [1, 3], "p1": [2, 4]}

    # For combinatorial: each column of grid1 paired with each column of grid2
    # grid1 (states): [[1,2],[3,4]] -> 2 runs
    # grid2 (params): [[1,2],[3,4]] -> 2 runs
    # Result: 4 runs total
    # States: repeat each column for each param column: [[1,1,2,2],[3,3,4,4]]
    # Params: tile columns: [[1,2,1,2],[3,4,3,4]]
    arraycomb1 = extend_test_array(
        np.asarray([[1, 1, 2, 2], [3, 3, 4, 4]]), system.initial_values
    )
    arraycomb2 = extend_test_array(
        np.asarray([[1, 2, 1, 2], [3, 4, 3, 4]]), system.parameters
    )

    # Combine input arrays
    inits, params = grid_builder(
        params=testarray1, states=testarray1, kind="combinatorial"
    )
    assert_array_equal(inits, arraycomb1)
    assert_array_equal(params, arraycomb2)
    inits, params = grid_builder(
        params=testarray1, states=testarray1, kind="verbatim"
    )

    assert_array_equal(inits, state_testarray1)
    assert_array_equal(params, param_testarray1)

    # full combo from dicts - produces 16 runs (2*2 state combos * 2*2 param combos)
    # unique_cartesian_product([1,3], [2,4]) gives [[1,1,3,3],[2,4,2,4]]
    fullcombosingle_state = np.asarray([[1, 1, 3, 3], [2, 4, 2, 4]])
    fullcombosingle_param = np.asarray([[1, 1, 3, 3], [2, 4, 2, 4]])
    # combine_grids repeats states for each param, tiles params
    statefullcombdouble = extend_test_array(
        np.repeat(fullcombosingle_state, 4, axis=1), system.initial_values
    )
    paramfullcombdouble = extend_test_array(
        np.tile(fullcombosingle_param, (1, 4)), system.parameters
    )

    inits, params = grid_builder(
        params=testparamdict, states=teststatedict, kind="combinatorial"
    )
    assert_array_equal(inits, statefullcombdouble)
    assert_array_equal(params, paramfullcombdouble)

    # Verbatim from dicts - pairs row-wise: 2 runs
    verbatim_state = extend_test_array(
        np.asarray([[1, 3], [2, 4]]), system.initial_values
    )
    verbatim_param = extend_test_array(
        np.asarray([[1, 3], [2, 4]]), system.parameters
    )
    inits, params = grid_builder(
        params=testparamdict, states=teststatedict, kind="verbatim"
    )
    assert_array_equal(inits, verbatim_state)
    assert_array_equal(params, verbatim_param)


def extend_test_array(array, values_object):
    """Extend a 2D array to match the system's variable count.
    
    Input array is in (variable, run) format. Output is padded in variable
    dimension to match values_object.n variables.
    """
    n_vars = array.shape[0]
    n_runs = array.shape[1]
    if n_vars >= values_object.n:
        return array[:values_object.n, :]
    # Pad with default values for missing variables
    padding = np.column_stack(
        [values_object.values_array[n_vars:]] * n_runs
    )
    return np.vstack([array, padding])


def test_docstring_examples(grid_builder, system, tolerance):
    # Example 1: combinatorial dict

    grid_builder = BatchGridBuilder.from_system(system)
    params = {"p0": [0.1, 0.2], "p1": [10, 20]}
    states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="combinatorial"
    )
    # Expected arrays in (variable, run) format - transposed from original
    expected_initial_large = np.array(
        [
            [1.0, 0.5, 1.2],
            [1.0, 0.5, 1.2],
            [1.0, 0.5, 1.2],
            [1.0, 0.5, 1.2],
            [1.0, 1.5, 1.2],
            [1.0, 1.5, 1.2],
            [1.0, 1.5, 1.2],
            [1.0, 1.5, 1.2],
            [2.0, 0.5, 1.2],
            [2.0, 0.5, 1.2],
            [2.0, 0.5, 1.2],
            [2.0, 0.5, 1.2],
            [2.0, 1.5, 1.2],
            [2.0, 1.5, 1.2],
            [2.0, 1.5, 1.2],
            [2.0, 1.5, 1.2],
        ]
    ).T
    expected_params_large = np.array(
        [
            [0.1, 10.0, 1.1],
            [0.1, 20.0, 1.1],
            [0.2, 10.0, 1.1],
            [0.2, 20.0, 1.1],
            [0.1, 10.0, 1.1],
            [0.1, 20.0, 1.1],
            [0.2, 10.0, 1.1],
            [0.2, 20.0, 1.1],
            [0.1, 10.0, 1.1],
            [0.1, 20.0, 1.1],
            [0.2, 10.0, 1.1],
            [0.2, 20.0, 1.1],
            [0.1, 10.0, 1.1],
            [0.1, 20.0, 1.1],
            [0.2, 10.0, 1.1],
            [0.2, 20.0, 1.1],
        ]
    ).T
    assert_allclose(
        initial_states,
        expected_initial_large,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert_allclose(
        parameters,
        expected_params_large,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

    # Example 2: verbatim arrays in (variable, run) format
    params = np.array([[0.1, 0.2], [10, 20]])  # 2 vars, 2 runs
    states = np.array([[1.0, 2.0], [0.5, 1.5]])  # 2 vars, 2 runs
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="verbatim"
    )
    # Expected arrays in (variable, run) format - extended with defaults
    expected_initial = np.array([[1.0, 2.0], [0.5, 1.5], [1.2, 1.2]])
    expected_params = np.array([[0.1, 0.2], [10.0, 20.0], [1.1, 1.1]])
    assert_allclose(
        initial_states,
        expected_initial,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert_allclose(
        parameters,
        expected_params,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

    # Example 3: combinatorial arrays (2 runs * 2 runs = 4 total runs)
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="combinatorial"
    )
    # Expected arrays in (variable, run) format
    # States: each column repeats for each param run
    # Params: columns tile for each state run
    expected_initial = np.array(
        [[1.0, 1.0, 2.0, 2.0], [0.5, 0.5, 1.5, 1.5], [1.2, 1.2, 1.2, 1.2]]
    )
    expected_params = np.array(
        [[0.1, 0.2, 0.1, 0.2], [10.0, 20.0, 10.0, 20.0], [1.1, 1.1, 1.1, 1.1]]
    )
    assert_allclose(
        initial_states,
        expected_initial,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert_allclose(
        parameters,
        expected_params,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

    # Example 4: single param sweep (params dict only)
    params_dict = {"p0": [0.1, 0.2]}
    initial_states, parameters = grid_builder(
        params=params_dict, kind="combinatorial"
    )
    # Expected arrays in (variable, run) format
    expected_params = np.array([[0.1, 0.90, 1.1], [0.2, 0.9, 1.1]]).T
    expected_initial = np.array([[0.5, -0.25, 1.2], [0.5, -0.25, 1.2]]).T
    assert_allclose(
        initial_states,
        expected_initial,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert_allclose(
        parameters,
        expected_params,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


@pytest.mark.parametrize(
        "solver_settings_override",
        [{"precision": np.float32},
         {"precision": np.float64}],
        indirect=True
)
def test_grid_builder_precision_enforcement(system, precision):
    """Test that BatchGridBuilder enforces system precision on output arrays.
    
    When users provide values in Python's default float64 precision,
    the returned arrays should match the system's configured precision.
    """
    grid_builder = BatchGridBuilder.from_system(system)
    
    # Test with array inputs (Python defaults to float64)
    inits, params = grid_builder(states=[1.0], params=[1.0])
    assert inits.dtype == precision
    assert params.dtype == precision
    
    # Test with dict inputs
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = grid_builder(
        states=None,
        params={param_names[0]: [3.0, 4.0]},
        kind="combinatorial"
    )
    assert inits.dtype == precision
    assert params.dtype == precision
    
    # Test with separate dict inputs
    inits, params = grid_builder(
        states={state_names[0]: [1.0, 2.0]},
        params={param_names[0]: [2.0]},
        kind="verbatim"
    )
    assert inits.dtype == precision
    assert params.dtype == precision
    
    # Test with mixed array and dict inputs (verbatim with matching lengths)
    # Array is (variable, run) format: 2 vars, 2 runs
    inits, params = grid_builder(
        states=[[1.0, 2.0], [4.0, 5.0]],
        params={param_names[0]: [7.0, 8.0]},
        kind="verbatim"
    )
    assert inits.dtype == precision
    assert params.dtype == precision

    system.update({'precision': np.float32})

    # Test with mixed array and dict inputs (verbatim with matching lengths)
    # Array is (variable, run) format: 2 vars, 2 runs
    inits, params = grid_builder(
        states=[[1.0, 2.0], [4.0, 5.0]],
        params={param_names[0]: [7.0, 8.0]},
        kind="verbatim"
    )
    assert inits.dtype == np.float32
    assert params.dtype == np.float32


def test_single_param_dict_sweep(grid_builder, system):
    """Test single parameter sweep with dict produces correct runs.

    User story US-1: params={'p1': np.linspace(0,1,100)} should
    produce 100 runs with p1 varied and all else at defaults.
    """
    param_names = list(system.parameters.names)
    sweep_values = np.linspace(0, 1, 100)

    inits, params = grid_builder(
        params={param_names[0]: sweep_values},
        kind="combinatorial"
    )

    # Should produce 100 runs
    assert inits.shape[1] == 100
    assert params.shape[1] == 100

    # Swept parameter should match input values
    assert_allclose(params[0, :], sweep_values, rtol=1e-7)

    # Other parameters should be at defaults
    for i in range(1, system.sizes.parameters):
        assert_allclose(
            params[i, :],
            np.full(100, system.parameters.values_array[i]),
            rtol=1e-7
        )

    # States should all be at defaults
    for i in range(system.sizes.states):
        assert_allclose(
            inits[i, :],
            np.full(100, system.initial_values.values_array[i]),
            rtol=1e-7
        )


def test_single_state_dict_single_run(grid_builder, system):
    """Test single state scalar override produces one run.

    User story: states={'x': 0.5} should produce 1 run with x=0.5.
    """
    state_names = list(system.initial_values.names)

    inits, params = grid_builder(
        states={state_names[0]: 0.5},
        kind="combinatorial"
    )

    # Should produce 1 run
    assert inits.shape[1] == 1
    assert params.shape[1] == 1

    # Overridden state should have new value
    assert_allclose(inits[0, 0], 0.5, rtol=1e-7)

    # Other states should be at defaults
    for i in range(1, system.sizes.states):
        assert_allclose(
            inits[i, 0],
            system.initial_values.values_array[i],
            rtol=1e-7
        )

    # All parameters at defaults
    for i in range(system.sizes.parameters):
        assert_allclose(
            params[i, 0],
            system.parameters.values_array[i],
            rtol=1e-7
        )


def test_states_dict_params_sweep(grid_builder, system):
    """Test state override with parameter sweep.

    User story US-2: states={'x': 0.2}, params={'p1': linspace(0,3,300)}
    should produce 300 runs with x=0.2 for all, p1 varied.
    """
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    sweep_values = np.linspace(0, 3, 300)

    inits, params = grid_builder(
        states={state_names[0]: 0.2},
        params={param_names[0]: sweep_values},
        kind="combinatorial"
    )

    # Should produce 300 runs
    assert inits.shape[1] == 300
    assert params.shape[1] == 300

    # Overridden state should be 0.2 for all runs
    assert_allclose(inits[0, :], np.full(300, 0.2), rtol=1e-7)

    # Swept parameter should match input values
    assert_allclose(params[0, :], sweep_values, rtol=1e-7)


def test_combinatorial_states_params(grid_builder, system):
    """Test combinatorial expansion of states and params.

    User story US-3: states={'y': [0.1, 0.2]}, params={'p1': linspace(0,1,100)}
    with kind='combinatorial' should produce 200 runs.
    """
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    state_values = [0.1, 0.2]
    param_values = np.linspace(0, 1, 100)

    inits, params = grid_builder(
        states={state_names[0]: state_values},
        params={param_names[0]: param_values},
        kind="combinatorial"
    )

    # Should produce 2 * 100 = 200 runs
    assert inits.shape[1] == 200
    assert params.shape[1] == 200


def test_1d_param_array_single_run(grid_builder, system):
    """Test 1D parameter array is treated as single run.

    User story US-5: 1D array of length n_params treated as single run.
    """
    n_params = system.sizes.parameters
    param_values = np.arange(n_params, dtype=float)

    inits, params = grid_builder(
        params=param_values,
        kind="combinatorial"
    )

    # Should produce 1 run
    assert inits.shape[1] == 1
    assert params.shape[1] == 1

    # Parameter values should match input
    assert_allclose(params[:, 0], param_values, rtol=1e-7)


def test_1d_state_array_partial_warning(grid_builder, system):
    """Test 1D partial state array triggers warning and fills defaults.

    User story US-5: Partial arrays should warn and fill missing values.
    """
    # Create array shorter than n_states
    partial_values = np.array([1.0, 2.0])

    with pytest.warns(UserWarning, match="Missing values"):
        inits, params = grid_builder(
            states=partial_values,
            kind="combinatorial"
        )

    # Should produce 1 run
    assert inits.shape[1] == 1

    # First two states should match input
    assert_allclose(inits[0, 0], 1.0, rtol=1e-7)
    assert_allclose(inits[1, 0], 2.0, rtol=1e-7)

    # Remaining states should be defaults
    for i in range(2, system.sizes.states):
        assert_allclose(
            inits[i, 0],
            system.initial_values.values_array[i],
            rtol=1e-7
        )


def test_empty_inputs_returns_defaults(grid_builder, system):
    """Test empty inputs return single run with all defaults.

    User story: Empty/None inputs should return defaults.
    """
    inits, params = grid_builder(
        params=None,
        states=None,
        kind="combinatorial"
    )

    # Should produce 1 run
    assert inits.shape[1] == 1
    assert params.shape[1] == 1

    # All values should be defaults
    assert_allclose(
        inits[:, 0],
        system.initial_values.values_array,
        rtol=1e-7
    )
    assert_allclose(
        params[:, 0],
        system.parameters.values_array,
        rtol=1e-7
    )


def test_verbatim_single_run_broadcast(grid_builder, system):
    """Test verbatim mode broadcasts single-run grids.

    User story: states={'x': 0.5}, params={'p1': [1,2,3]}, kind='verbatim'
    should produce 3 runs with x=0.5 broadcast to all.
    """
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)

    inits, params = grid_builder(
        states={state_names[0]: 0.5},
        params={param_names[0]: [1.0, 2.0, 3.0]},
        kind="verbatim"
    )

    # Should produce 3 runs
    assert inits.shape[1] == 3
    assert params.shape[1] == 3

    # State should be broadcast to all runs
    assert_allclose(inits[0, :], np.full(3, 0.5), rtol=1e-7)

    # Parameter should vary
    assert_allclose(params[0, :], [1.0, 2.0, 3.0], rtol=1e-7)


def test_call_combinatorial_1each(grid_builder, system):
    """Test combinatorial expansion with 1 state and 1 param swept."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    states = {state_names[0]: [0, 1]}
    params = {param_names[1]: np.arange(10)}
    inits, params_out = grid_builder(
        params=params, states=states, kind="combinatorial"
    )
    assert inits.shape == (system.sizes.states, 20)
    assert params_out.shape == (system.sizes.parameters, 20)


def test_call_verbatim_mismatch_raises(grid_builder, system):
    """Test verbatim with mismatched lengths raises ValueError."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    states = {state_names[0]: [0, 1, 2]}
    params = {param_names[0]: [0, 1]}
    with pytest.raises(ValueError):
        grid_builder(params=params, states=states, kind="verbatim")


def test_call_empty_dict_values(grid_builder, system):
    """Test that empty dict values are filtered correctly."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    states = {state_names[0]: [0, 1, 2], state_names[1]: []}
    params = {param_names[0]: np.arange(3), param_names[1]: np.array([])}
    inits, params_out = grid_builder(
        params=params, states=states, kind="verbatim"
    )
    assert inits.shape[1] == 3
    assert params_out.shape[1] == 3


# ---------------------------------------------------------------------------
# Tests for _process_input and _align_run_counts helper methods
# ---------------------------------------------------------------------------


def test_process_input_none(grid_builder, system):
    """Verify _process_input with None returns single-column defaults."""
    result = grid_builder._process_input(
        None, system.initial_values, kind="combinatorial"
    )

    # Should return single-column array with defaults
    assert result.shape == (system.sizes.states, 1)
    assert_allclose(result[:, 0], system.initial_values.values_array, rtol=1e-7)


def test_process_input_dict_combinatorial(grid_builder, system):
    """Verify _process_input with dict expands correctly for combinatorial."""
    state_names = list(system.initial_values.names)
    input_dict = {state_names[0]: [1.0, 2.0], state_names[1]: [3.0, 4.0]}

    result = grid_builder._process_input(
        input_dict, system.initial_values, kind="combinatorial"
    )

    # Combinatorial: 2 * 2 = 4 runs
    assert result.shape == (system.sizes.states, 4)
    # All variables should be present
    assert result.shape[0] == system.sizes.states


def test_process_input_dict_verbatim(grid_builder, system):
    """Verify _process_input with dict expands correctly for verbatim."""
    state_names = list(system.initial_values.names)
    input_dict = {state_names[0]: [1.0, 2.0, 3.0], state_names[1]: [4.0, 5.0, 6.0]}

    result = grid_builder._process_input(
        input_dict, system.initial_values, kind="verbatim"
    )

    # Verbatim: 3 runs (row-wise pairing)
    assert result.shape == (system.sizes.states, 3)
    # Check swept values are in the array
    assert_allclose(result[0, :], [1.0, 2.0, 3.0], rtol=1e-7)
    assert_allclose(result[1, :], [4.0, 5.0, 6.0], rtol=1e-7)


def test_process_input_array(grid_builder, system):
    """Verify _process_input with array sanitizes correctly."""
    # Create a 2D array in (variable, run) format
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.warns(UserWarning, match="Missing values"):
        result = grid_builder._process_input(
            input_array, system.initial_values, kind="combinatorial"
        )

    # Should be extended to full variable count
    assert result.shape[0] == system.sizes.states
    assert result.shape[1] == 2  # 2 runs preserved


def test_process_input_invalid_type(grid_builder, system):
    """Verify _process_input raises TypeError for invalid input."""
    with pytest.raises(TypeError, match="Input must be None, dict, or array-like"):
        grid_builder._process_input(
            "invalid_string", system.initial_values, kind="combinatorial"
        )


def test_align_run_counts_combinatorial(grid_builder, system):
    """Verify _align_run_counts produces Cartesian product."""
    # Create two arrays with different run counts
    states_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 vars, 2 runs
    params_array = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0],
                             [70.0, 80.0, 90.0]])  # 3 vars, 3 runs

    aligned_states, aligned_params = grid_builder._align_run_counts(
        states_array, params_array, kind="combinatorial"
    )

    # Combinatorial: 2 * 3 = 6 runs
    assert aligned_states.shape[1] == 6
    assert aligned_params.shape[1] == 6


def test_align_run_counts_verbatim(grid_builder, system):
    """Verify _align_run_counts pairs directly with broadcast."""
    # Single-run states, multi-run params
    states_array = np.array([[1.0], [2.0], [3.0]])  # 3 vars, 1 run
    params_array = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0],
                             [70.0, 80.0, 90.0]])  # 3 vars, 3 runs

    aligned_states, aligned_params = grid_builder._align_run_counts(
        states_array, params_array, kind="verbatim"
    )

    # Verbatim with broadcast: single-run broadcasts to 3 runs
    assert aligned_states.shape[1] == 3
    assert aligned_params.shape[1] == 3
    # States should be broadcast
    assert_allclose(aligned_states[0, :], [1.0, 1.0, 1.0], rtol=1e-7)
