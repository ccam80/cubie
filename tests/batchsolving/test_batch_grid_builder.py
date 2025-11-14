import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
import cubie.batchsolving.BatchGridBuilder as batchgridmodule
import itertools
# from cubie.odesystems.systems.decays import Decays


@pytest.fixture(scope="function")
def grid_builder(system):
    return BatchGridBuilder.from_system(system)


@pytest.fixture(scope="function")
def example_arrays():
    """Fixture providing three example arrays for states, parameters, observables."""
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    return a, b


@pytest.fixture(scope="function")
def example_grids(example_arrays):
    """Fixture providing example grids for combinatorial and verbatim."""
    a, b = example_arrays
    combinatorial_grid = np.zeros(shape=(a.shape[0] * b.shape[0], 2))
    for i, val_a in enumerate(a):
        for j, val_b in enumerate(b):
            combinatorial_grid[i * b.shape[0] + j] = [val_a, val_b]
    verbatim_grid = np.hstack((a, b))
    return combinatorial_grid, verbatim_grid


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
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
    idx, grid = batchgridmodule.combinatorial_grid(request, param)
    assert_array_equal(idx, np.array([0, 1]))
    assert grid.shape == (4, 2)

    idx_v, grid_v = batchgridmodule.verbatim_grid(request, param)
    assert_array_equal(idx_v, np.array([0, 1]))
    assert_array_equal(grid_v, np.array([[0, 10], [1, 20]]))


def test_grid_arrays(grid_builder, system):
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    request = {state_names[0]: [0, 1], param_names[0]: [10, 20]}
    inits, params = grid_builder.grid_arrays(request, kind="combinatorial")
    assert inits.shape == (4, system.sizes.states)
    assert params.shape == (4, system.sizes.parameters)


def test_call_with_request(grid_builder, system):
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    request = {state_names[0]: [0, 1], param_names[0]: [10, 20]}
    inits, params = grid_builder(request=request, kind="combinatorial")
    assert inits.shape[0] == params.shape[0] == 4


def test_unique_cartesian_product(example_arrays, example_grids):
    a, b = example_arrays
    cartesian_product = batchgridmodule.unique_cartesian_product([a, b])
    combinatorial_grid, _ = example_grids
    assert_array_equal(cartesian_product, combinatorial_grid)


def test_combinatorial_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals - 1) + numvals - 1
    request = {param.names[0]: a, param.names[1]: b}
    indices, grid = batchgridmodule.combinatorial_grid(request, param)
    test_result = np.zeros(shape=(numvals * (numvals - 1), 2))

    for i, val_a in enumerate(a):
        for j, val_b in enumerate(b):
            test_result[i * b.shape[0] + j] = [val_a, val_b]

    assert_array_equal(indices, np.array([0, 1]))
    assert grid.shape[0] == numvals * (numvals - 1)
    assert grid.shape[1] == 2
    assert_array_equal(grid, test_result)


def test_verbatim_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {param.names[0]: a, param.names[1]: b}
    test_result = np.zeros(shape=(numvals, 2))
    test_result[:, 0] = a
    test_result[:, 1] = b

    indices, grid = batchgridmodule.verbatim_grid(request, param)
    assert grid.shape[0] == numvals
    assert grid.shape[1] == 2
    assert_array_equal(indices, np.array([0, 1]))
    assert_array_equal(grid, test_result)


def test_generate_grid(system, batch_settings):
    state = system.initial_values
    numvals = batch_settings["num_state_vals"]
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {state.names[0]: a, state.names[1]: b}
    indices, grid = batchgridmodule.generate_grid(request, state, kind="combinatorial")
    assert grid.shape[0] == numvals * numvals
    indices, grid = batchgridmodule.generate_grid(request, state, kind="verbatim")
    assert grid.shape[0] == numvals
    with pytest.raises(ValueError):
        batchgridmodule.generate_grid(request, state, kind="badkind")


def test_grid_size_errors(system):
    """Test that ValueError is raised for invalid grid sizes."""
    param = system.parameters
    with pytest.raises(ValueError):
        batchgridmodule.verbatim_grid(
            {param.names[0]: [1, 2, 3], param.names[1]: [1, 2]}, param
        )


def test_combine_grids(example_arrays):
    # Combinatorial test: create two grids with different numbers of rows
    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[10, 20], [30, 40], [50, 60]])

    expected_grid1 = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]])
    expected_grid2 = np.array(
        [[10, 20], [30, 40], [50, 60], [10, 20], [30, 40], [50, 60]]
    )
    result_grid1, result_grid2 = batchgridmodule.combine_grids(
        grid1, grid2, kind="combinatorial"
    )
    assert np.array_equal(result_grid1, expected_grid1)
    assert np.array_equal(result_grid2, expected_grid2)

    # Verbatim test: grids with matching number of rows should be returned as is
    grid1_v = np.array([[1, 2], [3, 4]])
    grid2_v = np.array([[10, 20], [30, 40]])
    result_grid1_v, result_grid2_v = batchgridmodule.combine_grids(
        grid1_v, grid2_v, kind="verbatim"
    )
    assert np.array_equal(result_grid1_v, grid1_v)
    assert np.array_equal(result_grid2_v, grid2_v)


def test_extend_grid_to_array(system):
    param = system.parameters
    indices = np.array([0, 1])
    grid = np.array([[1, 2], [3, 4], [5, 6]])
    arr = batchgridmodule.extend_grid_to_array(grid, indices, param.values_array)
    assert arr.shape[1] == param.values_array.shape[0]
    assert_array_equal(arr[:, indices], grid)
    assert np.all(arr[:, 2] == param.values_array[2])


def test_generate_array(system, batch_settings, batch_request):
    # Test the generate_array utility function
    param = system.parameters
    numvals = batch_settings["num_param_vals"]
    a = np.arange(numvals)
    request = {
        param.names[0]: a,
    }
    arr = batchgridmodule.generate_array(request, param)
    assert arr.ndim == 2
    assert arr.shape[1] == param.values_array.shape[0]
    assert arr.shape[0] == numvals
    assert_array_equal(arr[:, 0], a)
    assert np.all(
        arr[:, 1:] == param.values_array[1:]
    )  # Check other parameters are unchanged


@pytest.mark.parametrize(
    "batch_settings",
    [
        {"num_state_vals": 2, "num_param_vals": 3, "kind": "combinatorial"},
        {"num_state_vals": 3, "num_param_vals": 3, "kind": "verbatim"},
    ],
    indirect=True,
)
def test_grid_arrays_1each(grid_builder, batch_request, batch_settings):
    # Test grid_arrays with different grid types and batch sizes
    kind = batch_settings["kind"]
    initial_values_array, params_array = grid_builder.grid_arrays(
        batch_request, kind=kind
    )
    assert (
        initial_values_array.shape[1]
        == grid_builder.states.values_array.shape[0]
    )
    assert (
        params_array.shape[1] == grid_builder.parameters.values_array.shape[0]
    )
    if kind == "combinatorial":
        assert (
            initial_values_array.shape[0]
            == batch_settings["num_state_vals"]
            * batch_settings["num_param_vals"]
        )
    elif kind == "verbatim":
        assert (
            initial_values_array.shape[0] == batch_settings["num_state_vals"]
        )


@pytest.mark.parametrize(
    "batch_settings",
    [
        {"num_state_vals": 2, "num_param_vals": 3, "kind": "combinatorial"},
        {"num_state_vals": 3, "num_param_vals": 3, "kind": "verbatim"},
    ],
    indirect=True,
)
def test_grid_arrays_2and2(system, grid_builder, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings["num_state_vals"]
    numparams = batch_settings["num_param_vals"]
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {
        state.names[0]: a,
        state.names[1]: a,
        param.names[0]: b,
        param.names[1]: b,
    }
    kind = batch_settings["kind"]
    initial_values_array, params_array = grid_builder.grid_arrays(
        request, kind=kind
    )

    assert (
        initial_values_array.shape[1]
        == grid_builder.states.values_array.shape[0]
    )
    assert (
        params_array.shape[1] == grid_builder.parameters.values_array.shape[0]
    )

    if kind == "combinatorial":
        assert initial_values_array.shape[0] == numinits**2 * numparams**2
    elif kind == "verbatim":
        assert initial_values_array.shape[0] == numinits


@pytest.mark.parametrize(
    "batch_settings",
    [
        {"num_state_vals": 3, "num_param_vals": 2, "kind": "verbatim"},
    ],
    indirect=True,
)
def test_grid_arrays_verbatim_mismatch(system, grid_builder, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings["num_state_vals"]
    numparams = batch_settings["num_param_vals"]
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {
        state.names[0]: a,
        state.names[1]: a,
        param.names[0]: b,
        param.names[1]: b,
    }
    kind = batch_settings["kind"]
    with pytest.raises(ValueError):
        initial_values_array, params_array = grid_builder.grid_arrays(
            request, kind=kind
        )


@pytest.mark.parametrize(
    "batch_settings",
    [
        {"num_state_vals": 3, "num_param_vals": 2, "kind": "combinatorial"},
        {"num_state_vals": 3, "num_param_vals": 3, "kind": "verbatim"},
    ],
    indirect=True,
)
def test_grid_arrays_empty_inputs(system, grid_builder, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings["num_state_vals"]
    numparams = batch_settings["num_param_vals"]
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {
        state.names[0]: a,
        state.names[1]: [],
        param.names[0]: b,
        param.names[1]: np.asarray([]),
    }
    kind = batch_settings["kind"]

    initial_values_array, params_array = grid_builder.grid_arrays(
        request, kind=kind
    )

    assert (
        initial_values_array.shape[1]
        == grid_builder.states.values_array.shape[0]
    )
    assert (
        params_array.shape[1] == grid_builder.parameters.values_array.shape[0]
    )

    if kind == "combinatorial":
        assert initial_values_array.shape[0] == numinits * numparams
    elif kind == "verbatim":
        assert initial_values_array.shape[0] == numinits


@pytest.fixture(scope="function")
def param_dict(system, batch_settings):
    param_names = list(system.parameters.names)
    return {
        param_names[1]: np.arange(batch_settings["num_param_vals"]),
    }


@pytest.fixture(scope="function")
def state_dict(system, batch_settings):
    state_names = list(system.initial_values.names)
    return {
        state_names[0]: np.arange(batch_settings["num_state_vals"]),
    }


@pytest.fixture(scope="function")
def param_seq(system, batch_settings):
    return np.arange(system.parameters.n - 1).tolist()


@pytest.fixture(scope="function")
def state_seq(system, batch_settings):
    return tuple(np.arange(system.initial_values.n - 1))


@pytest.fixture(scope="function")
def state_array(system, batch_settings):
    numvals = batch_settings["num_state_vals"]
    return np.hstack(
        [
            np.linspace(0.1, 0.5, numvals).reshape(-1, 1),
            np.linspace(1, 5, numvals).reshape(-1, 1),
        ]
    )


@pytest.fixture(scope="function")
def param_array(system, batch_settings):
    numvals = batch_settings["num_param_vals"]
    return np.hstack(
        (
            np.linspace(10, 50, numvals).reshape(-1, 1),
            np.linspace(100, 500, numvals).reshape(-1, 1),
        )
    )


@pytest.mark.parametrize(
    "mixed_type, params_type,states_type",
    list(itertools.product(["dict", "seq", "array"], repeat=3)),
)
@pytest.mark.parametrize("system_override", ["linear"], indirect=True)
def test_call_input_types(
    grid_builder,
    system,
    mixed_type,
    params_type,
    states_type,
    param_dict,
    param_seq,
    param_array,
    state_dict,
    state_seq,
    state_array,
    batch_request,
):
    # Prepare input for each type
    params = None
    states = None
    request = None
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

    if mixed_type == "dict":
        request = batch_request
    elif mixed_type == "seq":
        request = np.hstack((param_seq, state_seq))
    elif mixed_type == "array":
        request = np.hstack((param_array, state_array))

    valid_combos = [
        ("dict", None, None),  # Only option with request
        (None, "dict", "dict"),
        (None, "dict", "seq"),
        (None, "dict", "array"),
        (None, "dict", None),
        (None, "seq", "dict"),
        (None, "seq", "seq"),
        (None, "seq", "array"),
        (None, "seq", None),
        (None, "array", "dict"),
        (None, "array", "seq"),
        (None, "array", "array"),
        (None, "array", None),
        (None, None, "dict"),
        (None, None, "seq"),
        (None, None, "array"),
        (None, None, None),
    ]

    if (mixed_type, states_type, params_type) not in valid_combos:
        with pytest.raises(TypeError):
            initial, param = grid_builder(
                params=params, states=states, request=request
            )
        return
    else:
        initial, param = grid_builder(
            request=request, params=params, states=states
        )

    sizes = system.sizes

    assert initial.shape[0] == param.shape[0]
    assert initial.shape[1] == sizes.states
    assert param.shape[1] == sizes.parameters
    assert_array_equal(
        param[:, -1],
        np.full_like(param[:, -1], system.parameters.values_array[-1]),
    )
    assert_array_equal(
        initial[:, -1],
        np.full_like(initial[:, -1], system.initial_values.values_array[-1]),
    )


@pytest.mark.parametrize("system_override", ["linear"], indirect=True)
def test_call_outputs(system, grid_builder):
    testarray1 = np.array([[1, 2], [3, 4]])
    state_testarray1 = extend_test_array(testarray1, system.initial_values)
    param_testarray1 = extend_test_array(testarray1, system.parameters)
    testlistarray = [[1, 2], [3, 4]]
    testseq = [1, 2]
    testdict = {
        "x0": [1, 3],
        "x1": [2, 4],
        "p0": [1, 3],
        "p1": np.asarray([2, 4]),
    }
    teststatedict = {"x0": [1, 3], "x1": [2, 4]}
    testparamdict = {"p0": [1, 3], "p1": [2, 4]}

    fullcombosingle = np.asarray([[1, 2], [1, 4], [3, 2], [3, 4]])
    fullcombdouble1 = np.vstack(
        (fullcombosingle, fullcombosingle, fullcombosingle, fullcombosingle)
    )
    fullcombdouble2 = np.repeat(fullcombosingle, 4, axis=0)
    statefullcombdouble = extend_test_array(
        fullcombdouble2, system.initial_values
    )
    paramfullcombdouble = extend_test_array(fullcombdouble1, system.parameters)

    arraycomb1 = extend_test_array(
        np.asarray([[1, 2], [1, 2], [3, 4], [3, 4]]), system.initial_values
    )
    arraycomb2 = extend_test_array(
        np.asarray([[1, 2], [3, 4], [1, 2], [3, 4]]), system.parameters
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

    # full combo from dict:
    inits, params = grid_builder(testdict, kind="combinatorial")
    assert_array_equal(params, paramfullcombdouble)
    assert_array_equal(inits, statefullcombdouble)

    inits, params = grid_builder(testdict, kind="verbatim")
    assert_array_equal(inits, state_testarray1)
    assert_array_equal(params, param_testarray1)

    inits, params = grid_builder(
        params=testparamdict, states=teststatedict, kind="combinatorial"
    )
    assert_array_equal(inits, statefullcombdouble)
    assert_array_equal(params, paramfullcombdouble)


def extend_test_array(array, values_object):
    """Extend a 2D array to match the system's initial values size."""
    return np.pad(
        array,
        ((0, 0), (0, values_object.n - array.shape[1])),
        mode="constant",
        constant_values=values_object.values_array[array.shape[1] :],
    )


def test_docstring_examples(grid_builder, system, tolerance):
    # Example 1: combinatorial dict

    grid_builder = BatchGridBuilder.from_system(system)
    params = {"p0": [0.1, 0.2], "p1": [10, 20]}
    states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="combinatorial"
    )
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
    )
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
    )
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

    # Example 2: verbatim arrays
    params = np.array([[0.1, 0.2], [10, 20]])
    states = np.array([[1.0, 2.0], [0.5, 1.5]])
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="verbatim"
    )
    expected_initial = np.array([[1.0, 2.0, 1.2], [0.5, 1.5, 1.2]])
    expected_params = np.array([[0.1, 0.2, 1.1], [10.0, 20.0, 1.1]])
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

    # Example 3: combinatorial arrays
    initial_states, parameters = grid_builder(
        params=params, states=states, kind="combinatorial"
    )
    expected_initial = np.array(
        [[1.0, 2.0, 1.2], [1.0, 2.0, 1.2], [0.5, 1.5, 1.2], [0.5, 1.5, 1.2]]
    )
    expected_params = np.array(
        [[0.1, 0.2, 1.1], [10.0, 20.0, 1.1], [0.1, 0.2, 1.1], [10.0, 20.0, 1.1]]
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

    # Example 4: request dict
    request = {
        "p0": [0.1, 0.2],
        "p1": [10, 20],
        "x0": [1.0, 2.0],
        "x1": [0.5, 1.5],
    }
    initial_states, parameters = grid_builder(
        request=request, kind="combinatorial"
    )
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
    initial_states, parameters = grid_builder(request=request, kind="verbatim")
    assert_allclose(
        initial_states,
        initial_states,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert_allclose(
        parameters,
        parameters,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

    # Example 5: request dict with only params
    request = {"p0": [0.1, 0.2]}
    initial_states, parameters = grid_builder(
        request=request, kind="combinatorial"
    )
    expected_params = np.array([[0.1, 0.90, 1.1], [0.2, 0.9, 1.1]])
    expected_initial = np.array([[0.5, -0.25, 1.2], [0.5, -0.25, 1.2]])
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
    "precision_override", [np.float32, np.float64], indirect=True
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
    request = {state_names[0]: [1.0, 2.0], param_names[0]: [3.0, 4.0]}
    inits, params = grid_builder(request=request, kind="combinatorial")
    assert inits.dtype == precision
    assert params.dtype == precision
    
    # Test with separate dict inputs
    inits, params = grid_builder(
        states={state_names[0]: [1.0]},
        params={param_names[0]: [2.0]},
        kind="verbatim"
    )
    assert inits.dtype == precision
    assert params.dtype == precision
    
    # Test with mixed array and dict inputs (verbatim with matching lengths)
    inits, params = grid_builder(
        states=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        params={param_names[0]: [7.0, 8.0]},
        kind="verbatim"
    )
    assert inits.dtype == precision
    assert params.dtype == precision

    system.update({'precision': np.float32})

    # Test with mixed array and dict inputs (verbatim with matching lengths)
    inits, params = grid_builder(
        states=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        params={param_names[0]: [7.0, 8.0]},
        kind="verbatim"
    )
    assert inits.dtype == np.float32
    assert params.dtype == np.float32
