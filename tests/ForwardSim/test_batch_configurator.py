import pytest
import numpy as np
from numpy.testing import assert_array_equal
from CuMC.ForwardSim.BatchConfigurator import BatchConfigurator
from CuMC.ForwardSim import BatchConfigurator as BC


@pytest.fixture(scope="function")
def batch_configurator(system):
    """Fixture for BatchConfigurator initialized with the Decays3 system."""
    return BatchConfigurator.from_system(system)

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
    defaults =  {
        'num_state_vals': 10,
        'num_param_vals': 10,
        'kind': 'combinatorial',
    }
    if hasattr(request, 'param'):
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
        state_names[0]: np.arange(batch_settings['num_state_vals']),
        param_names[1]: np.arange(batch_settings['num_param_vals']),
    }


# ************************** Utility Function Tests **************************

def test_unique_cartesian_product(example_arrays, example_grids):
    a, b = example_arrays
    cartesian_product = BC.unique_cartesian_product([a,b])
    combinatorial_grid, _ = example_grids
    assert_array_equal(cartesian_product, combinatorial_grid)

def test_combinatorial_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings['num_param_vals']
    a = np.arange(numvals)
    b = np.arange(numvals - 1) + numvals - 1
    request = {param.names[0]: a,
               param.names[1]: b}
    indices, grid = BC.combinatorial_grid(request, param)
    test_result = np.zeros(shape=(numvals * (numvals-1), 2))

    for i, val_a in enumerate(a):
        for j, val_b in enumerate(b):
            test_result[i * b.shape[0] + j] = [val_a, val_b]

    assert_array_equal(indices, np.array([0, 1]))
    assert grid.shape[0] == numvals * (numvals-1)
    assert grid.shape[1] == 2
    assert_array_equal(grid, test_result)


def test_verbatim_grid(system, batch_settings):
    param = system.parameters
    numvals = batch_settings['num_param_vals']
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {param.names[0]: a,
               param.names[1]: b
               }
    test_result = np.zeros(shape=(numvals, 2))
    test_result[:, 0] = a
    test_result[:, 1] = b

    indices, grid = BC.verbatim_grid(request, param)
    assert grid.shape[0] == numvals
    assert grid.shape[1] == 2
    assert_array_equal(indices, np.array([0, 1]))
    assert_array_equal(grid, test_result)

def test_generate_grid(system, batch_settings):
    state = system.initial_values
    numvals = batch_settings['num_state_vals']
    a = np.arange(numvals)
    b = np.arange(numvals) + numvals
    request = {state.names[0]: a,
               state.names[1]: b
               }
    indices, grid = BC.generate_grid(request, state, kind='combinatorial')
    assert grid.shape[0] == numvals*numvals
    indices, grid = BC.generate_grid(request, state, kind='verbatim')
    assert grid.shape[0] == numvals
    with pytest.raises(ValueError):
        BC.generate_grid(request, state, kind='badkind')

def test_grid_size_errors(system):
    """Test that ValueError is raised for invalid grid sizes."""
    param = system.parameters
    with pytest.raises(ValueError):
        BC.verbatim_grid({param.names[0]: [1, 2, 3],
                          param.names[1]: [1, 2]}, param)


def test_combine_grids(example_arrays):
    # Combinatorial test: create two grids with different numbers of rows
    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[10, 20], [30, 40], [50, 60]])

    expected_grid1 = np.array([[1, 2], [1, 2], [1, 2],
                               [3, 4], [3, 4], [3, 4]]
                              )
    expected_grid2 = np.array([[10, 20], [30, 40], [50, 60],
                               [10, 20], [30, 40], [50, 60]]
                              )
    result_grid1, result_grid2 = BC.combine_grids(grid1, grid2, kind='combinatorial')
    assert np.array_equal(result_grid1, expected_grid1)
    assert np.array_equal(result_grid2, expected_grid2)

    # Verbatim test: grids with matching number of rows should be returned as is
    grid1_v = np.array([[1, 2], [3, 4]])
    grid2_v = np.array([[10, 20], [30, 40]])
    result_grid1_v, result_grid2_v = BC.combine_grids(grid1_v, grid2_v, kind='verbatim')
    assert np.array_equal(result_grid1_v, grid1_v)
    assert np.array_equal(result_grid2_v, grid2_v)


def test_extend_grid_to_array(system):
    param = system.parameters
    indices = np.array([0, 1])
    grid = np.array([[1, 2], [3, 4], [5,6]])
    arr = BC.extend_grid_to_array(grid, indices, param.values_array)
    assert arr.shape[1] == param.values_array.shape[0]
    assert_array_equal(arr[:, indices], grid)
    assert np.all(arr[:,2] == param.values_array[2])


def test_generate_array(system, batch_settings, batch_request):
    # Test the generate_array utility function
    param = system.parameters
    numvals = batch_settings['num_param_vals']
    a = np.arange(numvals)
    request = {param.names[0]: a,
               }
    arr = BC.generate_array(request, param)
    assert arr.ndim == 2
    assert arr.shape[1] == param.values_array.shape[0]
    assert arr.shape[0] == numvals
    assert_array_equal(arr[:,0], a)
    assert np.all(arr[:,1:] == param.values_array[1:])  # Check other parameters are unchanged


# ************************** BatchConfigurator Tests **************************

def test_from_system(system):
    bc = BC.BatchConfigurator.from_system(system)
    assert isinstance(bc, BC.BatchConfigurator)
    assert hasattr(bc, 'parameters')
    assert hasattr(bc, 'states')
    assert hasattr(bc, 'observables')

@pytest.mark.parametrize("updates", [
    { 'x0': 42.0 },
    { 'p0': 3.14 },
    { 'x0': 1.23, 'p1': 4.56 },
])
def test_update(batch_configurator, updates):
    # Should not raise for valid keys
    batch_configurator.update(updates)
    # Should update values in parameters or states
    for k, v in updates.items():
        if k in batch_configurator.states:
            assert np.isclose(batch_configurator.states[k], v)
        elif k in batch_configurator.parameters:
            assert np.isclose(batch_configurator.parameters[k], v)

@pytest.mark.parametrize("updates", [
    { 'not_a_key': 1.0 },
    { 'x0': 1.0, 'bad_param': 2.0 },
])
def test_update_invalid(batch_configurator, updates):
    with pytest.raises(KeyError):
        batch_configurator.update(updates)

def test_state_indices(batch_configurator, system):
    # Test with names and indices
    idx_by_name = batch_configurator.state_indices(system.initial_values.names)
    idx_by_idx = batch_configurator.state_indices(list(range(system.sizes.states)))
    assert np.all(idx_by_name == np.arange(system.sizes.states))
    assert np.all(idx_by_idx == np.arange(system.sizes.states))

def test_observable_indices(batch_configurator, system):
    idx_by_name = batch_configurator.observable_indices(system.observables.names)
    idx_by_idx = batch_configurator.observable_indices(list(range(system.sizes.observables)))
    assert np.all(idx_by_name == np.arange(system.sizes.observables))
    assert np.all(idx_by_idx == np.arange(system.sizes.observables))

def test_parameter_indices(batch_configurator, system):
    idx_by_name = batch_configurator.parameter_indices(system.parameters.names)
    idx_by_idx = batch_configurator.parameter_indices(list(range(system.sizes.parameters)))
    assert np.all(idx_by_name == np.arange(system.sizes.parameters))
    assert np.all(idx_by_idx == np.arange(system.sizes.parameters))

@pytest.mark.parametrize("batch_settings", [
    {'num_state_vals': 2, 'num_param_vals': 3, 'kind': 'combinatorial'},
    {'num_state_vals': 3, 'num_param_vals': 3, 'kind': 'verbatim'},
], indirect=True)
def test_grid_arrays_1each(batch_configurator, batch_request, batch_settings):
    # Test grid_arrays with different grid types and batch sizes
    kind = batch_settings['kind']
    initial_values_array, params_array = batch_configurator.grid_arrays(batch_request, kind=kind)
    assert initial_values_array.shape[1] == batch_configurator.states.values_array.shape[0]
    assert params_array.shape[1] == batch_configurator.parameters.values_array.shape[0]
    if kind == 'combinatorial':
        assert initial_values_array.shape[0] == batch_settings['num_state_vals'] * batch_settings['num_param_vals']
    elif kind == 'verbatim':
        assert initial_values_array.shape[0] == batch_settings['num_state_vals']

@pytest.mark.parametrize("batch_settings", [
    {'num_state_vals': 2, 'num_param_vals': 3, 'kind': 'combinatorial'},
    {'num_state_vals': 3, 'num_param_vals': 3, 'kind': 'verbatim'},
], indirect=True)
def test_grid_arrays_2and2(system, batch_configurator, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings['num_state_vals']
    numparams = batch_settings['num_param_vals']
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {state.names[0]: a,
               state.names[1]: a,
               param.names[0]: b,
               param.names[1]: b,
               }
    kind = batch_settings['kind']
    initial_values_array, params_array = batch_configurator.grid_arrays(request, kind=kind)

    assert initial_values_array.shape[1] == batch_configurator.states.values_array.shape[0]
    assert params_array.shape[1] == batch_configurator.parameters.values_array.shape[0]

    if kind == 'combinatorial':
        assert initial_values_array.shape[0] == numinits **2 * numparams **2
    elif kind == 'verbatim':
        assert initial_values_array.shape[0] == numinits


@pytest.mark.parametrize("batch_settings", [
    {'num_state_vals': 3, 'num_param_vals': 2, 'kind': 'verbatim'},
    ], indirect=True
                         )
def test_grid_arrays_verbatim_mismatch(system, batch_configurator, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings['num_state_vals']
    numparams = batch_settings['num_param_vals']
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {state.names[0]: a,
               state.names[1]: a,
               param.names[0]: b,
               param.names[1]: b,
               }
    kind = batch_settings['kind']
    with pytest.raises(ValueError):
        initial_values_array, params_array = batch_configurator.grid_arrays(request, kind=kind)



@pytest.mark.parametrize("batch_settings", [
    {'num_state_vals': 3, 'num_param_vals': 2, 'kind': 'combinatorial'},
    {'num_state_vals': 3, 'num_param_vals': 3, 'kind': 'verbatim'},
    ], indirect=True
                         )
def test_grid_arrays_empty_inputs(system, batch_configurator, batch_settings):
    state = system.initial_values
    param = system.parameters
    numinits = batch_settings['num_state_vals']
    numparams = batch_settings['num_param_vals']
    a = np.arange(numinits)
    b = np.arange(numparams)
    request = {state.names[0]: a,
               state.names[1]: [],
               param.names[0]: b,
               param.names[1]: np.asarray([]),
               }
    kind = batch_settings['kind']

    initial_values_array, params_array = batch_configurator.grid_arrays(request, kind=kind)

    assert initial_values_array.shape[1] == batch_configurator.states.values_array.shape[0]
    assert params_array.shape[1] == batch_configurator.parameters.values_array.shape[0]

    if kind == 'combinatorial':
        assert initial_values_array.shape[0] == numinits * numparams
    elif kind == 'verbatim':
        assert initial_values_array.shape[0] == numinits








