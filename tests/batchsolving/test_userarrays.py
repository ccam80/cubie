import pytest
import numpy as np
import pandas as pd
from cubie.batchsolving.UserArrays import UserArrays
from cubie.batchsolving.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving import summary_metrics
from cubie.batchsolving.BatchConfigurator import BatchConfigurator

# ------------THESE COPIED FROM TEST_SOLVERKERNEL UNTIL REFACTOR  ----------- #
@pytest.fixture(scope='function')
def batchconfig_instance(system):
    return BatchConfigurator.from_system(system)

@pytest.fixture(scope='function')
def square_drive(system, solver_settings, precision, request):
    """amplitude 1 square wave, request "cycles" to change default cycles per simulation from 5"""
    if hasattr(request, 'param'):
        if 'cycles' in request.param:
            cycles = request.getattr('cycles', 5)
    else:
        cycles = 5
    numvecs = system.sizes.drivers
    length = int(solver_settings['duration'] // solver_settings['dt_min'])
    driver = np.zeros((length, numvecs), dtype=precision)
    half_period = length//(2 * cycles)

    for i in range(cycles):
        driver[i*half_period:(i+1)*half_period, :] = 1.0

    return driver

@pytest.fixture(scope='function')
def batch_settings_override(request):
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope="function")
def batch_settings(batch_settings_override):
    """Fixture providing default batch settings."""
    defaults = {
        'num_state_vals_0': 2,
        'num_state_vals_1': 0,
        'num_param_vals_0': 2,
        'num_param_vals_1': 0,
        'kind': 'combinatorial',
    }

    if batch_settings_override:
        for key, value in batch_settings_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults

@pytest.fixture(scope='function')
def batch_request(system, batch_settings):
    """Parametrized batch settings."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.linspace(0.1, 1.0, batch_settings['num_state_vals_0']),
        # system)
        state_names[1]: np.linspace(0.1, 1.0, batch_settings['num_state_vals_1']),
        # system)
        param_names[0]: np.linspace(0.1, 1.0, batch_settings['num_param_vals_0']),
        param_names[1]: np.linspace(0.1, 1.0, batch_settings['num_param_vals_1']),
    }

@pytest.fixture(scope='function')
def batch_input_arrays(batch_request, batch_settings, batchconfig_instance,
                       square_drive):
    return batchconfig_instance.grid_arrays(batch_request,
                                                 kind=batch_settings['kind'])


# --------------------------------------------------------------------------- #


@pytest.fixture(scope='function')
def solver_with_arrays(solver, batch_input_arrays, solver_settings,
                       square_drive): # Kludge
    inits, params = batch_input_arrays

    solver.kernel.run(duration=solver_settings['duration'],
                     params=params,
                     inits=inits, # debug: inits has no varied parameters
                     forcing_vectors=square_drive,
                     blocksize=solver_settings['blocksize'],
                     stream=solver_settings['stream'],
                     warmup=solver_settings['warmup'])
    return solver
@pytest.mark.parametrize("state_flag, obs_flag, expected",
                         [ (True, True, lambda s, o: np.concatenate((s, o), axis=-1)),
                           (True, False, lambda s, o: s),
                           (False, True, lambda s, o: o),
                           (False, False, lambda s, o: np.array([])) ])
def test_time_domain_array_variants(state_flag, obs_flag, expected):
    active = ActiveOutputs(state=state_flag, observables=obs_flag)
    s = np.array([[[1, 2]], [[3, 4]]])
    o = np.array([[[5, 6]], [[7, 8]]])
    out = UserArrays.time_domain_array(active, s, o)
    exp = expected(s, o)
    assert np.array_equal(out, exp)

@pytest.mark.parametrize("state_flag, obs_flag, expected",
                         [ (True, True, lambda s, o: np.concatenate((s, o), axis=-1)),
                           (True, False, lambda s, o: s),
                           (False, True, lambda s, o: o),
                           (False, False, lambda s, o: np.array([])) ])
def test_summaries_array_variants(state_flag, obs_flag, expected):
    active = ActiveOutputs(state_summaries=state_flag, observable_summaries=obs_flag)
    # use 3D arrays: shape (2,1,2)
    s = np.array([[[10, 20]], [[30, 40]]])
    o = np.array([[[50, 60]], [[70, 80]]])
    out = UserArrays.summaries_array(active, s, o)
    exp = expected(s, o)
    assert np.array_equal(out, exp)

@pytest.fixture
def sample_userarrays():
    # simple arrays and legends
    ua = UserArrays(
        time_domain=np.array([[0, 1], [2, 3]]),
        summaries=np.array([[[1, 2], [3, 4]]])
    )
    # manually set legends for testing
    ua.time_domain_legend = {0: 't0', 1: 't1'}
    ua.summaries_legend = {0: 's0', 1: 's1'}
    ua._singlevar_summary_legend = {0: "mean", 1: "rms"}
    return ua

def test_as_numpy_outputs(sample_userarrays):
    out = sample_userarrays.as_numpy
    assert isinstance(out, dict)
    assert 'time_domain' in out and 'summaries' in out
    assert np.array_equal(out['time_domain'], sample_userarrays.time_domain)
    assert np.array_equal(out['summaries'], sample_userarrays.summaries)
    assert out['time_domain_legend'] == sample_userarrays.time_domain_legend
    assert out['summaries_legend'] == sample_userarrays.summaries_legend

@pytest.mark.parametrize("solver_settings_override", [{"output_types": ["state", "observables", "time", "mean",
                                                                        "rms"]}],
                    indirect=True)
def test_per_summary_arrays(solver_with_arrays):
    ua = UserArrays.from_solver(solver_with_arrays)
    per_summary = ua.per_summary_arrays
    singlevar_legend = ua._singlevar_summary_legend
    assert 'legend' in per_summary
    for k in singlevar_legend.values():
        assert k in per_summary

@pytest.mark.parametrize("solver_settings_override", [{"output_types": ["state", "observables", "time", "mean",
                                                                        "rms"]}],
                    indirect=True)
def test_time_domain_legend_from_solver(solver_with_arrays):
    legend = UserArrays.time_domain_legend_from_solver(solver_with_arrays)
    # indexes 0,1 for state, then time at last offset, then obs
    assert legend[0] == 'x0'
    assert "time" in legend.values()
    assert any(v.startswith('o') for v in legend.values())

@pytest.mark.parametrize("solver_settings_override", [{"output_types": ["state", "observables", "time", "mean",
                                                                        "rms"]}],
                    indirect=True)
def test_summary_legend_from_solver(solver_with_arrays):
    # should return a legend dict
    res = UserArrays.summary_legend_from_solver(solver_with_arrays)
    assert isinstance(res, dict)
    summaries = [s for s in solver_with_arrays.output_types if
                 any(s.startswith(metric) for metric in summary_metrics.implemented_metrics)]
    length = (len(solver_with_arrays.saved_state_indices) + len(solver_with_arrays.saved_observable_indices)) * len(summaries)

@pytest.mark.parametrize("solver_settings_override", [{"output_types": ["state", "observables", "time", "mean",
                                                                        "rms"]}],
                    indirect=True)
def test_from_and_update_solver(solver_with_arrays):
    ua = UserArrays.from_solver(solver_with_arrays)
    assert isinstance(ua, UserArrays)
    ua2 = UserArrays()
    ua2.update_from_solver(solver_with_arrays)
    assert ua == ua2

# @pytest.mark.parametrize("solver_settings_override", [{"output_types": ["state", "observables", "time", "mean",
#                                                                         "rms"]}],
#                     indirect=True)
# def test_as_pandas_conversion(solver_with_arrays):
#     ua = UserArrays.from_solver(solver_with_arrays)
#     td_df, sum_df = ua.as_pandas
#     array_shape = ua.time_domain.shape
#     equivalent_pd_shape = (array_shape[0], array_shape[1] * array_shape[2])
#     assert isinstance(td_df, pd.DataFrame)
#     assert equivalent_pd_shape == td_df.shape
#     assert list(sum_df.columns) == list(ua.summaries_legend.values())

# @pytest.mark.nocudasim
# @pytest.mark.parametrize(
#     "loop_compile_settings_overrides,expect_state,expect_obs",
#     [
#         ({'saved_state_indices': [0,1], 'saved_observable_indices': [], 'output_functions': ['state'],
#           'summarised_state_indices': [0], 'summarised_observable_indices': []}, True, False),
#         ({'saved_state_indices': [], 'saved_observable_indices': [0,1], 'output_functions': ['observables'],
#           'summarised_state_indices': [], 'summarised_observable_indices': [0]}, False, True),
#         ({'output_functions': ['state','observables']}, True, True),
#     ],
#     indirect=['loop_compile_settings_overrides']
# )
# def test_from_solver_active_flags(loop_compile_settings, solver_settings, solver_with_arrays, expect_state, expect_obs):
#     ua = UserArrays.from_solver(solver_with_arrays)
#     active = solver_with_arrays.active_output_arrays
#     assert active.state is expect_state
#     assert active.observables is expect_obs
#     # time_domain should reflect flags
#     td = ua.time_domain
#     if expect_state and expect_obs:
#         assert td.shape[2] == solver_with_arrays.device_state.shape[2] + solver_with_arrays.device_observables.shape[2]
#     elif expect_state:
#         assert td.shape == solver_with_arrays.device_state.shape
#     elif expect_obs:
#         assert td.shape == solver_with_arrays.device_observables.shape
#     else:
#         assert td.size == 0
