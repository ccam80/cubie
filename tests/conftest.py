""" Fixtures for pytest and the functions that they use - any functions called by test modules should instead live
in tests/_utils.py"""

import pytest
import numpy as np
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions

"""Fixtures for instantiating lower-level components with default values that can be overriden through
indirect parametrization of the "override" fixture."""


@pytest.fixture(scope="function")
def expected_warning(request):
    """Tuple of (Warning type, warning message)"""
    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="function")
def expected_error(request):
    """Tuple of (Error type, error message)"""
    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="function")
def precision_override(request):
    return request.param if hasattr(request, 'param') else None


@pytest.fixture(scope="function")
def precision(precision_override):
    """
    Run tests with float32 by default, or override with float64.

    Usage:
    @pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    return precision_override if precision_override == np.float64 else np.float32


@pytest.fixture(scope="function")
def threecm_model(precision):
    from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
    threeCM = ThreeChamberModel(precision=precision)
    threeCM.build()
    return threeCM


@pytest.fixture(scope="function")
def decays_123_model(precision):
    from CuMC.SystemModels.Systems.decays import Decays
    decays3 = Decays(coefficients=[precision(1.0), precision(2.0), precision(3.0)], precision=precision)
    decays3.build()
    return decays3


@pytest.fixture(scope="function")
def decays_1_100_model(precision):
    from CuMC.SystemModels.Systems.decays import Decays
    decays100 = Decays(coefficients=np.arange(1, 101, dtype=precision), precision=precision)
    decays100.build()
    return decays100


def genericODE_settings(**kwargs):
    generic_ode_settings = {'constants':      {'c0': 0.0,
                                               'c1': 2.0,
                                               'c2': 3.0
                                               },
                            'initial_values': {'x0': 1.0,
                                               'x1': 0.0,
                                               'x2': 3.0
                                               },
                            'parameters':     {'p0': 2.0,
                                               'p1': 0.5,
                                               'p2': 5.5
                                               },
                            'observables':    {'o0': 4.2,
                                               'o1': 1.8,
                                               'o2': 4.6
                                               },
                            }
    generic_ode_settings.update(kwargs)
    return generic_ode_settings


@pytest.fixture(scope="function")
def genericODE_model_override(request):
    if hasattr(request, 'param'):
        return request.param
    return {}


@pytest.fixture(scope="function")
def genericODE_model(precision, genericODE_model_override):
    from CuMC.SystemModels.Systems.GenericODE import GenericODE
    generic = GenericODE(precision=precision, **genericODE_settings(**genericODE_model_override))
    generic.build()
    return generic


@pytest.fixture(scope="function")
def system_override(request):
    """Override for system model type, if provided."""
    print(request.param if hasattr(request, 'param') else {})
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope="function")
def system(request, system_override, precision):
    """
    Return the appropriate system model, defaulting to Decays123.

    Usage:
    @pytest.mark.parametrize("system_override", ["ThreeChamber"], indirect=True)
    def test_something(system):
        # system will be the ThreeChamber model here
    """
    # Use the override if provided, otherwise default to Decays123
    if system_override == {} or system_override is None:
        model_type = "Decays123"
    else:
        model_type = system_override

    # Initialize the appropriate model fixture based on the parameter
    if model_type == "ThreeChamber":
        model = request.getfixturevalue("threecm_model")
    elif model_type == "Decays123":
        model = request.getfixturevalue("decays_123_model")
    elif model_type == "Decays1_100":
        model = request.getfixturevalue("decays_1_100_model")
    elif model_type == "genericODE":
        model = request.getfixturevalue("genericODE_model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.build()
    return model


@pytest.fixture(scope='function')
def output_functions(loop_compile_settings, system):
    # Merge the default config with any overrides

    outputfunctions = OutputFunctions(system.sizes.states, system.sizes.parameters,
                                      loop_compile_settings['output_functions'],
                                      loop_compile_settings['saved_states'],
                                      loop_compile_settings['saved_observables'],
                                      )
    return outputfunctions


def update_loop_compile_settings(system, **kwargs):
    """The standard set of compile arguments, some of which aren't used by certain algorithms (like dtmax for a fixed step)."""
    loop_compile_settings_dict = {'dt_min':            0.001,
                                  'dt_max':            0.01,
                                  'dt_save':           0.01,
                                  'dt_summarise':      0.1,
                                  'atol':              1e-6,
                                  'rtol':              1e-3,
                                  'saved_states':      [0, 1],
                                  'saved_observables': [0, 1],
                                  'output_functions':  ["state"],
                                  }
    loop_compile_settings_dict.update(kwargs)
    return loop_compile_settings_dict


@pytest.fixture(scope='function')
def loop_compile_settings_overrides(request):
    """ Parametrize this fixture indirectly to change compile settings, no need to request this fixture directly
    unless you're testing that it worked."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def loop_compile_settings(request, system, loop_compile_settings_overrides):
    """
    Create a dictionary of compile settings for the loop function.
    This is the fixture your test should use - if you want to change the compile settings, indirectly parametrize the
    compile_settings_overrides fixture.
    """
    return update_loop_compile_settings(system, **loop_compile_settings_overrides)