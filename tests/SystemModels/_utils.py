import numpy as np
from numba import float32, float64
from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
from tests._utils import generate_test_array

def get_observables_list(SystemClass):
    """Get the list of observable names from a system class.
    Args:
        system_class (GenericODE subclass): A class that inherits from GenericODE, with default values set, or an
        instance thereof.
    Returns:
        list[str]: A list of observable names from the system class.
    """
    sys, precision = instantiate_or_use_instance(SystemClass, precision=np.float32)
    return [sys.compile_settings.observables.keys_by_index[i] for i in range(sys.sizes.observables)]


def random_system_values(SystemClass, precision=np.float64, randscale=1e6, axis=0):
    """Generate random values for initial values, parameters, constants, drivers sized to match a given system.
    If randscale is a single float, then all values will be drawn from a normal distribution with that scale. If
    randscale is a tuple, then each state/parameter/constant will be drawn from a normal distribution with its own scale
    between 10**randscale[0] and 10**randscale[1]

    Args:
        SystemClass (GenericODE subclass): A class that inherits from GenericODE, with default values set.If you pass
            an instance of that class, it's precision will override the precision argument.
        precision (np.dtype): The desired data type of the arrays. Default is np.float64.
        randscale (float | tuple[float]): The scale for the random values. If a single float, all values will be drawn
            from a normal distribution with that scale. If a tuple, each state/parameter/constant will be drawn from a
            normal distribution with its own scale between 10**randscale[0] and 10**randscale[1]. Default is 1e6
        axis (int): The axis along which to apply the mixed scale, if randscale is a tuple. Default is 0.

    Returns:
        state (dict): A dictionary of initial values for the system's states.
        parameters (dict): A dictionary of parameters for the system.
        drivers (np.ndarray): An array of driver values for the system.
        constants (dict): A dictionary of constants for the system.
    """

    sys, precision = instantiate_or_use_instance(SystemClass, precision)
    sizes = sys.sizes
    n_states = sizes.states
    n_params = sizes.parameters
    n_constants = sizes.constants
    n_drivers = sizes.drivers
    n_obs = sizes.observables

    array_sizes = (n_states, n_params, n_constants)
    sysarrays_to_make = (sys.compile_settings.initial_states, sys.compile_settings.parameters,
                         sys.compile_settings.constants)
    dicts = []


    for i, sysarray in enumerate(sysarrays_to_make):
        randvals = generate_test_array(precision, array_sizes[i], style='random', scale=randscale)
        keys = [sysarray.keys_by_index[i] for i in range(array_sizes[i])]
        dicts.append(dict(zip(keys, randvals)))

    state, parameters, constants = dicts
    drivers = generate_test_array(precision, n_drivers, style='random', scale=randscale)
    return state, parameters, drivers, constants


# Improvement: This whole scenario could be handled more elegantly by using Hypothesis - if we repeat this logic in testing,
#  make the move.
def create_random_test_set(SystemClass, precision=np.float64, randscale=1e6):
    """ Creates a random test_set for a given system class. The test set includes random initial values, parameters,
    drivers, and constants, sized to match the system's requirements.

    Args:
        SystemClass (GenericODE subclass): A class that inherits from GenericODE, with default values set.If you pass
            an instance of that class, it's precision will override the precision argument.
        precision (np.dtype): The desired data type for the arrays. Default is np.float64.
        randscale (float | tuple[float]): The scale for the random values. If a single float, all values will be drawn
            from a normal distribution with that scale. If a tuple, each state/parameter/constant will be drawn from a
            normal distribution with its own scale between 10**randscale[0] and 10**randscale[1]. Default is 1e6.

    Returns:
        test_set (tuple): A tuple containing:
            - instantiation_parameters (tuple): Parameters for instantiating the system.
            - input_data (tuple): Input data for the system.
            - description (str): Description of the test set.

    """
    inits, params, drivers, constants = random_system_values(SystemClass, precision, randscale)
    observables = get_observables_list(SystemClass)

    instantiation_parameters = (precision, inits, params, observables, constants, len(drivers))

    input_inits = np.asarray(list(inits.values()), dtype=precision)
    input_params = np.asarray(list(params.values()), dtype=precision)
    drivers = np.asarray(drivers, dtype=precision)

    input_data = (input_inits, input_params, drivers)

    return (instantiation_parameters, input_data,
            f"Random test set with numbers of scale {randscale} of type {precision}")


def create_minimal_input_sets(SystemClass, precision=np.float64):
    """Create system test sets with incomplete data to test error handling. This function returns the following test
    cases:

    - All instantiation inputs are empty lists or dicts, relying on default values in the system class for instantiation.
        dxdt function inputs are random.
    - All inputs are None, which should also rely on default values in the system class. dxdt function inputs are random.
    - All inputs are filled with zero values, and so are dxdt function inputs. Test for any surprising divide-by-zero
        errors.
    - Each instantiation input is missing one key, which should still allow instantiation, filling the missing keys
        with default values. dxdt function inputs are random.

    Args:
        SystemClass (GenericODE subclass): A class that inherits from GenericODE, with default values set. If you pass
            an instance of that class, it's precision will override the precision argument.
        precision (np.dtype): The desired data type for the arrays. Default is np.float64.

    returns:
        incomplete_sets (list[tuple]): A list of tuples, each containing:
            - instantiation_parameters (tuple): Parameters for instantiating the system.
            - input_data (tuple): Input data for the system.
            - description (str): Description of the test set.
    - """
    inits, params, drivers, constants = random_system_values(SystemClass, precision)
    observables = get_observables_list(SystemClass)

    input_inits = np.asarray(list(inits.values()), dtype=precision)
    input_params = np.asarray(list(params.values()), dtype=precision)
    drivers = np.asarray(drivers, dtype=precision)
    ndrivers = len(drivers)
    incomplete_sets = []

    # Set with zeros for all values
    zeros_inits = {k: precision(0.0) for k in inits}
    zeros_params = {k: precision(0.0) for k in params}
    zeros_constants = {k: precision(0.0) for k in constants}
    incomplete_sets.append(
        ((precision, zeros_inits, zeros_params, observables, zeros_constants, ndrivers),
         (input_inits,
          input_params,
          drivers),
         "All zeros")
    )

    # Set with None for all values
    nones_inits = {k: None for k in inits}
    nones_params = {k: None for k in params}
    nones_constants = {k: None for k in constants}
    incomplete_sets.append(
        ((precision, nones_inits, nones_params, observables, nones_constants, ndrivers),
         (input_inits,
          input_params,
          drivers),
         "All nones")
    )

    # Set with lists as values
    lists_inits = {k: [v] for k, v in inits.items()}
    lists_params = {k: [v] for k, v in params.items()}
    lists_constants = {k: [v] for k, v in constants.items()}
    incomplete_sets.append(
        ((precision, lists_inits, lists_params, observables, lists_constants, ndrivers),
         (input_inits,
          input_params,
          drivers),
         "All lists")
    )

    # Set with partial keys (remove one key from each)
    partial_inits = dict(list(inits.items())[1:])
    partial_params = dict(list(params.items())[1:])
    partial_constants = dict(list(constants.items())[1:])
    incomplete_sets.append(
        ((precision, partial_inits, partial_params, observables, partial_constants, ndrivers),
         (input_inits,
          input_params,
          drivers),
         "Partial keys")
    )

    return incomplete_sets


def instantiate_or_use_instance(obj, precision=np.float64):
    """If a class is passed, instantiate it with default parameters and precision, but if an instance is
     passed, return it directly and match its precision.

     Args:
         obj (class | class instance): A class or instance of that class.
         precision (np.dtype): The desired data type for the arrays. Default is np.float64.

    Returns:
        tuple: A tuple containing the instance of the class and the precision used.
    """

    if isinstance(obj, type):
        instance = obj(precision=precision)
        return instance, precision
    else:
        if obj.precision == float32:
            precision = np.float32
        elif obj.precision == float64:
            precision = np.float64
        return obj, precision


def generate_system_tests(SystemClass, log10_scalerange=(-6, 6), tests_per_category=5):
    """Generate a list of tests checking correct input/output across a range of floating point scales and both
    float precision types. The tests include:
        - Random tests at scales spread across the given range.
        - Mixed-scale random tests.
        - Incomplete input sets to test error handling.


    Args:
        SystemClass (GenericODE subclass): A class that inherits from GenericODE, with default values set. If an instance
            (which will have a precision attribute) is passed, tests will only be generated for that precision.
        log10_scalerange (tuple[float]): A tuple of (min_exponent, max_exponent) two floats, the lower and upper bounds
            of the log10 scale for the random values. Default is (-6, 6).
        tests_per_category (int): The number of tests to generate for each category. Default is 5.

    """
    if isinstance(SystemClass, type):
        precisions = (np.float32, np.float64)
    else:
        precisions = (SystemClass.precision,)


    test_cases = []
    samescales = np.arange(log10_scalerange[0], log10_scalerange[1] + 1, tests_per_category)
    for precision in (precisions):
        test_cases += [create_random_test_set(SystemClass, precision, 10.0 ** scale) for scale in samescales]

        # mixed-scale random tests
        test_cases += [create_random_test_set(SystemClass, precision, log10_scalerange) for scale in range(tests_per_category)]

    #Incomplete input sets
    test_cases += create_minimal_input_sets(SystemClass, precision)

    return test_cases
