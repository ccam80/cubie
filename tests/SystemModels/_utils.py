import numpy as np
from numba import float32, float64
from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel


def get_sizes_from_model(SystemClass):
    """Given a model with default labels for states etc, return the sizes of each array

    Args:
        SystemClass (GenericODE subclass): A class that inherits from GenericODE, with default values set. If you pass
            an instance of that class, it's precision will override the precision argument.
    Returns:
        A tuple of integers representing the number of (states, parameters, observables, constants, and drivers).
    """
    precision = np.float32
    sys, precision = instantiate_or_use_instance(SystemClass, precision)
    return sys.num_states, sys.num_parameters, sys.num_observables, sys.num_constants, sys.num_drivers


def random_float_array(shape, precision=np.float64, scale=1e6):
    """Generate a random float array of given shape and dtype, drawn from a normal distribution with a std dev of the
    argument "scale". Normal was chosen here to slightly increase the magnitude-spread of values.

    Args:
        shape (tuple[int] | int): The shape of the array to generate.
        precision (np.dtype): The desired data type of the array.
        scale (float): The standard deviation of the normal distribution from which to draw values.
    Returns:
        random_array (np.ndarray): A numpy array of the specified shape and dtype, filled with random values.

    """
    rng = np.random.default_rng()
    return rng.normal(scale=scale, size=shape).astype(precision)


def mixed_scale_float_array(shape, precision=np.float64, log10_scale=(-6, 6), axis=0):
    """ Generates a float array where each element is drawn from a normal distribution. The std dev of the distribution
    is 1*10^k, with drawn from a uniform distribution between log10_scale[0] and log10_scale[1]. The resulting array
    can be used to test the system with a wide dynamic range of values, straining the numerical stability of the system.

    Args:
        shape (tuple[int] | int): The shape of the array to generate.
        precision (np.dtype): The desired data type of the array. default: np.float64.
        log10_scale (tuple[float]): A tuple of (min_exponent, max_exponent) two floats, the lower and upper bounds of
            the log10 scale for the standard deviation. default: (-6, 6).
        axis (int): all values along this axis will be drawn from a distribution of the same scale - in the context of
            an ODE system, this means that each state will contain values at the same scale, so set it to the index
            that corresponds to the state/parameter/value. default: 0

    Returns:
        random_array (np.ndarray): A numpy array of the specified shape and dtype, filled with random values drawn from
            normal distributions with varying scales.

    """
    rng = np.random.default_rng()
    if isinstance(shape, int):
        shape = (shape,)
    if axis > len(shape):
        raise ValueError(f"Axis {axis} is out of bounds for shape {shape}.")
    scale_exponents = rng.uniform(log10_scale[0], log10_scale[1], size=shape[axis])
    scale_values = 10.0 ** scale_exponents
    random_array = np.empty(shape, dtype=precision)
    for i in range(shape[axis]):
        random_array[i] = rng.normal(scale=scale_values[i], size=shape[:axis] + shape[axis + 1:]).astype(precision)
    return random_array


def get_observables_list(SystemClass):
    """Get the list of observable names from a system class.
    Args:
        system_class (GenericODE subclass): A class that inherits from GenericODE, with default values set, or an
        instance thereof.
    Returns:
        list[str]: A list of observable names from the system class.
    """
    sys, precision = instantiate_or_use_instance(SystemClass, precision=np.float32)
    return [sys.observables.keys_by_index[i] for i in range(sys.num_observables)]


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
    n_states, n_params, n_obs, n_constants, n_drivers = get_sizes_from_model(sys)
    array_sizes = (n_states, n_params, n_constants)
    sysarrays_to_make = (sys.init_values, sys.parameters, sys.constants)
    dicts = []
    if isinstance(randscale, float):
        randscale = (randscale,)

    for i, sysarray in enumerate(sysarrays_to_make):
        if len(randscale) == 1:
            randvals = random_float_array(array_sizes[i], precision, randscale[0])
        elif len(randscale) == 2:
            randvals = mixed_scale_float_array(array_sizes[i], precision, log10_scale=randscale, axis=0)
        else:
            raise ValueError(f"randscale must be a single float or a tuple of two floats, got {randscale}.")

        keys = [sysarray.keys_by_index[i] for i in range(array_sizes[i])]
        dicts.append(dict(zip(keys, randvals)))

    state, parameters, constants = dicts
    drivers = random_float_array(n_drivers, precision)
    return state, parameters, drivers, constants


# Feature: This whole scenario could be handled more elegantly by using Hypothesis - if we repeat this logic in testing,
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

