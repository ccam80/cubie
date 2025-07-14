import numpy as np


def calculate_expected_summaries(state_input, observables_input,
                                 loop_compile_settings,
                                 output_functions,
                                 precision,
                                 ):
    """Helper function to calculate expected summary values from a given pair of state and observable arrays.
    Takes extra arguments from loop_compile_settings and output_functions - pass the fixture outputs straight
    to this function.

    Arguments:
    - state_input: 2D array of shape (n_states, n_samples) with the state data to summarise
    - observables_input: 2D array of shape (max_observables, n_samples) with the observable data to summarise
    - loop_compile_settings: Dictionary with settings for the loop function, including dt_summarise, dt_save,
        output_functions, saved_states, saved_observables, n_peaks
    - output_functions: Output functions object with summary_output_length method
    - precision: Numpy dtype to use for the output arrays (e.g. np.float32 or np.float64)

    Returns:
    - expected_state_summaries: 2D array of shape (n_saved_states * summary_size_per_state, n_samples)
    - expected_obs_summaries: 2D array of shape (n_saved_observables * summary_size_per_state, n_samples)
        """
    num_samples = state_input.shape[1]

    # #Check for a list of saved state/obs but that recording not requested
    # save_state_bool = "state" in loop_compile_settings['output_functions']
    # save_observables_bool = "observables" in loop_compile_settings['output_functions']
    # saved_states = np.asarray(loop_compile_settings['saved_states']) if save_state_bool else np.asarray([])
    # saved_observables = np.asarray(loop_compile_settings['saved_observables']) if save_observables_bool else np.asarray([])
    #
    # n_saved_states = len(saved_states)
    # n_saved_observables = len(saved_observables)

    summarise_every = int(loop_compile_settings['dt_summarise'] / loop_compile_settings['dt_save'])
    summary_samples = int(num_samples / summarise_every)
    summary_size_per_state = output_functions.memory_per_summarised_variable['output']

    state_summaries_height = summary_size_per_state * state_input.shape[0]
    obs_summaries_height = summary_size_per_state * observables_input.shape[0]

    expected_state_summaries = np.zeros((state_summaries_height, summary_samples), dtype=precision)
    expected_obs_summaries = np.zeros((obs_summaries_height, summary_samples), dtype=precision)

    for (_input_array, _output_array) in ((state_input, expected_state_summaries),
                                          (observables_input, expected_obs_summaries)
                                          ):
        calculate_single_summary_array(_input_array,
                                       summarise_every,
                                       summary_size_per_state,
                                       loop_compile_settings['output_functions'],
                                       loop_compile_settings['n_peaks'],
                                       _output_array,
                                       )

    return expected_state_summaries, expected_obs_summaries


def calculate_single_summary_array(input_array,
                                   summarise_every,
                                   summary_size_per_state,
                                   output_functions_list,
                                   n_peaks,
                                   output_array,
                                   ):
    """ Summarise states in input array in the same way that the device functions do.

    Arguments:
    - input_array: 2D array of shape (n_items, n_samples) with the input data to summarise
    - summarise_every: Number of samples to summarise over
    - summary_size_per_state: Number of summary values per state (e.g. 1 for mean, 1 + n_peaks for mean and peaks)
    - output_functions_list: List of output function names to apply (e.g. ["mean", "peaks", "max", "rms"])
    - n_peaks: Number of peaks to find in the "peaks" output function
    - output_array: 2D array to store the summarised output, shape (n_items * summary_size_per_state, n_samples)

    Returns:
    - None, but output_array is filled with the summarised values.

        """
    # Sort outputs list to match the order in build_output_functions
    types_order = ["mean", "peaks", "max", "rms"]
    sorted_outputs = sorted(output_functions_list,
                            key=lambda x: types_order.index(x) if x in types_order else len(types_order),
                            )
    summary_samples = int(input_array.shape[1] / summarise_every)
    n_items = input_array.shape[0]

    # Manual cycling through possible summaries to match the approach used when building the device functions
    for j in range(n_items):
        for i in range(summary_samples):
            summary_index = 0
            for output_type in sorted_outputs:
                start_index = i * summarise_every
                end_index = (i + 1) * summarise_every
                if output_type == 'mean':
                    output_array[j * summary_size_per_state + summary_index, i] = np.mean(
                            input_array[j, start_index: end_index], axis=0,
                            )
                    summary_index += 1

                if output_type == 'peaks':
                    # Use the last two samples, like the live version does
                    start_index = i * summarise_every - 2 if i > 0 else 0
                    maxima = local_maxima(
                            input_array[j, start_index: end_index],
                            )[:n_peaks] + start_index
                    output_start_index = j * summary_size_per_state + summary_index
                    output_array[output_start_index: output_start_index + maxima.size, i] = maxima
                    summary_index += n_peaks

                if output_type == 'max':
                    _max = np.max(input_array[j, start_index: end_index], axis=0)
                    output_array[j * summary_size_per_state + summary_index, i] = _max
                    summary_index += 1

                if output_type == 'rms':
                    rms = np.sqrt(np.mean(input_array[j, start_index: end_index] ** 2, axis=0))
                    output_array[j * summary_size_per_state + summary_index, i] = rms
                    summary_index += 1


def local_maxima(signal: np.ndarray) -> np.ndarray:
    return np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])) + 1


### ********************************************************************************************************* ###
#                                        RANDOM GENERATION
### ********************************************************************************************************* ###
def single_scale_float_array(shape: int | tuple[int], precision=np.float64, scale=1e6):
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


def mixed_scale_float_array(shape: int | tuple[int],
                            precision=np.float64,
                            log10_scale=(-6, 6),
                            axis=0,
                            ):
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
    _random_array = np.empty(shape, dtype=precision)
    for i in range(shape[axis]):
        _random_array[i] = rng.normal(scale=scale_values[i], size=shape[:axis] + shape[axis + 1:]).astype(precision)
    return _random_array


def random_array(precision, size: int | tuple[int], scale=1e6):
    """Generate a random float array of given size and dtype, drawn from a normal distribution with a std dev of the
    argument "scale". Normal was chosen here to slightly increase the magnitude-spread of values.

    Args:
        precision (np.dtype): The desired data type of the array.
        size (int): The size of the array to generate.
        scale (float): The standard deviation of the normal distribution from which to draw values.
    Returns:
        random_array (np.ndarray): A numpy array of the specified size and dtype, filled with random values.

    """
    if isinstance(scale, float):
        scale = (scale,)
    if len(scale) == 1:
        randvals = single_scale_float_array(size, precision, scale[0])
    elif len(scale) == 2:
        randvals = mixed_scale_float_array(size, precision, log10_scale=scale, axis=0)
    else:
        raise ValueError(f"scale must be a single float or a tuple of two floats, got {scale}.")

    return randvals


def nan_array(precision, size):
    """Generate an array of NaNs of given size and dtype.

    Args:
        precision (np.dtype): The desired data type of the array.
        size (int): The size of the array to generate.
    Returns:
        nan_array (np.ndarray): A numpy array of the specified size and dtype, filled with NaN values.
    """
    return np.full(size, np.nan, dtype=precision)


def zero_array(precision, size):
    """Generate an array of zeros of given size and dtype.

    Args:
        precision (np.dtype): The desired data type of the array.
        size (int): The size of the array to generate.
    Returns:
        zero_array (np.ndarray): A numpy array of the specified size and dtype, filled with zeros.
    """
    return np.zeros(size, dtype=precision)


def ones_array(precision, size):
    """Generate an array of ones of given size and dtype.

    Args:
        precision (np.dtype): The desired data type of the array.
        size (int): The size of the array to generate.
    Returns:
        one_array (np.ndarray): A numpy array of the specified size and dtype, filled with ones.
    """
    return np.ones(size, dtype=precision)


def generate_test_array(precision, size, style, scale=None):
    """Generate a test array of given size and dtype, with the specified type.

    Args:
        precision (np.dtype): The desired data type of the array.
        size (int): The size of the array to generate.
        style (str): The type of array to generate. Options: 'random', 'nan', 'zero', 'ones'.
        scale (float | tuple[float]): The scale for the random array, if type is 'random'. Default: None.
    Returns:
        test_array (np.ndarray): A numpy array of the specified size and dtype, filled with values according to the type.
    """
    if style == 'random':
        if scale is None:
            raise ValueError("scale must be specified if type is 'random'.")
        return random_array(precision, size, scale)
    elif style == 'nan':
        return nan_array(precision, size)
    elif style == 'zero':
        return zero_array(precision, size)
    elif style == 'ones':
        return ones_array(precision, size)
    #feature: a sin wave might be good to add here
    else:
        raise ValueError(f"Unknown array type: {style}. Use 'random', 'nan', 'zero', or 'ones'.")