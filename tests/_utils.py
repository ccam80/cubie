from __future__ import annotations

import attrs
import math
from typing import Mapping, Optional, Union

import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from cubie import SingleIntegratorRun
from cubie.outputhandling import OutputFunctions
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.odesystems.baseODE import BaseODE
from numpy.typing import NDArray

Array = NDArray[np.floating]

# --------------------------------------------------------------------------- #
#                      Standard Parameter Sets                                #
# --------------------------------------------------------------------------- #

MID_RUN_PARAMS = {
    'dt': 0.001,
    'save_every': 0.02,
    'summarise_every': 0.1,
    'dt_max': 0.5,
    'output_types': ['state', 'time', 'observables', 'mean'],
}

LONG_RUN_PARAMS = {
    'duration': 0.3,
    'dt': 0.0005,
    'save_every': 0.05,
    'summarise_every': 0.15,
    'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
}


STEP_CASES = [
    pytest.param({"algorithm": "euler", "step_controller": "fixed"}, id="euler"),
    pytest.param({"algorithm": "backwards_euler", "step_controller": "fixed"}, id="backwards_euler"),
    pytest.param({"algorithm": "backwards_euler_pc", "step_controller": "fixed"}, id="backwards_euler_pc"),
    pytest.param({"algorithm": "crank_nicolson", "step_controller": "pid"}, id="crank_nicolson"),
    pytest.param({"algorithm": "rosenbrock", "step_controller": "i"},
                 id="rosenbrock"),
    pytest.param({"algorithm": "erk", "step_controller": "pid"}, id="erk"),
    pytest.param({"algorithm": "dirk", "step_controller": "fixed"}, id="dirk"),
    pytest.param({"algorithm": "firk", "step_controller": "fixed"}, id="firk"),
    # Specific ERK tableaus
    pytest.param({"algorithm": "dormand-prince-54", "step_controller": "pid"}, id="erk-dormand-prince-54", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "cash-karp-54", "step_controller": "pid"}, id="erk-cash-karp-54", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "fehlberg-45", "step_controller": "i"},
                 id="erk-fehlberg-45", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "bogacki-shampine-32", "step_controller": "pid"}, id="erk-bogacki-shampine-32", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "heun-21", "step_controller": "fixed"}, id="erk-heun-21", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "ralston-33", "step_controller": "fixed"}, id="erk-ralston-33", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "classical-rk4", "step_controller": "fixed"}, id="erk-classical-rk4", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "dop853", "step_controller": "pid"},
                 id="erk-dop853", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "tsit5", "step_controller": "pid"}, id="erk-tsit5", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "vern7", "step_controller": "pid"}, id="erk-vern7", marks=pytest.mark.specific_algos),
    # Specific DIRK tableaus
    pytest.param({"algorithm": "implicit_midpoint", "step_controller": "fixed"}, id="dirk-implicit-midpoint", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "trapezoidal_dirk", "step_controller": "fixed"}, id="dirk-trapezoidal", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "sdirk_2_2", "step_controller": "fixed"},
                 id="dirk-sdirk-2-2", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "lobatto_iiic_3", "step_controller": "fixed"}, id="dirk-lobatto-iiic-3", marks=pytest.mark.specific_algos),\
    pytest.param({"algorithm": "l_stable_dirk_3", "step_controller": "fixed"}, id="dirk-l-stable-3", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "l_stable_sdirk_4", "step_controller": "pid"}, id="dirk-l-stable-4", marks=pytest.mark.specific_algos),
    # Specific FIRK tableaus
    pytest.param({"algorithm": "radau", "step_controller": "i"}, id="firk-radau", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "firk_gauss_legendre_2", "step_controller": "fixed"}, id="firk-gauss-legendre-2", marks=pytest.mark.specific_algos),
    # Specific Rosenbrock-W tableaus
    pytest.param({"algorithm": "ros3p", "step_controller": "pid"}, id="rosenbrock-ros3p", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "ode23s", "step_controller": "pid"},
                 id="rosenbrock-ode23s", marks=pytest.mark.specific_algos),
    pytest.param({"algorithm": "rodas3p", "step_controller": "pid"},
                 id="rosenbrock-rodas3p", marks=pytest.mark.specific_algos)
]



def merge_dicts(*dicts):
    """Merge multiple dictionaries, later dicts override earlier ones.

    Used to combine base settings (e.g., MID_RUN_PARAMS) with
    test-specific overrides into a single solver_settings_override.

    Parameters
    ----------
    *dicts : dict
        Dictionaries to merge. Later dicts override earlier ones.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def merge_param(base_settings, param):
    """Merge base settings into a pytest.param case.

    Combines base settings (e.g., MID_RUN_PARAMS) with test case settings,
    preserving pytest.param id and marks.

    Parameters
    ----------
    base_settings : dict
        Base settings to merge (applied first).
    param : pytest.param or dict
        Test case param. Can be pytest.param with id/marks or plain dict.

    Returns
    -------
    pytest.param
        Merged param with combined settings, original id and marks.
    """
    import pytest

    if hasattr(param, 'values'):
        # It's a pytest.param
        case_settings = param.values[0] if param.values else {}
        merged = merge_dicts(base_settings, case_settings)
        return pytest.param(
            merged,
            id=param.id,
            marks=param.marks if param.marks else (),
        )
    else:
        # It's a plain dict
        return pytest.param(merge_dicts(base_settings, param))

# Merged cases with STEP_OVERRIDES baked in
ALGORITHM_PARAM_SETS = [merge_param(MID_RUN_PARAMS, case)
                     for case in STEP_CASES]

def calculate_expected_summaries(
    state,
    observables,
    summarised_state_indices,
    summarised_observable_indices,
    samples_per_summary,
    output_types,
    summary_height_per_variable,
    precision,
    sample_summaries_every=1.0,
    exclude_first=False,
):
    """Helper function to calculate expected summary values from a given
    pair of state and observable arrays. Summarises the whole output state
    and observable array, select from within this if testing for selective
    summarisation.

    Arguments:
    - state: 2D array of shape (summary_samples, n_saved_states)
        output generated by system.
    - observables: 2D array of shape (summary_samples,
        n_saved_observables) output generated by system.
    - samples_per_summary: Number of samples to summarise over (batch size)
    - output_types: List of output function names to apply
        (e.g. ["mean", "peaks[3]", "max", "rms"])
    - precision: Numpy dtype to use for the output arrays
        (e.g. np.float32 or np.float64)
    - sample_summaries_every: Time between summary samples
        (for derivative calculations). Default: 1.0
    - exclude_first: If True, exclude the first sample (t=0) from summary
        calculations. Used when mimicking IVP loop behavior. Default: False.

    Returns:
    - expected_state_summaries: 2D array of shape (summary_samples,
        n_saved_states * summary_size_per_state)
    - expected_obs_summaries: 2D array of shape (summary_samples,
        n_saved_observables * summary_size_per_state)
    """
    # Optionally exclude t=0 row (first sample) from summary calculations
    # to match IVP loop behavior where first update_summaries is skipped
    if exclude_first:
        state = state[1:, summarised_state_indices]
        observables = observables[1:, summarised_observable_indices]
    else:
        state = state[:, summarised_state_indices]
        observables = observables[:, summarised_observable_indices]
    n_saved_states = state.shape[1]
    n_saved_observables = observables.shape[1]
    saved_samples = state.shape[0]
    summary_samples = int(saved_samples / samples_per_summary)

    state_summaries_height = summary_height_per_variable * n_saved_states
    obs_summaries_height = summary_height_per_variable * n_saved_observables

    expected_state_summaries = np.zeros(
        (summary_samples, state_summaries_height), dtype=precision
    )
    expected_obs_summaries = np.zeros(
        (summary_samples, obs_summaries_height), dtype=precision
    )

    for output in output_types:
        if output.startswith("peaks") or output.startswith("negative_peaks"):
            n_peaks = int(output.split('[')[1].split(']')[0]) if '[' in output else 0
        else:
            n_peaks = 0

    for _input_array, _output_array in (
        (state, expected_state_summaries),
        (observables, expected_obs_summaries),
    ):
        # When exclude_first=True, peak indices need +1 offset to convert
        # from sliced array indices to original save_idx values
        peak_index_offset = 1 if exclude_first else 0
        calculate_single_summary_array(_input_array, samples_per_summary,
                                       summary_height_per_variable,
                                       output_types,
                                       output_array=_output_array,
                                       sample_summaries_every=sample_summaries_every,
                                       peak_index_offset=peak_index_offset)

    return expected_state_summaries, expected_obs_summaries


def calculate_single_summary_array(
    input_array,
    samples_per_summary,
    summary_size_per_state,
    output_functions_list,
    output_array,
    sample_summaries_every=1.0,
    peak_index_offset=0,
):
    """Summarise states in input array in the same way that the device
    functions do.

    Arguments:
    - input_array: 2D array of shape (n_items, n_samples) with the input
        data to summarise
    - samples_per_summary: Number of samples to summarise over
    - summary_size_per_state: Number of summary values per state
        (e.g. 1 for mean, 1 + n_peaks for mean and peaks[n])
    - output_functions_list: List of output function names to apply
        (e.g. ["mean", "peaks[3]", "max", "rms"])
    - n_peaks: Number of peaks to find in the "peaks[n]" output function
    - output_array: 2D array to store the summarised output,
        shape (n_items * summary_size_per_state, n_samples)
    - sample_summaries_every: Time between summary samples
        (for derivative calculations). Default: 1.0
    - peak_index_offset: Offset added to peak indices. When exclude_first
        is True, this is 1 to convert sliced array indices to save_idx
        values. Default: 0.

    Returns:
    - None, but output_array is filled with the summarised values.

    """
    summary_samples = int(input_array.shape[0] / samples_per_summary)
    try:
        n_items = output_array.shape[1] // summary_size_per_state
    except ZeroDivisionError:
        n_items = 0

    # Manual cycling through possible summaries_array to match the approach
    # used when building the device functions
    for j in range(n_items):
        for i in range(summary_samples):
            summary_index = 0
            for output_type in output_functions_list:
                start_index = i * samples_per_summary
                end_index = (i + 1) * samples_per_summary
                if output_type == "mean":
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = np.mean(
                        input_array[start_index:end_index, j],
                        axis=0,
                    )
                    summary_index += 1

                if output_type.startswith("peaks"):
                    n_peaks = output_type.split("[", 1)[1].split("]", 1)[0]
                    n_peaks = int(n_peaks) if n_peaks else 0
                    # Use the last two samples, like the live version does
                    start_index = i * samples_per_summary - 2 if i > 0 else 0
                    maxima = (
                        local_maxima(
                            input_array[start_index:end_index, j],
                        )[:n_peaks]
                        + start_index
                        + peak_index_offset  # Offset for sliced array indexing
                    )
                    output_start_index = (
                        j * summary_size_per_state + summary_index
                    )
                    output_array[
                        i,
                        output_start_index : output_start_index + maxima.size,
                    ] = maxima
                    summary_index += n_peaks

                if output_type == "max":
                    _max = np.max(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _max
                    summary_index += 1

                if output_type == "rms":
                    rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = rms
                    summary_index += 1

                if output_type == "std":
                    std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = std
                    summary_index += 1

                if output_type == "min":
                    _min = np.min(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _min
                    summary_index += 1

                if output_type == "max_magnitude":
                    max_mag = np.max(
                        np.abs(input_array[start_index:end_index, j]), axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = max_mag
                    summary_index += 1

                if output_type == "extrema":
                    _max = np.max(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _min = np.min(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _min
                    summary_index += 2

                if output_type.startswith("negative_peaks"):
                    # Use the last two samples, like the live version does
                    start_index = i * samples_per_summary - 2 if i > 0 else 0
                    n_peaks = output_type.split("[", 1)[1].split("]", 1)[0]
                    n_peaks = int(n_peaks) if n_peaks else 0
                    minima = (
                        local_minima(
                            input_array[start_index:end_index, j],
                        )[:n_peaks]
                        + start_index
                        + peak_index_offset  # Offset for sliced array indexing
                    )
                    output_start_index = (
                        j * summary_size_per_state + summary_index
                    )
                    output_array[
                        i,
                        output_start_index : output_start_index + minima.size,
                    ] = minima
                    summary_index += n_peaks

                if output_type == "mean_std_rms":
                    _mean = np.mean(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _mean
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _std
                    output_array[
                        i, j * summary_size_per_state + summary_index + 2
                    ] = _rms
                    summary_index += 3

                if output_type == "mean_std":
                    _mean = np.mean(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _mean
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _std
                    summary_index += 2

                if output_type == "std_rms":
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _std
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _rms
                    summary_index += 2

                if output_type == "dxdt_max":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_max = np.max(derivatives)
                    else:
                        _dxdt_max = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_max
                    summary_index += 1

                if output_type == "dxdt_min":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_min = np.min(derivatives)
                    else:
                        _dxdt_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_min
                    summary_index += 1

                if output_type == "dxdt_extrema":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_max = np.max(derivatives)
                        _dxdt_min = np.min(derivatives)
                    else:
                        _dxdt_max = 0.0
                        _dxdt_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _dxdt_min
                    summary_index += 2

                if output_type == "d2xdt2_max":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_max = np.max(second_derivatives)
                    else:
                        _d2xdt2_max = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_max
                    summary_index += 1

                if output_type == "d2xdt2_min":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_min = np.min(second_derivatives)
                    else:
                        _d2xdt2_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_min
                    summary_index += 1

                if output_type == "d2xdt2_extrema":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_max = np.max(second_derivatives)
                        _d2xdt2_min = np.min(second_derivatives)
                    else:
                        _d2xdt2_max = 0.0
                        _d2xdt2_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _d2xdt2_min
                    summary_index += 2


def local_maxima(signal: np.ndarray) -> np.ndarray:
    """Find local maxima in a signal.
    
    Returns indices of local maxima. The +1 offset corrects for the
    signal[1:-1] slicing used in the comparison (flatnonzero returns
    indices into the sliced array, not the original signal).
    """
    return (
        np.flatnonzero(
            (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        )
        + 1  # Correct for signal[1:-1] indexing offset
    )


def local_minima(signal: np.ndarray) -> np.ndarray:
    """Find local minima in a signal.
    
    Returns indices of local minima. The +1 offset corrects for the
    signal[1:-1] slicing used in the comparison (flatnonzero returns
    indices into the sliced array, not the original signal).
    """
    return (
        np.flatnonzero(
            (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])
        )
        + 1  # Correct for signal[1:-1] indexing offset
    )


def deterministic_array(
    precision,
    size: Union[int, tuple[int]],
    scale=1.0
):
    """Generate a deterministic array of numerically challenging values.

    Creates reproducible test arrays with values spanning multiple orders
    of magnitude, including edge cases like near-zero values, large values,
    and mathematically interesting constants (π, e).

    Parameters
    ----------
    precision : numpy.dtype
        The desired data type of the array (np.float32 or np.float64).
    size : int or tuple of int
        The shape of the array to generate.
    scale : float, int, list, or tuple, optional
        Guidance for value magnitudes. Default is 1.0.
        - Single number: Values centered around that magnitude
        - Tuple/list of two numbers: Interpreted as (min_exp, max_exp)
          for values spanning 10^min_exp to 10^max_exp

    Returns
    -------
    numpy.ndarray
        A deterministic array of the specified shape and dtype filled
        with numerically challenging values.

    Notes
    -----
    The generated values include:
    - Very small positive values (1e-12, 1e-9, 1e-6, 1e-3)
    - Values near unity (0.1, 0.5, 1.0, 2.0)
    - Mathematical constants (π, e)
    - Large values (1e3, 1e6, 1e9, 1e12)
    - Alternating signs for additional coverage

    Values are tiled/broadcast to fill the requested shape and filtered
    based on the scale parameter to stay within appropriate ranges.
    """
    # Handle empty arrays
    if isinstance(size, int):
        shape = (size,)
    else:
        shape = tuple(size)
    total_elements = int(np.prod(shape))

    if total_elements == 0:
        return np.empty(shape, dtype=precision)

    # Interpret scale parameter
    if isinstance(scale, (list, tuple)) and len(scale) == 2:
        min_exp, max_exp = scale
    else:
        # Single scale value: create range centered around it
        if isinstance(scale, (list, tuple)):
            scale = scale[0]
        scale_exp = math.log10(abs(scale)) if scale != 0 else 0
        min_exp = scale_exp - 6
        max_exp = scale_exp + 6

    # Base set of challenging values (positive)
    base_values = [
        1e-12, 1e-9, 1e-6, 1e-3,
        0.1, 0.5, 1.0, 2.0,
        math.pi, math.e,
        1e3, 1e6, 1e9, 1e12,
    ]

    # Filter values to be within the scale range
    filtered_values = []
    for v in base_values:
        v_exp = math.log10(v)
        if min_exp <= v_exp <= max_exp:
            filtered_values.append(v)

    # Ensure we have at least some values
    if not filtered_values:
        # Use scale-appropriate values if filter removed everything
        mid_exp = (min_exp + max_exp) / 2
        filtered_values = [
            10 ** min_exp,
            10 ** ((min_exp + mid_exp) / 2),
            10 ** mid_exp,
            10 ** ((mid_exp + max_exp) / 2),
            10 ** max_exp,
        ]

    # Create array with alternating signs
    values_with_signs = []
    for i, v in enumerate(filtered_values):
        sign = 1 if i % 2 == 0 else -1
        values_with_signs.append(sign * v)

    # Tile values to fill the requested size
    num_base = len(values_with_signs)
    result = np.empty(total_elements, dtype=precision)
    for i in range(total_elements):
        result[i] = values_with_signs[i % num_base]

    return result.reshape(shape)


# ******************** Device Test Kernels *********************************  #
@attrs.define
class LoopRunResult:
    """Container holding the outputs produced by a single loop execution."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int
    counters: Array = None


def run_device_loop(
    singleintegratorrun: SingleIntegratorRun,
    system: BaseODE,
    initial_state: Array,
    solver_config: Mapping[str, float],
    driver_array: Optional[ArrayInterpolator] = None,

) -> LoopRunResult:
    """Execute ``loop`` on the CUDA simulator and return host-side outputs."""

    precision = system.precision
    save_every = singleintegratorrun.save_every
    warmup = solver_config["warmup"]
    duration = solver_config["duration"]
    t0 = solver_config["t0"]
    save_samples = int(np.floor(precision(duration) / precision(save_every))) + 1

    heights = singleintegratorrun.output_array_heights

    state_width = max(heights.state, 1)
    observable_width = max(heights.observables, 1)
    state_summary_width = max(heights.state_summaries, 1)
    observable_summary_width = max(heights.observable_summaries, 1)

    state_output = np.zeros((save_samples, state_width), dtype=precision)
    observables_output = np.zeros(
        (save_samples, observable_width), dtype=precision
    )

    summarise_every = singleintegratorrun.summarise_every
    summary_samples = int(np.floor(precision(duration) /
                                   precision(summarise_every)))

    state_summary_output = np.zeros(
        (summary_samples, state_summary_width), dtype=precision
    )
    observable_summary_output = np.zeros(
        (summary_samples, observable_summary_width), dtype=precision
    )
    
    # Iteration counters output (4 counters per save)
    counters_output = np.zeros((save_samples, 4), dtype=np.int32)

    params = np.array(
        system.parameters.values_array,
        dtype=precision,
        copy=True,
    )
    init_state = np.array(initial_state, dtype=precision, copy=True)
    status = np.zeros(1, dtype=np.int32)

    d_init = cuda.to_device(init_state)
    d_params = cuda.to_device(params)
    if driver_array is None:
        order = int(solver_config["driverspline_order"])
        width = min(system.num_drivers, 1)
        coeff_shape = (1, width, order + 1)
        driver_coefficients = np.zeros(coeff_shape, dtype=precision)
    else:
        driver_coefficients = np.array(
            driver_array.coefficients, dtype=precision, copy=True
        )
    d_driver_coeffs = cuda.to_device(driver_coefficients)
    d_state_out = cuda.to_device(state_output)
    d_obs_out = cuda.to_device(observables_output)
    d_state_sum = cuda.to_device(state_summary_output)
    d_obs_sum = cuda.to_device(observable_summary_output)
    d_counters_out = cuda.to_device(counters_output)
    d_status = cuda.to_device(status)

    shared_bytes = max(4,singleintegratorrun.shared_memory_bytes)
    shared_elements = max(1,singleintegratorrun.shared_memory_elements)
    persistent_required = max(1,singleintegratorrun.persistent_local_elements)

    loop_fn = singleintegratorrun.device_function
    numba_precision = from_dtype(precision)

    @cuda.jit(
            # (
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:,:,::1],
            #     numba_precision[:,::1],
            #     numba_precision[:,::1],
            #     numba_precision[:,::1],
            #     numba_precision[:,::1],
            #     numba_precision[:,::1],
            #     numba_precision[::1]
            # )
     )
    def kernel(
        init_vec,
        params_vec,
        driver_coeffs_vec,
        state_out_arr,
        obs_out_arr,
        state_sum_arr,
        obs_sum_arr,
        counters_out_arr,
        status_arr,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return

        shared = cuda.shared.array(shared_elements, dtype=numba_precision)
        shared[:] = numba_precision(0.0)
        local = cuda.local.array(persistent_required, dtype=numba_precision)
        local[:] = numba_precision(0.0)
        status_arr[0] = loop_fn(
            init_vec,
            params_vec,
            driver_coeffs_vec,
            shared,
            local,
            state_out_arr,
            obs_out_arr,
            state_sum_arr,
            obs_sum_arr,
            counters_out_arr,
            duration,
            warmup,
            t0,
        )

    kernel[1, 1, 0, shared_bytes](
        d_init,
        d_params,
        d_driver_coeffs,
        d_state_out,
        d_obs_out,
        d_state_sum,
        d_obs_sum,
        d_counters_out,
        d_status,
    )
    cuda.synchronize()

    state_host = d_state_out.copy_to_host()
    observables_host = d_obs_out.copy_to_host()
    state_summary_host = d_state_sum.copy_to_host()
    observable_summary_host = d_obs_sum.copy_to_host()
    counters_host = d_counters_out.copy_to_host()
    status_value = int(d_status.copy_to_host()[0])

    return LoopRunResult(
        state=state_host,
        observables=observables_host,
        state_summaries=state_summary_host,
        observable_summaries=observable_summary_host,
        counters=counters_host,
        status=status_value,
    )


def assert_integration_outputs(
    reference,
    device,
    output_functions,
    rtol: float,
    atol: float,
) -> None:
    """Compare state, summary, and time outputs between CPU and device."""
    if isinstance(reference, dict):
        reference = LoopRunResult(**reference)
    flags = output_functions.compile_flags
    if device.counters is None:
        print("\nNo counters provided")
    else:
        print(device.counters)
    state_ref, time_ref = extract_state_and_time(
        reference.state, output_functions
    )
    state_dev, time_dev = extract_state_and_time(
        device.state,
        output_functions,
    )
    observables_ref = reference.observables
    observables_dev = device.observables

    if output_functions.save_time:
        assert_allclose(
            time_dev,
            time_ref,
            rtol=rtol,
            atol=atol,
            err_msg="time mismatch.\n"
            f"device: {time_dev}\nreference: {time_ref}",
        )

    if flags.save_state:
        assert_allclose(
            state_dev,
            state_ref,
            rtol=rtol,
            atol=atol,
            verbose=True,
            err_msg="state mismatch.\n"
            f"device: {state_dev}\nreference: {state_ref}\ndelta (ref - "
            f"dev): {state_ref - state_dev}\n",
        )

    if flags.save_observables:
        assert_allclose(
            observables_dev,
            observables_ref,
            rtol=rtol,
            atol=atol,
            err_msg="observables mismatch.\n"
            f"device: {observables_dev}\n"
            f"reference: {observables_ref}",
        )

    if flags.summarise_state:
        assert_allclose(
            device.state_summaries,
            reference.state_summaries,
            rtol=rtol,
            atol=atol,
            err_msg="state summaries mismatch.\n"
                    f"device: {device.state_summaries}\n"
                    f"reference: {reference.state_summaries}")

    if flags.summarise_observables:
        assert_allclose(
            device.observable_summaries,
            reference.observable_summaries,
            rtol=rtol,
            atol=atol,
            err_msg="observable summary mismatch.\n"
            f"device: {device.observable_summaries}\n"
            f"reference: {reference.observable_summaries}",
        )


def extract_state_and_time(
    state_output: Array, output_functions: OutputFunctions
) -> tuple[Array, Optional[Array]]:
    """Split state output into state variables and optional time column."""
    n_state_columns = output_functions.n_saved_states
    if not output_functions.save_time:
        return state_output, None
    if state_output.ndim == 2:
        state_values = state_output[:, :n_state_columns]
        time_values = state_output[:, n_state_columns : n_state_columns + 1]
    else:
        state_values = state_output[:, :, :n_state_columns]
        time_values = state_output[:, :, n_state_columns : ]

    return state_values, time_values


def _driver_sequence(
    *,
    samples: int,
    total_time: float,
    n_drivers: int,
    precision,
) -> Array:
    """Drive system with a sine wave."""

    width = max(n_drivers, 1)
    drivers = np.zeros((samples, width), dtype=precision)
    if n_drivers > 0 and total_time > 0.0:
        times = np.linspace(0.0, total_time, samples, dtype=precision)
        for idx in range(n_drivers):
            drivers[:, idx] = precision(
                1.0 + np.sin(2 * np.pi * (idx + 1) * times / total_time))
    return drivers


def _build_enhanced_algorithm_settings(algorithm_settings, system, driver_array):
    """Add system and driver functions to algorithm settings.

    Functions are passed directly to get_algorithm_step, not stored
    in algorithm_settings dict.
    """
    enhanced = algorithm_settings.copy()
    enhanced['dxdt_function'] = system.dxdt_function
    enhanced['observables_function'] = system.observables_function
    enhanced['get_solver_helper_fn'] = system.get_solver_helper
    enhanced['n_drivers'] = system.num_drivers

    if driver_array is not None:
        enhanced['driver_function'] = driver_array.evaluation_function
        enhanced['driver_del_t'] = driver_array.driver_del_t
    else:
        enhanced['driver_function'] = None
        enhanced['driver_del_t'] = None

    return enhanced
