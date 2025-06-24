if __name__ == "__main__":
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
    os.environ["NUMBA_OPT"] = "0"

from numba import cuda, float64, int32, from_dtype, float32
from math import sqrt
from dataclasses import dataclass
from numpy import asarray
from pdb import set_trace

"""Functions for saving values inside an integrator kernel, whether
they are states or summary metrics.

All functions should have the same signature and arguments, so that
they're easy to interchange without causing compilation headaches.

Each function will be called once for each state/output, rather than working
with arrays, because at the time of writing numba did not support the particular
slice assignment wanted in the integrator kernels.
"""
@dataclass
class OutputFunctions:
    save_state_func: cuda.jit(device=True, inline=True) = None
    update_summary_metrics_func: cuda.jit(device=True, inline=True) = None
    save_summary_metrics_func: cuda.jit(device=True, inline=True) = None
    shared_memory_requirements: int = 0,
    summary_output_length: int = 0


@dataclass
class Flags:
    save_state: bool = False
    save_observables: bool = False
    summarise_mean: bool = False          # extend with more toggles as needed
    summarise_max: bool = False
    summarise_peaks: bool = False
    summarise_rms: bool= False
    summarise: bool = False               # True if any of the summarise_* flags are set


_TOKENS_TO_COMPILE_FLAGS = {
    "state":   "save_state",
    "observables": "save_observables",
    "peaks":  "summarise_peaks",
    "mean":   "summarise_mean",
    "rms": "summarise_rms",
    "max":    "summarise_max",
}


class _TempMemoryRequirements(dict):
    """ A way to incorporate the number of peaks or other multiple-summary metrics into a memory requirements
    dictionary. Essentially a dictionary with a small set of variable keys."""
    def __init__(self, num_peaks: int):
        super().__init__({
            "state": 0,
            "observables": 0,
            "mean": 1,
            "peaks": 3 + num_peaks,   # prev + prev_prev + peak_counter
            "rms": 1,
            "max": 1,
        })


class _OutputMemoryRequirements(dict):
    """ A way to incorporate the number of peaks or other multiple-summary metrics into a memory requirements
    dictionary. Essentially a dictionary with a small set of variable keys."""
    def __init__(self, num_peaks: int):
        super().__init__({
            "state": 0,
            "observables": 0,
            "mean": 1,
            "peaks": num_peaks,
            "rms": 1,
            "max": 1,
        })

def parse_flags(tokens: list[str]) -> "Flags":
    f = Flags()
    for tok in tokens:
        try:
            attr = _TOKENS_TO_COMPILE_FLAGS[tok]
            setattr(f, attr, True)
        except KeyError:
            raise ValueError(f"Unknown option: {tok}")
    if f.summarise_mean or f.summarise_max or f.summarise_peaks or f.summarise_rms:
        f.summarise = True
    return f

def build_output_functions(outputs_list,
                          saved_states, # Ensure this has been set to the full range (handle None case before it gets  here)
                          saved_observables,
                          numpeaks=None):
    """Compile three functions: Save state, update summary metrics,and save summaries.

    Save state is called once per "save_every" loop steps, at the output frequency of the integrator.
    It saves the requested state values from the temp/working arrays to the output arrays.

    Update summary metrics is called once per "save_every" loop steps, at the output frequency of the integrator.
    It updates a temp/working array of summary values with summaries based on state/observables, for example
    updating a mean value or recording indices of a peak/max.

    Save summary metrics is called once per "summarise_every" loop steps, usually at a much higher frequency
    than the time-domain integrator output. This saves the running summary in the temp/working array to an output array.
    """
    # Possibly unnecessary: Assign function-local variables to ensure they appear in the compiled function's global scope
    # and so "baked into" the compiled function.
    if outputs_list is None:
        flags = Flags()
        flags.save_state = True
    else:
        flags = parse_flags(outputs_list)

    save_state = flags.save_state
    save_observables = flags.save_observables
    summarise = flags.summarise
    summarise_mean = flags.summarise_mean
    summarise_max = flags.summarise_max
    summarise_peaks = flags.summarise_peaks
    summarise_rms = flags.summarise_rms

    saved_states_local = asarray(saved_states)
    saved_observables_local = asarray(saved_observables)

    nstates = len(saved_states)
    nobs = len(saved_observables)
    numpeaks = numpeaks if numpeaks is not None else 0

    #Return memory per-state so that the number can be used to allocate separate arrays for each of state, observables.
    temporary_requirements = sum([_TempMemoryRequirements(numpeaks)[output_type] for output_type in outputs_list])
    output_requirements = sum([_OutputMemoryRequirements(numpeaks)[output_type] for output_type in outputs_list])

    @cuda.jit(device=True, inline=True)
    def save_state_func(current_state,
                        current_observables,
                        output_states_slice,
                        output_observables_slice):
        """Save the current state at the specified index."""
        if save_state:
            for k in range(nstates):
                output_states_slice[k] = current_state[saved_states_local[k]]

        if save_observables:
            for m in range(nobs):
                output_observables_slice[m] = current_observables[saved_observables_local[m]]

    @cuda.jit(device=True, inline=True)
    def update_summary_metrics_func(current_state,
                                    current_observables,
                                    temp_state_summaries,
                                    temp_observable_summaries,
                                    current_step):
        """Update the summary metrics based on the current state and observables."""
        # TODO: twiddle the internal layout of temporary array - in this form we cycle first through
        #  summary types, then states, memory arrangement should replicate this.

        #Efficiency note: this function (and the below) contain duplicated code for each array type. Using a common
        # function to handle the cases would be more efficient, but numba does not seem to recognise a change in the
        # compile-time constants nstates/nobs
        if summarise:

            temp_array_index = 0

            for k in range(nstates):
                if summarise_mean:
                    running_mean(current_state[saved_states_local[k]], temp_state_summaries, temp_array_index)
                    temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
                if summarise_peaks:
                    running_peaks(current_state[saved_states_local[k]], temp_state_summaries, temp_array_index,
                                  current_step, numpeaks)
                    temp_array_index += 3 + numpeaks
                if summarise_max:
                    running_max(current_state[saved_states_local[k]], temp_state_summaries, temp_array_index)
                    temp_array_index += 1
                if summarise_rms:
                    running_rms(current_state[saved_states_local[k]], temp_state_summaries, temp_array_index, current_step)
                    temp_array_index += 1

            if save_observables:
                temp_array_index = 0

                for k in range(nobs):
                    if summarise_mean:
                        running_mean(current_observables[saved_observables_local[k]], temp_observable_summaries, temp_array_index)
                        temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
                    if summarise_peaks:
                        running_peaks(current_observables[saved_observables_local[k]], temp_observable_summaries, temp_array_index,
                                      current_step, numpeaks)
                        temp_array_index += 3 + numpeaks
                    if summarise_max:
                        running_max(current_observables[saved_observables_local[k]], temp_observable_summaries, temp_array_index)
                        temp_array_index += 1
                    if summarise_rms:
                        running_rms(current_observables[saved_observables_local[k]], temp_observable_summaries, temp_array_index, current_step)
                        temp_array_index += 1

    @cuda.jit(device=True, inline=True)
    def save_summary_metrics_func(temp_state_summaries,
                                temp_observable_summaries,
                                output_state_summaries_slice,
                                output_observable_summaries_slice,
                                summarise_every):
        """Update the summary metrics based on the current state and observables."""
        # Efficiency note: this function (and the below) contain duplicated code for each array type. Using a common
        # function to handle the cases would be more efficient, but numba does not seem to recognise a change in the
        # compile-time constants nstates/nobs. This is probably expected behaviour for numba
        if summarise:
            output_index = 0
            temp_index = 0

            for i in range(nstates):
                # TODO: twiddle the internal layout of temporary array - in this form we cycle first through
                #  summary types, then states, memory arrangement should replicate this.
                #  Consider making a sub-function to handle the cases, to reduce code duplication.

                if summarise_mean:
                    output_state_summaries_slice[output_index] = temp_state_summaries[temp_index] / summarise_every
                    temp_state_summaries[temp_index] = 0.0
                    output_index += 1
                    temp_index += 1

                if summarise_peaks:
                    for p in range(numpeaks):
                        output_state_summaries_slice[output_index + p] = temp_state_summaries[temp_index + 3 + p]
                        temp_state_summaries[temp_index + 3 + p] = 0.0
                    temp_state_summaries[temp_index + 2] = 0.0  # Reset peak counter
                    output_index += numpeaks
                    temp_index += 3 + numpeaks

                if summarise_max:
                    output_state_summaries_slice[output_index] = temp_state_summaries[temp_index]
                    temp_state_summaries[temp_index] = -1.0e30 # A very negative number, to allow us to capture max values greater than this
                    output_index += 1
                    temp_index += 1

                if summarise_rms:
                    output_state_summaries_slice[output_index] = temp_state_summaries[temp_index] / summarise_every
                    output_state_summaries_slice[output_index] = sqrt(output_state_summaries_slice[output_index])
                    temp_state_summaries[temp_index] = 0.0
                    output_index += 1
                    temp_index += 1
            if save_observables:
                output_index = 0
                temp_index = 0

                for i in range(nobs):
                    # TODO: twiddle the internal layout of temporary array - in this form we cycle first through
                    #  summary types, then states, memory arrangement should replicate this.
                    #  Consider making a sub-function to handle the cases, to reduce code duplication.

                    if summarise_mean:
                        output_observable_summaries_slice[output_index] = temp_observable_summaries[temp_index] / summarise_every
                        temp_observable_summaries[temp_index] = 0.0
                        output_index += 1
                        temp_index += 1

                    if summarise_peaks:
                        for p in range(numpeaks):
                            output_observable_summaries_slice[output_index + p] = temp_observable_summaries[temp_index + 3 + p]
                            temp_observable_summaries[temp_index + 3 + p] = 0.0
                        temp_observable_summaries[temp_index + 2] = 0.0  # Reset peak counter
                        output_index += numpeaks
                        temp_index += 3 + numpeaks

                    if summarise_max:
                        output_observable_summaries_slice[output_index] = temp_observable_summaries[temp_index]
                        temp_observable_summaries[temp_index] = -1.0e30 # A very negative number, to allow us to capture max values greater than this
                        output_index += 1
                        temp_index += 1

                    if summarise_rms:
                        output_observable_summaries_slice[output_index] = temp_observable_summaries[temp_index] / summarise_every
                        output_observable_summaries_slice[output_index] = sqrt(output_observable_summaries_slice[output_index])
                        temp_observable_summaries[temp_index] = 0.0
                        output_index += 1
                        temp_index += 1

    # Return the functions and the shared memory requirements
    funcreturn = OutputFunctions()
    funcreturn.save_state_func = save_state_func
    funcreturn.update_summary_metrics_func = update_summary_metrics_func
    funcreturn.save_summary_metrics_func = save_summary_metrics_func
    funcreturn.temp_memory_requirements = temporary_requirements
    funcreturn.summary_output_length = output_requirements

    return funcreturn


@cuda.jit(device=True, inline=True)
def running_mean(value,
                temp_array,
                temp_array_start_index,
                ):
    """Update running mean - 1 temp memory slot required per state"""
    temp_array[temp_array_start_index] += value

@cuda.jit(device=True, inline=True)
def running_max(value,
                temp_array,
                temp_array_start_index,
                ):
    """Update the maximum value in the temporary array."""
    if value > temp_array[temp_array_start_index]:
        temp_array[temp_array_start_index] = value

@cuda.jit(device=True, inline=True)
def running_peaks(value,
                  temp_array,
                  temp_array_start_index,
                  current_index,
                  npeaks):
    """Update the peak value in the temporary array.
    Per-state temporary array requirements:
    [prev, prev_prev, peak_counter, peak1, peak2, ..., peakN]
    total requirement: 3 + npeaks slots per state"""

    prev = temp_array[temp_array_start_index + 0]
    prev_prev = temp_array[temp_array_start_index + 1]
    peak_counter = int(temp_array[temp_array_start_index + 2])

    #only check if we have enough points and we haven't already maxed out the counter, and that we're not working with
    # a 0.0 value (at the start of the run, for example). This assumes no natural 0.0 values, which seems realistic
    # for many systems. A more robust implementation would check if we're within 3 samples of summarise_every, probably.
    if (current_index > 2) and (peak_counter < npeaks) and (prev_prev != 0.0):
        if prev > value and prev_prev < prev:
            #Bingo
            temp_array[temp_array_start_index + 3 + peak_counter] = float(current_index - 1)
            temp_array[temp_array_start_index + 2] = float(int(temp_array[temp_array_start_index + 2]) + 1)

    temp_array[temp_array_start_index+0] = value  # Update previous value
    temp_array[temp_array_start_index + 1] = prev  # Update previous previous value

@cuda.jit(device=True, inline=True)
def running_rms(value,
                temp_array,
                temp_array_start_index,
                current_step):
    """Update the RMS value in the temporary array.
    Per-state temporary array requirements:
    [sum_of_squares]
    total requirement: 1 slots per state
    Divide by total number of steps**2 at the end to get RMS."""

    sum_of_squares = temp_array[temp_array_start_index]
    sum_of_squares += value * value
    temp_array[temp_array_start_index] = sum_of_squares
