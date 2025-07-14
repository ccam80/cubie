from numba import cuda


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
                  npeaks,
                  ):
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
    if (current_index >= 2) and (peak_counter < npeaks) and (prev_prev != 0.0):
        if prev > value and prev_prev < prev:
            #Bingo
            temp_array[temp_array_start_index + 3 + peak_counter] = float(current_index - 1)
            temp_array[temp_array_start_index + 2] = float(int(temp_array[temp_array_start_index + 2]) + 1)

    temp_array[temp_array_start_index + 0] = value  # Update previous value
    temp_array[temp_array_start_index + 1] = prev  # Update previous previous value


@cuda.jit(device=True, inline=True)
def running_rms(value,
                temp_array,
                temp_array_start_index,
                current_step,
                ):
    """Update the RMS value in the temporary array.
    Per-state temporary array requirements:
    [running_meansquare]
    total requirement: 1 slots per state
    We keep a running square, and mean/square root when saving.
    this accumulates some numerical error in the case of widely varying input scales, and tests have been
    relaxed to rtol=1e-3 for float32, 1e-9 for float64 to accomodate.
    """
    sum_of_squares = temp_array[temp_array_start_index]
    if current_step == 0:
        sum_of_squares = 0.0
    sum_of_squares += (value * value)
    temp_array[temp_array_start_index] = sum_of_squares