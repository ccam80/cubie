from numba import cuda

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import \
    SummaryMetric, register_metric


@register_metric(summary_metrics)
class Peaks(SummaryMetric):
    """
    Summary metric to calculate the mean of a set of values.
    """

    def __init__(self):
        update_func, save_func = self.CUDA_factory()

        # For metrics with a variable number of outputs, define sizes as
        # functions of a parameter to be passed when
        # requesting sizes.

        def buffer_size_func(n):
            return 3 + n

        def output_size_func(n):
            return n

        super().__init__(name="peaks", buffer_size=buffer_size_func,
                         output_size=output_size_func,
                         update_device_func=update_func,
                         save_device_func=save_func, )

    def CUDA_factory(self):
        """
        Generate the CUDA functions to calculate the metric. The signatures of the functions are fixed:

        - update(value, buffer, current_index, customisable_variable)
            Perform math required to maintain a running prerequisite for the metric, like a sum or a count.
            Args:
                value (float): The new value to add to the running sum
                buffer (CUDA device array): buffer location (will be sized to accomodate self.buffer_size 
                values)
                current_index (int): Current index or time, given by the loop, for saving times at which things occur
                customisable_variable (scalar): An extra variable that can be used for metric-specific calculations,
                like the number of peaks to count or similar.
            Returns:
                nothing, modifies the buffer in-place.

        - save(buffer, output_array, summarise_every, customisable_variable):
            Perform final math to transform running variable into the metric, then reset buffer to a starting state.
            Args:
                buffer (CUDA device array): buffer location which contains the running value
                output_array (CUDA device array): Output array location (will be sized to accomodate self.output_size values)
                summarise_every (int): Number of steps between saves, for calculating average metrics.
                customisable_variable (scalar): An extra variable that can be used for metric-specific calculations,
            Returns:
                nothing, modifies the output array in-place.
        """

        @cuda.jit(["float32, float32[::1], int64, int64",
                   "float64, float64[::1], int64, int64"], device=True,
                  inline=True, )
        def update(value, buffer, current_index, customisable_variable, ):
            """Update running sum - 1 buffer memory slot required per state"""
            npeaks = customisable_variable
            prev = buffer[0]
            prev_prev = buffer[1]
            peak_counter = int(buffer[2])

            # Check if we have enough points and we haven't already maxed out the counter, and that we're not working
            # with
            # a 0.0 value (at the start of the run, for example). This assumes no natural 0.0 values, which seems realistic
            # for many systems. A more robust implementation would check if we're within 3 samples of summarise_every, probably.
            if (current_index >= 2) and (peak_counter < npeaks) and (
                    prev_prev != 0.0):
                if prev > value and prev_prev < prev:
                    # Bingo
                    buffer[3 + peak_counter] = float(current_index - 1)
                    buffer[2] = float(int(buffer[2]) + 1)
            buffer[0] = value  # Update previous value
            buffer[1] = prev  # Update previous previous value

        @cuda.jit(["float32[::1], float32[::1], int64, int64",
                   "float64[::1], float64[::1], int64, int64"], device=True,
                  inline=True, )
        def save(buffer, output_array, summarise_every,
                 customisable_variable, ):
            """Calculate mean from running sum - 1 output memory slot required per state"""
            n_peaks = customisable_variable
            for p in range(n_peaks):
                output_array[p] = buffer[3 + p]
                buffer[3 + p] = 0.0
            buffer[2] = 0.0

        return update, save
