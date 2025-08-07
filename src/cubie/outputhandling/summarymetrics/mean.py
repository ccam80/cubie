from numba import cuda

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import \
    SummaryMetric, register_metric


@register_metric(summary_metrics)
class Mean(SummaryMetric):
    """
    Summary metric to calculate the mean of a set of values.
    """

    def __init__(self):
        update_func, save_func = self.CUDA_factory()

        super().__init__(name="mean", buffer_size=1, output_size=1,
                         update_device_func=update_func,
                         save_device_func=save_func, )

    def CUDA_factory(self):
        """
        Generate the CUDA functions to calculate the metric. The signatures
        of the functions are fixed:

        - update(value, buffer, current_index, customisable_variable)
            Perform math required to maintain a running prerequisite for the metric, like a sum or a count.
            Args:
                value (float): The new value to add to the running sum
                buffer (CUDA device array): buffer location (will be sized to accomodate self.buffer_size values)
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

            buffer[0] += value

        @cuda.jit(["float32[::1], float32[::1], int64, int64",
                   "float64[::1], float64[::1], int64, int64"], device=True,
                  inline=True, )
        def save(buffer, output_array, summarise_every,
                 customisable_variable, ):
            """Calculate mean from running sum - 1 output memory slot required per state"""
            output_array[0] = buffer[0] / summarise_every
            buffer[0] = 0.0

        return update, save
