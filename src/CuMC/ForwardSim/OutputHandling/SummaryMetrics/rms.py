from CuMC.ForwardSim.OutputHandling.SummaryMetrics import summary_metrics
from CuMC.ForwardSim.OutputHandling.SummaryMetrics.metrics import SummaryMetric, register_metric
from numba import cuda, float32
from math import sqrt

@register_metric(summary_metrics)
class Max(SummaryMetric):
    """
    Summary metric to calculate the mean of a set of values.
    """

    def __init__(self):
        update_func, save_func = self.CUDA_factory()

        super().__init__(name="rms",
                         temp_size=1,
                         output_size=1,
                         update_device_func=update_func,
                         save_device_func=save_func)

    def CUDA_factory(self):
        """
        Generate the CUDA functions to calculate the metric. The signatures of the functions are fixed:

        - update(value, temp_array, current_index, customisable_variable)
            Perform math required to maintain a running prerequisite for the metric, like a sum or a count.
            Args:
                value (float): The new value to add to the running sum
                temp_array (CUDA device array): Temporary array location (will be sized to accomodate self.temp_size values)
                current_index (int): Current index or time, given by the loop, for saving times at which things occur
                customisable_variable (scalar): An extra variable that can be used for metric-specific calculations,
                like the number of peaks to count or similar.
            Returns:
                nothing, modifies the temp_array in-place.

        - save(temp_array, output_array, summarise_every, customisable_variable):
            Perform final math to transform running variable into the metric, then reset temp array to a starting state.
            Args:
                temp_array (CUDA device array): Temporary array location which contains the running value
                output_array (CUDA device array): Output array location (will be sized to accomodate self.output_size values)
                summarise_every (int): Number of steps between saves, for calculating average metrics.
                customisable_variable (scalar): An extra variable that can be used for metric-specific calculations,
            Returns:
                nothing, modifies the output array in-place.
        """

        precision = 'float32'
        @cuda.jit(f"{precision}, {precision}[::1], int64, int64",
                  device=True,
                  inline=True)
        def update(value,
                   temp_array,
                   current_index,
                   customisable_variable,
                   ):
            """Update running sum - 1 temp memory slot required per state"""
            sum_of_squares = temp_array[0]
            if current_index == 0:
                sum_of_squares = 0.0
            sum_of_squares += (value * value)
            temp_array[0] = sum_of_squares

        @cuda.jit(f"{precision}[::1], {precision}[::1], int64, int64",
                  device=True,
                  inline=True)
        def save(temp_array,
                 output_array,
                 summarise_every,
                 customisable_variable,
                 ):
            """Calculate mean from running sum - 1 output memory slot required per state"""
            output_array = sqrt(temp_array[0] / summarise_every)
            temp_array = 0.0

        return update, save