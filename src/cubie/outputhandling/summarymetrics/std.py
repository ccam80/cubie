"""
Standard deviation summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that calculates the standard deviation
of values encountered during integration for each variable.
"""

from numba import cuda
from math import sqrt

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Std(SummaryMetric):
    """Summary metric that calculates the standard deviation of a variable.

    Notes
    -----
    The metric uses two buffer slots: one for the sum and one for the sum of
    squares. The standard deviation is calculated using the formula:
    std = sqrt((sum_of_squares / n) - (sum / n)^2)
    """

    def __init__(self) -> None:
        """Initialise the Std summary metric with fixed buffer sizes."""
        super().__init__(
            name="std",
            buffer_size=2,
            output_size=1,
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for standard deviation calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates both sum and sum of squares while the
        save callback computes the standard deviation and clears the buffer.
        """

        # no cover: start
        @cuda.jit(
            [
                "float32, float32[::1], int32, int32",
                "float64, float64[::1], int32, int32",
            ],
            device=True,
            inline=True,
        )
        def update(
            value,
            buffer,
            current_index,
            customisable_variable,
        ):
            """Update the running sum and sum of squares with a new value.

            Parameters
            ----------
            value
                float. New value to add to the running statistics.
            buffer
                device array. Storage containing [sum, sum_of_squares].
            current_index
                int. Current integration step index (unused for std).
            customisable_variable
                int. Metric parameter placeholder (unused for std).

            Notes
            -----
            Adds the value to buffer[0] (sum) and value^2 to buffer[1]
            (sum of squares).
            """
            buffer[0] += value
            buffer[1] += value * value

        @cuda.jit(
            [
                "float32[::1], float32[::1], int32, int32",
                "float64[::1], float64[::1], int32, int32",
            ],
            device=True,
            inline=True,
        )
        def save(
            buffer,
            output_array,
            summarise_every,
            customisable_variable,
        ):
            """Calculate the standard deviation from running statistics.

            Parameters
            ----------
            buffer
                device array. Buffer containing [sum, sum_of_squares].
            output_array
                device array. Output array location for saving the std value.
            summarise_every
                int. Number of steps contributing to each summary window.
            customisable_variable
                int. Metric parameter placeholder (unused for std).

            Notes
            -----
            Calculates std = sqrt((sum_sq/n) - (sum/n)^2) and saves to
            output_array[0], then resets buffer for the next summary period.
            """
            mean = buffer[0] / summarise_every
            mean_of_squares = buffer[1] / summarise_every
            variance = mean_of_squares - (mean * mean)
            output_array[0] = sqrt(variance)
            buffer[0] = 0.0
            buffer[1] = 0.0

        # no cover: end
        return MetricFuncCache(update=update, save=save)
