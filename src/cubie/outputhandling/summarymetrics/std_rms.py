"""
Composite metric for standard deviation and RMS calculations.

This module implements a composite summary metric that efficiently computes
standard deviation and RMS from a single pass over the data using shared
running sums. This is more efficient than computing each separately when
both metrics are needed.
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
class StdRms(SummaryMetric):
    """Composite metric that calculates std and rms together.

    Notes
    -----
    Uses two buffer slots for sum and sum_of_squares, which are used to
    compute both output metrics. This is more efficient than computing
    std and rms separately when both are needed.
    
    The output array contains [std, rms] in that order.
    """

    def __init__(self) -> None:
        """Initialise the StdRms composite metric."""
        super().__init__(
            name="std_rms",
            buffer_size=2,
            output_size=2,
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for composite calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates sum and sum_of_squares while the
        save callback computes both metrics from these running sums
        and clears the buffer.
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
            """Update running sums with a new value.

            Parameters
            ----------
            value
                float. New value to add to the running statistics.
            buffer
                device array. Storage containing [sum, sum_of_squares].
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

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
            """Calculate std and rms from running sums.

            Parameters
            ----------
            buffer
                device array. Buffer containing [sum, sum_of_squares].
            output_array
                device array. Output location for [std, rms].
            summarise_every
                int. Number of steps contributing to each summary window.
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Calculates:
            - mean = sum / n (intermediate)
            - std = sqrt((sum_sq/n) - (mean)^2)
            - rms = sqrt(sum_sq / n)
            
            Saves to output_array[0:2] and resets buffer for next period.
            """
            mean = buffer[0] / summarise_every
            mean_of_squares = buffer[1] / summarise_every
            variance = mean_of_squares - (mean * mean)
            
            output_array[0] = sqrt(variance)
            output_array[1] = sqrt(mean_of_squares)
            
            buffer[0] = 0.0
            buffer[1] = 0.0

        # no cover: end
        return MetricFuncCache(update=update, save=save)
