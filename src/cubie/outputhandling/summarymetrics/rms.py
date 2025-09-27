"""
Root Mean Square (RMS) summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that calculates the root mean square
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
class RMS(SummaryMetric):
    """Summary metric that calculates the root mean square of a variable.

    Notes
    -----
    The metric keeps a running sum of squares in a single buffer slot and
    evaluates ``sqrt(sum_of_squares / summarise_every)`` when saving results.
    """

    def __init__(self) -> None:
        """Initialise the RMS summary metric with fixed buffer sizes."""
        super().__init__(
            name="rms",
            buffer_size=1,
            output_size=1,
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for RMS value calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates the running sum of squares while the
        save callback computes the RMS and clears the buffer.
        """

        # no cover: start
        @cuda.jit(
            [
                "float32, float32[::1], int64, int64",
                "float64, float64[::1], int64, int64",
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
            """Update the running sum of squares with a new value.

            Parameters
            ----------
            value
                float. New value to square and add to the running sum.
            buffer
                device array. Storage containing the running sum of squares.
            current_index
                int. Current integration step index, used to reset the sum.
            customisable_variable
                int. Metric parameter placeholder (unused for RMS).

            Notes
            -----
            Resets ``buffer[0]`` on the first step of a period before adding
            the squared value.
            """
            sum_of_squares = buffer[0]
            if current_index == 0:
                sum_of_squares = 0.0
            sum_of_squares += value * value
            buffer[0] = sum_of_squares

        @cuda.jit(
            [
                "float32[::1], float32[::1], int64, int64",
                "float64[::1], float64[::1], int64, int64",
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
            """Calculate the RMS from the running sum of squares.

            Parameters
            ----------
            buffer
                device array. Buffer containing the running sum of squares.
            output_array
                device array. Output array location for saving the RMS value.
            summarise_every
                int. Number of steps contributing to each summary window.
            customisable_variable
                int. Metric parameter placeholder (unused for RMS).

            Notes
            -----
            Saves ``sqrt(buffer[0] / summarise_every)`` to ``output_array[0]``
            and resets ``buffer[0]`` for the next summary period.
            """
            output_array[0] = sqrt(buffer[0] / summarise_every)
            buffer[0] = 0.0

        # no cover: end
        return MetricFuncCache(update = update, save = save)
