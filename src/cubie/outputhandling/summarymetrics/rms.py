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
    """
    Summary metric to calculate the root mean square (RMS) of a variable.

    This metric computes the RMS value over a summary period by maintaining
    a running sum of squares in the buffer and calculating the square root
    of the mean when saving the final result.

    Notes
    -----
    The metric uses a single buffer slot per variable to accumulate the sum
    of squared values. The RMS is calculated as sqrt(sum_of_squares / n_samples).
    """

    def __init__(self):
        """
        Initialize the RMS summary metric.

        Creates CUDA device functions for updating running sums of squares and
        calculating RMS values, and configures the metric with appropriate
        buffer and output sizes.
        """
        super().__init__(
            name="rms",
            buffer_size=1,
            output_size=1,
        )

    def build(self):
        """
        Generate CUDA device functions for RMS value calculation.

        Creates optimized CUDA device functions with fixed signatures for
        updating running sums of squares and calculating final RMS values.

        Returns
        -------
        MetricFuncCache[update: callable, save: callable]
            Cache object containing update and save functions for CUDA
            execution. Both functions must be compiled with @cuda.jit
            decorators.

        Notes
        -----
        The generated functions have the following signatures:

        update(value, buffer, current_index, customisable_variable):
            Adds the square of the new value to the running sum.

        save(buffer, output_array, summarise_every, customisable_variable):
            Calculates RMS by taking square root of mean of squares and
            resets buffer.
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
            """
            Update running sum of squares with new value.

            Parameters
            ----------
            value : float
                New value to square and add to the running sum.
            buffer : array-like
                Buffer location containing the running sum of squares.
            current_index : int
                Current integration step index, used to reset sum at start.
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for RMS).

            Notes
            -----
            Squares the new value and adds it to buffer[0] to maintain the
            running sum of squares. Resets the sum if current_index is 0.
            Requires 1 buffer memory slot per variable.
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
            """
            Calculate RMS from running sum of squares and reset buffer.

            Parameters
            ----------
            buffer : array-like
                Buffer containing the running sum of squared values.
            output_array : array-like
                Output array location for saving the RMS value.
            summarise_every : int
                Number of steps between saves, used as divisor for mean
                calculation.
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for
                RMS).

            Notes
            -----
            Calculates the RMS by taking the square root of the mean of
            squares (sqrt(sum_of_squares / summarise_every)) and saves to
            output_array[0]. Resets buffer[0] to 0.0 for the next summary
            .
            Requires 1 output memory slot per variable.
            """
            output_array[0] = sqrt(buffer[0] / summarise_every)
            buffer[0] = 0.0

        # no cover: end
        return MetricFuncCache(update = update, save = save)
