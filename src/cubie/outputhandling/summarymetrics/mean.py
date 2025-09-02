"""
Mean value summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that calculates the arithmetic mean
of values encountered during integration for each variable.
"""

from numba import cuda

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Mean(SummaryMetric):
    """
    Summary metric to calculate the arithmetic mean of a variable.

    This metric computes the mean value over a summary period by maintaining
    a running sum in the buffer and dividing by the number of samples when
    saving the final result.

    Notes
    -----
    The metric uses a single buffer slot per variable to accumulate the sum
    of values. The mean is calculated by dividing this sum by the number
    of integration steps in the summary period.
    """

    def __init__(self):
        """
        Initialize the Mean summary metric.

        Creates CUDA device functions for updating running sums and
        calculating mean values, and configures the metric with appropriate
        buffer and output sizes.
        """
        super().__init__(
            name="mean",
            buffer_size=1,
            output_size=1,
        )

    def build(self):
        """
        Generate CUDA device functions for mean value calculation.

        Creates optimized CUDA device functions with fixed signatures for
        updating running sums and calculating final mean values.

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
            Adds the new value to the running sum.

        save(buffer, output_array, summarise_every, customisable_variable):
            Calculates mean by dividing sum by number of steps and resets
            buffer.
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
            Update running sum with new value.

            Parameters
            ----------
            value : float
                New value to add to the running sum.
            buffer : array-like
                Buffer location containing the running sum.
            current_index : int
                Current integration step index (unused for mean calculation).
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for
                mean).

            Notes
            -----
            Adds the new value to buffer[0] in-place to maintain the running
            sum. Requires 1 buffer memory slot per variable.
            """
            buffer[0] += value

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
            Calculate mean from running sum and reset buffer.

            Parameters
            ----------
            buffer : array-like
                Buffer containing the running sum of values.
            output_array : array-like
                Output array location for saving the mean value.
            summarise_every : int
                Number of steps between saves, used as divisor for mean
                calculation.
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for
                mean).

            Notes
            -----
            Calculates the mean by dividing the running sum by the number of
            integration steps (summarise_every) and saves to output_array[0].
            Resets buffer[0] to 0.0 for the next summary period  Requires 1
            output memory slot per variable.
            """
            output_array[0] = buffer[0] / summarise_every
            buffer[0] = 0.0

        # no cover: end
        return MetricFuncCache(update = update, save = save)
