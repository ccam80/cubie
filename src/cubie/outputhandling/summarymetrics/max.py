"""
Maximum value summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that tracks the maximum value
encountered during integration for each variable.
"""

from numba import cuda

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Max(SummaryMetric):
    """
    Summary metric to calculate the maximum value of a variable.

    This metric tracks the maximum value encountered during integration
    by maintaining a running maximum in the buffer and outputting the
    final maximum at the end of each summary period.

    Notes
    -----
    The metric uses a single buffer slot per variable to store the current
    maximum value. The buffer is initialized to a very negative number
    (-1.0e30) to ensure any real value will be captured as the maximum.
    """

    def __init__(self):
        """
        Initialize the Max summary metric.

        Creates CUDA device functions for updating and saving maximum values
        and configures the metric with appropriate buffer and output sizes.
        """
        super().__init__(
            name="max",
            buffer_size=1,
            output_size=1,
        )

    def build(self):
        """
        Generate CUDA device functions for maximum value calculation.

        Creates optimized CUDA device functions with fixed signatures for
        updating running maximums and saving final results.

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
            Updates the running maximum if the new value is larger.

        save(buffer, output_array, summarise_every, customisable_variable):
            Saves the current maximum to output and resets buffer.
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
            Update running maximum with new value.

            Parameters
            ----------
            value : float
                New value to compare against current maximum.
            buffer : array-like
                Buffer location containing the current maximum value.
            current_index : int
                Current integration step index (unused for max calculation).
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for
                max).

            Notes
            -----
            Updates buffer[0] in-place if the new value is greater than the
            current maximum. Requires 1 buffer memory slot per variable.
            """
            if value > buffer[0]:
                buffer[0] = value

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
            Save maximum value to output and reset buffer.

            Parameters
            ----------
            buffer : array-like
                Buffer containing the current maximum value.
            output_array : array-like
                Output array location for saving the maximum value.
            summarise_every : int
                Number of steps between saves (unused for max calculation).
            customisable_variable : int
                Extra parameter for metric-specific calculations (unused for
                max).

            Notes
            -----
            Saves the maximum value from buffer[0] to output_array[0] and
            resets the buffer to a very negative number (-1.0e30) to prepare
            for the next summary period. Requires 1 output memory slot per
            variable.
            """
            output_array[0] = buffer[0]
            buffer[0] = -1.0e30  # A very non-maximal number

        # no cover: end
        return MetricFuncCache(update = update, save = save)
