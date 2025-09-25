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
    """Summary metric that tracks the maximum value of a variable.

    Notes
    -----
    A single buffer slot stores the running maximum. The buffer resets to
    ``-1.0e30`` after each save so any new value can replace it.
    """

    def __init__(self) -> None:
        """Initialise the Max summary metric with fixed buffer sizes."""
        super().__init__(
            name="max",
            buffer_size=1,
            output_size=1,
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum value calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback keeps the running maximum while the save callback
        writes the result and resets the buffer sentinel.
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
            """Update the running maximum with a new value.

            Parameters
            ----------
            value
                float. New value to compare against the current maximum.
            buffer
                device array. Storage for the current maximum value.
            current_index
                int. Current integration step index (unused for this metric).
            customisable_variable
                int. Metric parameter placeholder (unused for max).

            Notes
            -----
            Updates ``buffer[0]`` if the new value exceeds the current maximum.
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
            """Save the maximum value to output and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Buffer containing the current maximum value.
            output_array
                device array. Output location for saving the maximum value.
            summarise_every
                int. Number of steps between saves (unused for max).
            customisable_variable
                int. Metric parameter placeholder (unused for max).

            Notes
            -----
            Copies ``buffer[0]`` to ``output_array[0]`` and resets the buffer
            sentinel to ``-1.0e30`` for the next period.
            """
            output_array[0] = buffer[0]
            buffer[0] = -1.0e30  # A very non-maximal number

        # no cover: end
        return MetricFuncCache(update = update, save = save)
