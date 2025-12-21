"""
Minimum value summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that tracks the minimum value
encountered during integration for each variable.
"""

from numba import cuda

from cubie.cuda_simsafe import selp, compile_kwargs
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Min(SummaryMetric):
    """Summary metric that tracks the minimum value of a variable.

    Notes
    -----
    A single buffer slot stores the running minimum. The buffer resets to
    ``1.0e30`` after each save so any new value can replace it.
    """

    def __init__(self, precision) -> None:
        """Initialise the Min summary metric with fixed buffer sizes."""
        super().__init__(
            name="min",
            precision=precision,
            buffer_size=1,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for minimum value calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback keeps the running minimum while the save callback
        writes the result and resets the buffer sentinel.
        """

        precision = self.compile_settings.precision

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def update(
            value,
            buffer,
            offset,
            current_index,
            customisable_variable,
        ):
            """Update the running minimum with a new value.

            Parameters
            ----------
            value
                float. New value to compare against the current minimum.
            buffer
                device array. Full buffer containing metric working storage.
            offset
                int. Offset to this metric's storage within the buffer.
            current_index
                int. Current integration step index (unused for this metric).
            customisable_variable
                int. Metric parameter placeholder (unused for min).

            Notes
            -----
            Uses predicated commit to update ``buffer[offset + 0]`` if the new
            value is less than the current minimum, avoiding warp divergence.
            """
            update_flag = value < buffer[offset + 0]
            buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])

        @cuda.jit(
            # [
            #     "float32[::1], float32[::1], int32, int32",
            #     "float64[::1], float64[::1], int32, int32",
            # ],
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def save(
            buffer,
            buffer_offset,
            output_array,
            output_offset,
            summarise_every,
            customisable_variable,
        ):
            """Save the minimum value to output and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Full buffer containing metric working storage.
            buffer_offset
                int. Offset to this metric's storage within the buffer.
            output_array
                device array. Full output array for saving results.
            output_offset
                int. Offset to this metric's storage within the output.
            summarise_every
                int. Number of steps between saves (unused for min).
            customisable_variable
                int. Metric parameter placeholder (unused for min).

            Notes
            -----
            Copies ``buffer[buffer_offset + 0]`` to
            ``output_array[output_offset + 0]`` and resets the buffer
            sentinel to ``1.0e30`` for the next period.
            """
            output_array[output_offset + 0] = buffer[buffer_offset + 0]
            buffer[buffer_offset + 0] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
