"""
Extrema (both max and min) summary metric for CUDA-accelerated batch
integration.

This module implements a summary metric that tracks both the maximum and
minimum values encountered during integration for each variable.
"""

from numba import cuda

from cubie.cuda_simsafe import selp
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Extrema(SummaryMetric):
    """Summary metric that tracks both maximum and minimum values.

    Notes
    -----
    Uses two buffer slots: buffer[0] for maximum and buffer[1] for minimum.
    Outputs two values in the same order.
    """

    def __init__(self, precision) -> None:
        """Initialise the Extrema summary metric."""
        super().__init__(
            name="extrema",
            precision=precision,
            buffer_size=2,
            output_size=2,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for extrema calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback maintains both max and min while the save callback
        writes both results and resets the buffers.
        """

        precision = self.compile_settings.precision

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
        )
        def update(
            value,
            buffer,
            offset,
            current_index,
            customisable_variable,
        ):
            """Update the running maximum and minimum with a new value.

            Parameters
            ----------
            value
                float. New value to compare against current extrema.
            buffer
                device array. Full buffer containing metric working storage.
            offset
                int. Offset to this metric's storage within the buffer.
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Uses predicated commit to update ``buffer[offset + 0]`` (max) if
            value exceeds it, and ``buffer[offset + 1]`` (min) if value is
            less than it, avoiding warp divergence.
            """
            update_max = value > buffer[offset + 0]
            update_min = value < buffer[offset + 1]
            buffer[offset + 0] = selp(update_max, value, buffer[offset + 0])
            buffer[offset + 1] = selp(update_min, value, buffer[offset + 1])

        @cuda.jit(
            # [
            #     "float32[::1], float32[::1], int32, int32",
            #     "float64[::1], float64[::1], int32, int32",
            # ],
            device=True,
            inline=True,
        )
        def save(
            buffer,
            buffer_offset,
            output_array,
            output_offset,
            summarise_every,
            customisable_variable,
        ):
            """Save both extrema to output and reset the buffers.

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
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Saves max to ``output_array[output_offset + 0]`` and min to
            ``output_array[output_offset + 1]``, then resets buffers to their
            sentinel values.
            """
            output_array[output_offset + 0] = buffer[buffer_offset + 0]
            output_array[output_offset + 1] = buffer[buffer_offset + 1]
            buffer[buffer_offset + 0] = precision(-1.0e30)
            buffer[buffer_offset + 1] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
