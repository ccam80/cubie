"""
Maximum magnitude summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that tracks the maximum absolute value
encountered during integration for each variable.
"""

from numba import cuda
from math import fabs

from cubie.cuda_simsafe import selp
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class MaxMagnitude(SummaryMetric):
    """Summary metric that tracks the maximum absolute value of a variable.

    Notes
    -----
    A single buffer slot stores the running maximum magnitude. The buffer
    resets to ``0.0`` after each save.
    """

    def __init__(self, precision) -> None:
        """Initialise the MaxMagnitude summary metric."""
        super().__init__(
            name="max_magnitude",
            precision=precision,
            buffer_size=1,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum magnitude calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback keeps the running maximum of absolute values while
        the save callback writes the result and resets the buffer.
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
            """Update the running maximum magnitude with a new value.

            Parameters
            ----------
            value
                float. New value whose absolute value is compared.
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
            Uses predicated commit to update ``buffer[offset + 0]`` if
            ``abs(value)`` exceeds the current maximum magnitude, avoiding
            warp divergence.
            """
            abs_value = fabs(value)
            update_flag = abs_value > buffer[offset + 0]
            buffer[offset + 0] = selp(update_flag, abs_value, buffer[offset + 0])

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
            """Save the maximum magnitude to output and reset the buffer.

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
            Copies ``buffer[buffer_offset + 0]`` to
            ``output_array[output_offset + 0]`` and resets the buffer
            to ``0.0`` for the next period.
            """
            output_array[output_offset + 0] = buffer[buffer_offset + 0]
            buffer[buffer_offset + 0] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
