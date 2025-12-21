"""
Extrema (both max and min) first derivative summary metric for CUDA-accelerated
batch integration.

This module implements a summary metric that tracks both maximum and minimum
first derivative values encountered during integration for each variable using
finite differences.
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
class DxdtExtrema(SummaryMetric):
    """Summary metric that tracks both maximum and minimum derivative values.

    Notes
    -----
    Uses three buffer slots: buffer[0] for previous value, buffer[1] for
    maximum unscaled derivative, and buffer[2] for minimum unscaled derivative.
    Outputs two values: maximum derivative followed by minimum derivative.
    """

    def __init__(self, precision) -> None:
        """Initialise the DxdtExtrema summary metric."""
        super().__init__(
            name="dxdt_extrema",
            precision=precision,
            buffer_size=3,
            output_size=2,
            unit_modification="[unit]*s^-1",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for derivative extrema calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes finite differences and tracks both
        maximum and minimum unscaled derivatives. The save callback scales
        by dt_save and resets the buffers.
        """

        dt_save = self.compile_settings.dt_save
        precision = self.compile_settings.precision

        # no cover: start
        @cuda.jit(
            # [
            #     "float32, float32[::1], int32, int32",
            #     "float64, float64[::1], int32, int32",
            # ],
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
            """Update maximum and minimum first derivatives with a new value.

            Parameters
            ----------
            value
                float. New value to compute derivative from.
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
            Computes unscaled derivative as (value - buffer[offset + 0]) and
            updates buffer[offset + 1] if larger and buffer[offset + 2] if
            smaller. Uses predicated commit pattern to avoid warp divergence.
            """
            derivative_unscaled = value - buffer[offset + 0]
            update_max = (derivative_unscaled > buffer[offset + 1]) and (
                buffer[offset + 0] != precision(0.0)
            )
            update_min = (derivative_unscaled < buffer[offset + 2]) and (
                buffer[offset + 0] != precision(0.0)
            )
            buffer[offset + 1] = selp(
                update_max, derivative_unscaled, buffer[offset + 1]
            )
            buffer[offset + 2] = selp(
                update_min, derivative_unscaled, buffer[offset + 2]
            )
            buffer[offset + 0] = value

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
            """Save scaled derivative extrema and reset buffers.

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
            Scales the extrema by dt_save and saves to
            output_array[output_offset + 0] (max) and
            output_array[output_offset + 1] (min), then resets buffers to
            sentinel values.
            """
            output_array[output_offset + 0] = (
                buffer[buffer_offset + 1] / precision(dt_save)
            )
            output_array[output_offset + 1] = (
                buffer[buffer_offset + 2] / precision(dt_save)
            )
            buffer[buffer_offset + 1] = precision(-1.0e30)
            buffer[buffer_offset + 2] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
