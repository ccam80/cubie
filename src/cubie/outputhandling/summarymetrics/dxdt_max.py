"""
Maximum first derivative summary metric for CUDA-accelerated batch
integration.

This module implements a summary metric that tracks the maximum first
derivative (rate of change) encountered during integration for each variable
using finite differences.
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
class DxdtMax(SummaryMetric):
    """Summary metric that tracks maximum first derivative values.

    Notes
    -----
    Uses two buffer slots: buffer[0] for previous value and buffer[1] for
    maximum unscaled derivative. The derivative is computed using finite
    differences and scaled by dt_save in the save function.
    """

    def __init__(self, precision) -> None:
        """Initialise the DxdtMax summary metric."""
        super().__init__(
            name="dxdt_max",
            precision=precision,
            buffer_size=2,
            output_size=1,
            unit_modification="[unit]*s^-1",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum derivative calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes finite differences and tracks the
        maximum unscaled derivative. The save callback scales by dt_save
        and resets the buffers.
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
            """Update the maximum first derivative with a new value.

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
            updates buffer[offset + 1] if larger. Uses predicated commit
            pattern to avoid warp divergence.
            """
            derivative_unscaled = value - buffer[offset + 0]
            update_flag = (derivative_unscaled > buffer[offset + 1]) and (
                buffer[offset + 0] != precision(0.0)
            )
            buffer[offset + 1] = selp(
                update_flag, derivative_unscaled, buffer[offset + 1]
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
            """Save scaled maximum derivative and reset buffers.

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
            Scales the maximum unscaled derivative by dt_save and saves to
            output_array[output_offset + 0], then resets buffers to sentinel
            values.
            """
            output_array[output_offset + 0] = (
                buffer[buffer_offset + 1] / precision(dt_save)
            )
            buffer[buffer_offset + 1] = precision(-1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
