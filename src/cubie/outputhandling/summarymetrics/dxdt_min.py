"""
Minimum first derivative summary metric for CUDA-accelerated batch
integration.

This module implements a summary metric that tracks the minimum first
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
class DxdtMin(SummaryMetric):
    """Summary metric that tracks minimum first derivative values.

    Notes
    -----
    Uses two buffer slots: buffer[0] for previous value and buffer[1] for
    minimum unscaled derivative. The derivative is computed using finite
    differences and scaled by dt_save in the save function.
    """

    def __init__(self, precision) -> None:
        """Initialise the DxdtMin summary metric."""
        super().__init__(
            name="dxdt_min",
            precision=precision,
            buffer_size=2,
            output_size=1,
            unit_modification="[unit]*s^-1",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for minimum derivative calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes finite differences and tracks the
        minimum unscaled derivative. The save callback scales by dt_save
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
            current_index,
            customisable_variable,
        ):
            """Update the minimum first derivative with a new value.

            Parameters
            ----------
            value
                float. New value to compute derivative from.
            buffer
                device array. Storage for [prev_value, min_unscaled].
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Computes unscaled derivative as (value - buffer[0]) and updates
            buffer[1] if smaller. Uses predicated commit pattern to avoid
            warp divergence.
            """
            derivative_unscaled = value - buffer[0]
            update_flag = (derivative_unscaled < buffer[1]) and (buffer[0] != precision(0.0))
            buffer[1] = selp(update_flag, derivative_unscaled, buffer[1])
            buffer[0] = value

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
            output_array,
            summarise_every,
            customisable_variable,
        ):
            """Save scaled minimum derivative and reset buffers.

            Parameters
            ----------
            buffer
                device array. Buffer containing [prev_value, min_unscaled].
            output_array
                device array. Output location for minimum derivative.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Scales the minimum unscaled derivative by dt_save and saves to
            output_array[0], then resets buffers to sentinel values.
            """
            output_array[0] = buffer[1] / precision(dt_save)
            buffer[1] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
