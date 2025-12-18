"""
Minimum second derivative summary metric for CUDA-accelerated batch
integration.

This module implements a summary metric that tracks the minimum second
derivative (rate of change of rate of change) encountered during integration
for each variable using central finite differences.
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
class D2xdt2Min(SummaryMetric):
    """Summary metric that tracks minimum second derivative values.

    Notes
    -----
    Uses three buffer slots: buffer[0] for previous value, buffer[1] for
    previous-previous value, and buffer[2] for minimum unscaled second
    derivative. The derivative is computed using central differences and
    scaled by dt_save² in the save function.
    """

    def __init__(self, precision) -> None:
        """Initialise the D2xdt2Min summary metric."""
        super().__init__(
            name="d2xdt2_min",
            precision=precision,
            buffer_size=3,
            output_size=1,
            unit_modification="[unit]*s^-2",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for minimum second derivative.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes central finite differences and tracks
        the minimum unscaled second derivative. The save callback scales
        by dt_save² and resets the buffers.
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
            """Update the minimum second derivative with a new value.

            Parameters
            ----------
            value
                float. New value to compute second derivative from.
            buffer
                device array. Storage for [prev_value, prev_prev_value, min_unscaled].
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Computes unscaled second derivative using central difference formula
            (value - 2*buffer[0] + buffer[1]) and updates buffer[2] if smaller.
            Uses predicated commit pattern to avoid warp divergence. Guard on
            buffer[1] ensures two previous values are available.
            """
            second_derivative_unscaled = value - precision(2.0) * buffer[0] + buffer[1]
            update_flag = (second_derivative_unscaled < buffer[2]) and (buffer[1] != precision(0.0))
            buffer[2] = selp(update_flag, second_derivative_unscaled, buffer[2])
            buffer[1] = buffer[0]
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
            """Save scaled minimum second derivative and reset buffers.

            Parameters
            ----------
            buffer
                device array. Buffer containing [prev_value, prev_prev_value, min_unscaled].
            output_array
                device array. Output location for minimum second derivative.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Scales the minimum unscaled second derivative by dt_save² and saves
            to output_array[0], then resets buffers to sentinel values.
            """
            output_array[0] = buffer[2] / (precision(dt_save) * precision(dt_save))
            buffer[2] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
