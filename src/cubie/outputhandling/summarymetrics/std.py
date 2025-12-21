"""
Standard deviation summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that calculates the standard deviation
of values encountered during integration for each variable.
"""

from numba import cuda
from cubie.cuda_simsafe import compile_kwargs
from math import sqrt

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Std(SummaryMetric):
    """Summary metric that calculates the standard deviation of a variable.

    Notes
    -----
    The metric uses three buffer slots: one for the shift value (first sample),
    one for the sum of shifted values, and one for the sum of squares of
    shifted values. The standard deviation is calculated using a numerically
    stable shifted-data algorithm to avoid catastrophic cancellation.
    """

    def __init__(self, precision) -> None:
        """Initialise the Std summary metric with fixed buffer sizes."""
        super().__init__(
            name="std",
            precision=precision,
            buffer_size=3,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for standard deviation calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates both sum and sum of squares while the
        save callback computes the standard deviation and clears the buffer.
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
            """Update the running sum and sum of squares with shifted values.

            Parameters
            ----------
            value
                float. New value to add to the running statistics.
            buffer
                device array. Full buffer containing metric working storage.
            offset
                int. Offset to this metric's storage within the buffer.
            current_index
                int. Current integration step index within the summary period.
            customisable_variable
                int. Metric parameter placeholder (unused for std).

            Notes
            -----
            On first sample (current_index == 0), stores the value as shift
            and resets accumulators. For all samples including the first,
            computes shifted_value = value - shift and adds it to
            buffer[offset + 1] (sum) and shifted_value^2 to buffer[offset + 2]
            (sum of squares). This shifting improves numerical stability.
            """
            if current_index == 0:
                buffer[offset + 0] = value  # Store shift value
                buffer[offset + 1] = precision(0.0)    # Reset sum
                buffer[offset + 2] = precision(0.0)    # Reset sum of squares

            shifted_value = value - buffer[offset + 0]
            buffer[offset + 1] += shifted_value
            buffer[offset + 2] += shifted_value * shifted_value

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
            """Calculate the standard deviation from shifted running statistics.

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
                int. Number of steps contributing to each summary window.
            customisable_variable
                int. Metric parameter placeholder (unused for std).

            Notes
            -----
            Calculates variance using the shifted data algorithm:
            variance = (sum_sq_shifted/n) - (sum_shifted/n)^2
            Then computes std = sqrt(variance) and saves to
            output_array[output_offset + 0].
            Resets buffer for the next summary period.
            """
            mean_shifted = buffer[buffer_offset + 1] / summarise_every
            mean_of_squares_shifted = buffer[buffer_offset + 2] / summarise_every
            variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)
            output_array[output_offset + 0] = sqrt(variance)
            mean = buffer[buffer_offset + 0] + mean_shifted
            buffer[buffer_offset + 0] = mean
            buffer[buffer_offset + 1] = precision(0.0)
            buffer[buffer_offset + 2] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
