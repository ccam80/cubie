"""
Peak detection summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that detects and records the timing
of local maxima (peaks) in variable values during integration.
"""

from numba import cuda, int32
from cubie.cuda_simsafe import compile_kwargs, selp

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Peaks(SummaryMetric):
    """Summary metric that records the indices of detected peaks.

    Notes
    -----
    The buffer stores the two previous values, a peak counter, and slots for
    the recorded peak indices. The algorithm assumes ``0.0`` does not occur in
    valid data so it can serve as an initial sentinel.
    """

    def __init__(self, precision) -> None:
        """Initialise the Peaks summary metric with parameterised sizes."""
        super().__init__(
            name="peaks",
            precision=precision,
            buffer_size=lambda n: 3 + n,
            output_size=lambda n: n,
            unit_modification="s",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for peak detection.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback compares the current value against stored history
        to identify peaks, while the save callback copies stored indices and
        resets the buffer for the next period.
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
            """Update peak detection with a new value.

            Parameters
            ----------
            value
                float. New value to analyse for peak detection.
            buffer
                device array. Full buffer containing metric working storage.
            offset
                int. Offset to this metric's storage within the buffer.
                Layout at offset: ``[prev, prev_prev, counter, times...]``.
            current_index
                int. Current integration step index, used to record peaks.
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Detects peaks when the prior value exceeds both the current and
            second-prior values. Peak indices are stored from
            ``buffer[offset + 3]`` onward. Uses predicated commit pattern
            for all buffer writes to avoid warp divergence.
            """
            npeaks = customisable_variable
            prev = buffer[offset + 0]
            prev_prev = buffer[offset + 1]
            peak_counter = int32(buffer[offset + 2])

            is_valid = (
                (current_index >= 2)
                and (peak_counter < npeaks)
                and (prev_prev != precision(0.0))
            )
            is_peak = prev > value and prev_prev < prev
            should_record = is_valid and is_peak

            # Clamp index to valid range for predicated access
            safe_idx = peak_counter if peak_counter < npeaks else npeaks - 1

            # Predicated commit for peak recording
            new_counter = precision(int32(buffer[offset + 2]) + 1)
            buffer[offset + 3 + safe_idx] = selp(
                should_record,
                precision(current_index - 1),
                buffer[offset + 3 + safe_idx],
            )
            buffer[offset + 2] = selp(
                should_record, new_counter, buffer[offset + 2]
            )
            buffer[offset + 0] = value
            buffer[offset + 1] = prev

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
            """Save detected peak time indices and reset the buffer.

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
                int. Number of steps between saves (unused for peak detection).
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Copies peak indices from ``buffer[buffer_offset + 3:]`` to the
            output array then clears the storage for the next summary interval.
            """
            n_peaks = int32(customisable_variable)
            for p in range(n_peaks):
                output_array[output_offset + p] = buffer[buffer_offset + 3 + p]
                buffer[buffer_offset + 3 + p] = precision(0.0)
            buffer[buffer_offset + 2] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
