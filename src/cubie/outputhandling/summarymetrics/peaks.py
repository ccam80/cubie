"""
Peak detection summary metric for CUDA-accelerated batch integration.

This module implements a summary metric that detects and records the timing
of local maxima (peaks) in variable values during integration.
"""

from numba import cuda

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

    def __init__(self) -> None:
        """Initialise the Peaks summary metric with parameterised sizes."""
        super().__init__(
            name="peaks",
            buffer_size=lambda n: 3 + n,
            output_size=lambda n: n,
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
            """Update peak detection with a new value.

            Parameters
            ----------
            value
                float. New value to analyse for peak detection.
            buffer
                device array. Layout ``[prev, prev_prev, counter, times...]``.
            current_index
                int. Current integration step index, used to record peaks.
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Detects peaks when the prior value exceeds both the current and
            second-prior values. Peak indices are stored from ``buffer[3]``
            onward.
            """
            npeaks = customisable_variable
            prev = buffer[0]
            prev_prev = buffer[1]
            peak_counter = int(buffer[2])

            if (
                (current_index >= 2)
                and (peak_counter < npeaks)
                and (prev_prev != 0.0)
            ):
                if prev > value and prev_prev < prev:
                    # Bingo
                    buffer[3 + peak_counter] = (current_index - 1)
                    buffer[2] = float(int(buffer[2]) + 1)
            buffer[0] = value  # Update previous value
            buffer[1] = prev  # Update previous previous value

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
            """Save detected peak time indices and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Buffer containing detected peak time indices.
            output_array
                device array. Output array for saving peak time indices.
            summarise_every
                int. Number of steps between saves (unused for peak detection).
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Copies peak indices from ``buffer[3:]`` to the output array then
            clears the storage for the next summary interval.
            """
            n_peaks = customisable_variable
            for p in range(n_peaks):
                output_array[p] = buffer[3 + p]
                buffer[3 + p] = 0.0
            buffer[2] = 0.0

        # no cover: end
        return MetricFuncCache(update = update, save = save)
