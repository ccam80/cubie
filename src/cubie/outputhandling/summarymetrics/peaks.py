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
    """
    Summary metric to detect and record peak locations in a variable.

    This metric identifies local maxima by tracking when a value is greater
    than both its predecessor and successor values. It records the time
    indices where peaks occur, up to a configurable maximum number.

    Notes
    -----
    The metric uses a parameterized buffer size that depends on the maximum
    number of peaks to detect. The buffer stores: previous value,
    previous-previous value, peak counter, and the time indices of detected
    peaks.

    The peak detection algorithm requires at least 2 previous values and
    assumes no natural 0.0 values in the data (uses 0.0 as initialization
    marker).
    """

    def __init__(self):
        """
        Initialize the Peaks summary metric.

        Creates CUDA device functions for peak detection and result saving,
        and configures the metric with parameterized buffer and output sizes
        based on the maximum number of peaks to detect.
        """
        super().__init__(
            name="peaks",
            buffer_size=lambda n: 3 + n,
            output_size=lambda n: n,
        )

    def build(self):
        """
        Generate CUDA device functions for peak detection.

        Creates optimized CUDA device functions with fixed signatures for
        detecting peaks during integration and saving peak time indices.

        Returns
        -------
        MetricFuncCache[update: callable, save: callable]
            Cache object containing update and save functions for CUDA
            execution. Both functions must be compiled with @cuda.jit
            decorators.

        Notes
        -----
        The generated functions have the following signatures:

        update(value, buffer, current_index, customisable_variable):
            Detects peaks by comparing current value with previous values.

        save(buffer, output_array, summarise_every, customisable_variable):
            Saves detected peak time indices to output and resets buffer.
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
            """
            Update peak detection with new value.

            Parameters
            ----------
            value : float
                New value to analyze for peak detection.
            buffer : array-like
                Buffer containing: [prev_value, prev_prev_value, peak_count,
                peak_times...].
            current_index : int
                Current integration step index, used for recording peak times.
            customisable_variable : int
                Maximum number of peaks to detect (n_peaks parameter).

            Notes
            -----
            Detects peaks by checking if previous value is greater than both
            the current value and the value before that. Records the time index
            of detected peaks in buffer positions 3 onwards. Requires at least
            2 previous values and assumes no natural 0.0 values in the data.
            Buffer layout: [prev, prev_prev, peak_counter, peak_times...]
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
                    buffer[3 + peak_counter] = float(current_index - 1)
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
            """
            Save detected peak time indices and reset buffer.

            Parameters
            ----------
            buffer : array-like
                Buffer containing detected peak time indices.
            output_array : array-like
                Output array for saving peak time indices.
            summarise_every : int
                Number of steps between saves (unused for peak detection).
            customisable_variable : int
                Maximum number of peaks to detect (n_peaks parameter).

            Notes
            -----
            Copies peak time indices from buffer positions 3 onwards to the
            output array, then resets the peak storage and counter for the
            next summary period. Output size equals the maximum number of
            peaks that can be detected.
            """
            n_peaks = customisable_variable
            for p in range(n_peaks):
                output_array[p] = buffer[3 + p]
                buffer[3 + p] = 0.0
            buffer[2] = 0.0

        # no cover: end
        return MetricFuncCache(update = update, save = save)
