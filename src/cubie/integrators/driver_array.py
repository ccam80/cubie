"""Utilities for transforming array driver samples into CUDA interpolants."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from attrs import define, field
from numba import cuda, int32
from numpy.typing import NDArray

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import (
    PrecisionDtype,
    float_array_validator,
    precision_validator,
)

FloatArray = NDArray[np.floating]


@define
class DriverArrayConfig:
    """Configuration describing a driver-array interpolation problem.

    Attributes
    ----------
    precision : numpy.dtype
        Precision to be used when generating polynomial coefficients.
    forcingArray : numpy.ndarray
        Driver sample values. A one-dimensional input is promoted to a
        two-dimensional column vector.
    timeArray : numpy.ndarray
        Sample times corresponding to the rows in ``forcingArray``.
    order : int
        Polynomial order for the interpolation over each segment.
    wrap : bool
        Whether the partition logic should wrap past the final segment.
    """

    precision: PrecisionDtype = field(validator=precision_validator)
    forcingArray: FloatArray = field(validator=float_array_validator)
    timeArray: FloatArray = field(validator=float_array_validator)
    order: int = field()
    wrap: bool = field(default=False)

    def __attrs_post_init__(self) -> None:
        self._normalise_driver_array()
        self._validate_time_array()

    def _normalise_driver_array(self) -> None:
        """Promote the forcing array to the expected dimensionality.

        Raises
        ------
        ValueError
            Raised when the driver array fails dimensionality requirements or
            when the time array length does not match the driver samples.
        """

        if self.forcingArray.ndim == 1:
            self.forcingArray = self.forcingArray[:, np.newaxis]
        elif self.forcingArray.ndim != 2:
            raise ValueError(
                "driver_array must be one- or two-dimensional with"
                " columns per driver.",
            )

        if self.timeArray.ndim != 1:
            raise ValueError("time_array must be one-dimensional.")

        if self.timeArray.size != self.num_samples:
            raise ValueError(
                "time_array length must match the number of rows in"
                " driver_array.",
            )

        if self.num_samples < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct"
                " splines.",
            )

    def _validate_time_array(self) -> None:
        """Verify that time samples are increasing and uniformly spaced.

        Raises
        ------
        ValueError
            Raised if the time array is not strictly increasing or the
            spacing between samples is non-uniform.
        """

        time_differences = np.diff(self.timeArray)
        if np.any(time_differences <= 0.0):
            raise ValueError("time_array must be strictly increasing.")
        if not np.allclose(
            time_differences,
            time_differences[0],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("time_array must be uniformly spaced.")

    @property
    def num_drivers(self) -> int:
        """Number of independent driver signals."""

        return self.forcingArray.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of samples available for interpolation."""

        return self.forcingArray.shape[0]

    @property
    def resolution(self) -> float:
        """Spacing between adjacent samples."""

        return float(np.diff(self.timeArray[0:2])[0])

    @property
    def num_segments(self) -> int:
        """Number of polynomial segments generated from the samples."""

        return self.num_samples - 1


@define
class DriverArrayCache:
    """Container holding CUDA-ready interpolation artefacts.

    Attributes
    ----------
    device_function : callable
        CUDA device function evaluating all driver polynomials at a time.
    coefficients : numpy.ndarray
        Host-side coefficient array storing polynomial coefficients in
        segment-major order.
    """

    device_function: Callable[..., None]
    coefficients: FloatArray


class DriverArray(CUDAFactory):
    """Factory emitting CUDA device functions for array-driven forcings."""

    def __init__(
        self,
        driver_array: FloatArray,
        time_array: FloatArray,
        order: int = 3,
        wrap: bool = False,
    ) -> None:
        super().__init__()
        config = DriverArrayConfig(
            precision=driver_array.dtype,
            forcingArray=driver_array,
            timeArray=time_array,
            order=order,
            wrap=wrap,
        )
        self.setup_compile_settings(config)
        self._coefficients_segment_major: FloatArray | None = None

    @property
    def num_drivers(self) -> int:
        """Return the number of driver signals."""

        return self.compile_settings.num_drivers

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""

        return self.compile_settings.num_samples

    @property
    def resolution(self) -> float:
        """Return the sample spacing."""

        return self.compile_settings.resolution

    @property
    def driver_array(self) -> FloatArray:
        """Return the normalised driver array."""

        return self.compile_settings.forcingArray

    @property
    def time_array(self) -> FloatArray:
        """Return the time samples."""

        return self.compile_settings.timeArray

    @property
    def order(self) -> int:
        """Return the interpolating polynomial order."""

        return self.compile_settings.order

    @property
    def wrap(self) -> bool:
        """Return whether the driver should wrap past the final sample."""

        return self.compile_settings.wrap

    @property
    def num_segments(self) -> int:
        """Return the number of polynomial segments."""

        return self.num_samples - 1

    def build_coefficients(self) -> FloatArray:
        """Return segment-major polynomial coefficients.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_segments, num_drivers, order + 1)`` holding
            coefficients with degree indices increasing along the final axis.
        """

        if self._coefficients_segment_major is None:
            self._coefficients_segment_major = self._compute_segment_major()
        return self._coefficients_segment_major

    def _compute_segment_major(self) -> FloatArray:
        """Vectorise polynomial fitting across all segments and drivers.

        Returns
        -------
        numpy.ndarray
            Segment-major coefficient array of shape ``(num_segments,
            num_drivers, order + 1)``.
        """

        num_segments = self.num_segments
        num_drivers = self.num_drivers
        order = self.order
        drivers = self.driver_array
        times = self.time_array
        resolution = self.resolution

        indices_start = np.minimum(
            np.arange(num_segments, dtype=np.int64),
            self.num_samples - order - 1,
        )
        offsets = np.arange(order + 1, dtype=np.int64)
        window_indices = indices_start[:, np.newaxis] + offsets[np.newaxis, :]

        window_times = times[window_indices]
        base_times = times[:num_segments][:, np.newaxis]
        normalised_times = (window_times - base_times) / resolution

        powers = np.arange(order + 1, dtype=drivers.dtype)
        vandermonde = normalised_times[..., np.newaxis] ** powers

        window_values = drivers[window_indices, :]
        coefficients = np.linalg.solve(vandermonde, window_values)
        coefficients = np.transpose(coefficients, (0, 2, 1))
        return np.ascontiguousarray(coefficients)

    def invalidate_coefficients(self) -> None:
        """Invalidate cached polynomial coefficients."""

        self._coefficients_segment_major = None

    def build(self) -> DriverArrayCache:
        """Compile device helpers and return them alongside host coefficients.

        Returns
        -------
        DriverArrayCache
            Cache instance bundling device functions and host coefficients.
        """

        order = self.order
        num_drivers = self.num_drivers
        resolution = self.resolution
        inv_resolution = 1.0 / resolution
        start_time = float(self.time_array[0])
        num_segments = self.num_segments
        num_segments_i32 = int32(num_segments)
        wrap = self.wrap
        coeffs_host = self.build_coefficients()

        @cuda.jit(device=True, inline=True)
        def evaluate_all(
            time: float, coefficients: np.ndarray, out: np.ndarray
        ) -> None:
            """Evaluate all driver polynomials at ``time`` on the device.

            Parameters
            ----------
            time : float
                Query time for evaluation.
            coefficients : numpy.ndarray
                Segment-major coefficients with trailing polynomial degrees.
            out : numpy.ndarray
                Output array to populate with evaluated driver values.
            """

            scaled = (time - start_time) * inv_resolution
            idx = int32(math.floor(scaled))
            if wrap:
                seg = int32(idx % num_segments_i32)
                if seg < 0:
                    seg = int32(seg + num_segments_i32)
            else:
                if idx < 0:
                    seg = int32(0)
                elif idx >= num_segments_i32:
                    seg = int32(num_segments_i32 - 1)
                else:
                    seg = idx
            base_time = start_time + resolution * float(seg)
            tau = (time - base_time) * inv_resolution
            for driver_index in range(num_drivers):
                acc = 0.0
                for k in range(order, -1, -1):
                    acc = acc * tau + coefficients[seg, driver_index, k]
                out[driver_index] = acc

        return DriverArrayCache(
            device_function=evaluate_all,
            coefficients=coeffs_host,
        )
