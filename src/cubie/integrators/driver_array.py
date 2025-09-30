"""Utilities for transforming and interpreting forcing terms provided as
sampled arrays."""

import math

import numpy as np
from numba import cuda, int32
from numpy.typing import NDArray

from attrs import define, field

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import (
    precision_validator,
    PrecisionDtype,
    float_array_validator,
)

FloatArray = NDArray[np.floating]


@define
class DriverArrayConfig:
    """Configuration for an array-derived driver function object"""
    precision: PrecisionDtype = field(
        validator=precision_validator
    )
    forcingArray: FloatArray = field(
            validator=float_array_validator
    )
    timeArray: FloatArray = field(
            validator=float_array_validator
    )
    order: int = field()
    wrap: bool = field(default=False)

    def __attrs_post_init__(self):
        self.check_shapes()
        self.check_time_array()

    def check_shapes(self):
        if self.forcingArray.ndim == 1:
            # normalize 1-D input to a 2-D column vector for consistent
            # downstream handling (columns == drivers)
            self.forcingArray = self.forcingArray[:, np.newaxis]
        elif self.forcingArray.ndim != 2:
            raise ValueError(
                    "driver_array must be one- or two-dimensional with columns "
                    "per driver.",
            )
        if self.timeArray.ndim != 1:
            raise ValueError("time_array must be one-dimensional.")

        if self.timeArray.size != self.num_samples:
            raise ValueError(
                "time_array length must match the number of rows in "
                "driver_array.",
            )

        if self.num_samples < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct "
                "splines.",
            )

    def check_time_array(self):
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
    def num_drivers(self):
        return self.forcingArray.shape[1]

    @property
    def num_samples(self):
        return self.forcingArray.shape[0]

    @property
    def resolution(self):
        return np.diff(self.timeArray[0:2])[0]

    @property
    def num_segments(self):
        return self.num_samples - 1


class DriverArray(CUDAFactory):
    """CUDAFactory which creates an interpolator for array-derived forcing terms.

    Simplified version:
      - Eliminated coefficient layout switching and compile_settings access from
      methods (now via properties only).
    """
    def __init__(self,
                 driver_array: FloatArray,
                 time_array: FloatArray,
                 order: int = 3,
                 wrap: bool = False):
        super().__init__()
        config = DriverArrayConfig(
            precision=driver_array.dtype,
            forcingArray=driver_array,
            timeArray=time_array,
            order=order,
            wrap=wrap,
        )
        self.setup_compile_settings(config)
        self._coeff_cache_segment_major = None  # shape (segments, drivers, order+1)
        self._vandermonde_inverse_cache = {}

    # --- Properties exposing compile settings (avoid direct attribute access elsewhere) ---
    @property
    def num_drivers(self):
        return self.compile_settings.num_drivers

    @property
    def num_samples(self):
        return self.compile_settings.num_samples

    @property
    def resolution(self):
        return self.compile_settings.resolution

    @property
    def driver_array(self):
        return self.compile_settings.forcingArray

    @property
    def time_array(self):
        return self.compile_settings.timeArray

    @property
    def order(self):
        return self.compile_settings.order

    @property
    def wrap(self):
        return self.compile_settings.wrap

    @property
    def num_segments(self):
        return self.num_samples - 1

    # --- Coefficient generation & caching ---
    def _build_segment_major_coeffs(self):
        """Build and cache coefficients in segment-major layout.

        Output shape: (num_segments, num_drivers, order+1) with degree index increasing.
        """
        if self._coeff_cache_segment_major is not None:
            return self._coeff_cache_segment_major

        n_segments = self.num_segments
        n_drivers = self.num_drivers
        order = self.order
        res = self.resolution
        times = self.time_array
        drivers = self.driver_array  # (samples, drivers)

        coeffs = np.empty((n_segments, n_drivers, order + 1), dtype=drivers.dtype)

        # Distinct window patterns near the tail may differ; cache by normalised time tuple.
        inv_cache = self._vandermonde_inverse_cache

        for seg in range(n_segments):
            start_index = min(seg, self.num_samples - order - 1)
            window_slice = slice(start_index, start_index + order + 1)
            window_times = times[window_slice]
            norm_times = (window_times - times[seg]) / res  # shape (order+1,)
            key = tuple(norm_times.tolist())
            A_inv = inv_cache.get(key)
            if A_inv is None:
                # Vandermonde with increasing powers
                V = np.vander(norm_times, N=order + 1, increasing=True)
                A_inv = np.linalg.inv(V)
                inv_cache[key] = A_inv
            window_values = drivers[window_slice, :]  # shape (order+1, drivers)
            # coeffs_deg (order+1, drivers) = A_inv @ window_values
            coeffs_deg = A_inv @ window_values
            # Store per driver (degree increasing)
            coeffs[seg, :, :] = coeffs_deg.T  # (drivers, order+1)

        self._coeff_cache_segment_major = coeffs
        return coeffs

    def invalidate_coefficients(self):
        self._coeff_cache_segment_major = None
        # Keep Vandermonde inverses (they depend only on order & patterns)

    def build(self):
        """Return (partition_fn, evaluate_all_fn, evaluate_driver_fn, coeffs_host).
        Coefficients host array shape: (num_segments, num_drivers, order+1)
        """
        order = self.order
        num_drivers = self.num_drivers
        resolution = self.resolution
        inv_resolution = 1.0 / resolution
        start_time = self.time_array[0]
        num_segments = self.num_segments
        wrap = self.wrap
        coeffs_host = self._build_segment_major_coeffs()

        if wrap:
            @cuda.jit(device=True, inline=True)
            def partition(time: float) -> int:  # type: ignore
                scaled = (time - start_time) * inv_resolution
                idx = int32(math.floor(scaled))
                return int32(idx % num_segments)
        else:
            @cuda.jit(device=True, inline=True)
            def partition(time: float) -> int:  # type: ignore
                scaled = (time - start_time) * inv_resolution
                idx = int32(math.floor(scaled))
                idx = int32(0) if idx < 0 else idx
                return idx if idx < num_segments else int32(num_segments - 1)

        @cuda.jit(device=True, inline=True)
        def evaluate_all(time,
                         coefficients,
                         out):
            seg = partition(time)
            base_time = start_time + resolution * float(seg)
            tau = (time - base_time) * inv_resolution
            for d in range(num_drivers):
                acc = 0.0
                for k in range(order, -1, -1):
                    acc = acc * tau + coefficients[seg, d, k]
                out[d] = acc


        return partition, evaluate_all, coeffs_host

    # --- Public helper returning coefficients (backwards-compatible shape) ---
    def to_spline_constants(self):  # retains prior public name
        """Return spline coefficients.

        Single driver -> (num_segments, order+1)
        Multiple drivers -> (num_drivers, num_segments, order+1)
        (Cached computation reused.)
        """
        coeffs = self._build_segment_major_coeffs()  # (segments, drivers, order+1)
        if self.num_drivers == 1:
            return coeffs[:, 0, :]
        # Convert to original multi-driver orientation (drivers, segments, degree)
        return np.transpose(coeffs, (1, 0, 2))
