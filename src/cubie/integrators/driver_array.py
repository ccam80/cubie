"""Utilities for transforming array driver samples into CUDA interpolants."""

from __future__ import annotations

import math
from typing import Callable, Union, Dict

import numpy as np
from attrs import define, field, validators
from numba import cuda, int32
from numpy.typing import NDArray

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import (
    PrecisionDtype,
    float_array_validator,
    getype_validator,
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
    drivers_dict: dict[str, Union[numpy.ndarray, float, bool]]
        convenience input for drivers. A dict containing a mapping from
        driver symbol to a vector of values; either a time vector with
        sampling times of forcing values or a scalar "dt" float which
        represents sample spacing; an optional "wrap" boolean which
        determines whether the driver should wrap past the final sample or
        hold the final value after the final sample.
    order : int
        Polynomial order for the interpolation over each segment.
    wrap : bool
        Whether the vector should repeat or provide zero values
        outside of the sampled range.
    t0 : float
        start time of driver samples; overwritten if time is supplied as an
        array (defaults to first time in that array).
    forcingArray : numpy.ndarray
        Driver sample values. Extracted from ``drivers_dict``.
    dt : numpy.ndarray
        Sampling frequency. Extracted from ``drivers_dict``.
    """

    precision: PrecisionDtype = field(validator=precision_validator)
    _drivers_dict: dict[str, Union[float, bool, FloatArray]] = field(
        validator=validators.instance_of(dict)
        )
    order: int = field()
    wrap: bool = field(default=True)

    # These are generated from _drivers_dict after init
    driver_array: FloatArray = field(
        init=False,
        validator=float_array_validator)
    dt: FloatArray = field(
        init=False,
        validator=getype_validator(float, 0)
    )
    t0: float = field(
        default=0.0,
        validator=getype_validator(float, 0)
    )

    def __attrs_post_init__(self) -> None:
        self._normalise_driver_array()
        self._validate_time_inputs()

    def _normalise_driver_array(self) -> None:
        """Promote the forcing array to the expected dimensionality.

        Raises
        ------
        ValueError
            Raised when the driver array fails dimensionality requirements or
            when the time array length does not match the driver samples.
        """
        for key, array in self._drivers_dict.items():
            if key not in ["time", "dt", "wrap"]:
                array = np.asarray(array)
                if array.ndim != 1:
                    raise ValueError(f"Forcing array {key} must be "
                                     f"one-dimensional.")
        forcing_vectors = tuple(array for key, array in self._drivers_dict.items()
                        if key not in ["time", "dt", "wrap"])
        if not all(array.shape[0] == forcing_vectors[0].shape[0]
                   for array in forcing_vectors):
            raise ValueError(
                "All forcing vectors must have the same length / be sampled "
                "on the same grid",
            )
        self.driver_array = np.column_stack(forcing_vectors)
        if self.num_samples < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct"
                " splines.",
            )

    def _validate_time_inputs(self) -> None:
        """Verify that time samples are increasing and uniformly spaced.

        Raises
        ------
        ValueError
            Raised if the time array is not strictly increasing or the
            spacing between samples is non-uniform.
        """
        has_time = "time" in self._drivers_dict
        has_dt = "dt" in self._drivers_dict

        if has_dt:
            self.dt = self._drivers_dict["dt"]
        elif has_time:
            timeArray = self._drivers_dict["time"]
            if timeArray.ndim != 1:
                raise ValueError("time_array must be one-dimensional.")
            if timeArray.shape[0] != self.driver_array.shape[0]:
                raise ValueError("time_array length must match the number of"
                                 " rows in driver_array.")
            self.t0 = timeArray[0]
            time_differences = np.diff(timeArray)
            if np.any(time_differences <= 0.0):
                raise ValueError("time_array must be strictly increasing.")
            if not np.allclose(
                time_differences,
                time_differences[0],
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError("time_array must be uniformly spaced.")
            self.dt = time_differences[0]
        else:
            raise ValueError("Either time_array or dt must be provided.")

    @property
    def drivers_dict(self):
        return self._drivers_dict

    @drivers_dict.setter
    def drivers_dict(
        self, updated_dict: dict[str, Union[float, bool, FloatArray]]
    ) -> None:
        self._drivers_dict = updated_dict
        self._normalise_driver_array()
        self._validate_time_inputs()

    @property
    def num_drivers(self) -> int:
        """Number of independent driver signals."""

        return self.driver_array.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of samples available for interpolation."""

        return self.driver_array.shape[0]

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
    """Factory emitting CUDA device functions for interpolating array-driven
    forcing terms."""

    def __init__(
        self, precision: PrecisionDtype, drivers_dict: Dict[str, FloatArray], order: int = 3, wrap: bool = False,
    ) -> None:
        super().__init__()
        config = DriverArrayConfig(
            precision=precision,
            drivers_dict=drivers_dict,
            order=order,
            wrap=wrap,
        )
        self.setup_compile_settings(config)

    @property
    def num_drivers(self) -> int:
        """Return the number of driver signals."""

        return self.compile_settings.num_drivers

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""

        return self.compile_settings.num_samples

    @property
    def driver_array(self) -> FloatArray:
        """Return the normalised driver array."""

        return self.compile_settings.driver_array

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

    @property
    def precision(self) -> PrecisionDtype:
        """Return the numerical precision used for the run."""
        return self.compile_settings.precision

    @property
    def t0(self) -> float:
        """Return the start time of the driver samples."""
        return self.compile_settings.t0

    @property
    def dt(self) -> float:
        """Return the sample spacing."""
        return self.compile_settings.dt

    def _compute_coefficients(self) -> FloatArray:
        """Fit polynomial splines across all segments for all drivers

        Returns
        -------
        numpy.ndarray
            Segment-major coefficient array of shape ``(num_segments,
            num_drivers, order + 1)``.
        """
        precision=self.precision
        num_segments = self.num_segments
        num_drivers = self.num_drivers
        order = self.order
        drivers = self.driver_array
        dt = self.dt
        times = (np.arange(num_segments + 1, dtype=precision) * dt +
                 self.t0)

        # repeat index at sample [n - order - 1] for the last [order] samples
        indices_start = np.minimum(
            np.arange(num_segments, dtype=np.int64),
            self.num_samples - order - 1,
        )
        offsets = np.arange(order + 1, dtype=np.int64)
        window_indices = indices_start[:, np.newaxis] + offsets[np.newaxis, :]

        window_times = times[window_indices]
        base_times = times[:num_segments][:, np.newaxis]
        normalised_times = (window_times - base_times) / dt

        powers = np.arange(order + 1, dtype=precision)
        vandermonde = normalised_times[..., np.newaxis] ** powers

        window_values = drivers[window_indices, :]
        coefficients = np.linalg.solve(vandermonde, window_values)
        coefficients = np.transpose(coefficients, (0, 2, 1))
        return np.ascontiguousarray(coefficients)

    def build(self) -> DriverArrayCache:
        """Compile device helpers and return them alongside host coefficients.

        Returns
        -------
        DriverArrayCache
            Cache instance bundling device functions and host coefficients.
        """

        order = self.order
        num_drivers = self.num_drivers
        resolution = self.dt
        inv_resolution = 1.0 / resolution
        start_time = self.t0
        num_segments = self.num_segments
        num_segments_i32 = int32(num_segments)
        wrap = self.wrap
        coeffs_host = self._compute_coefficients()

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

    @property
    def driver_function(self) -> Callable:
        """Return the device function for evaluating all drivers."""
        return self.get_cached_output("device_function")

    @property
    def coefficients(self) -> FloatArray:
        """Return the host-side coefficients array."""
        return self.get_cached_output("coefficients")