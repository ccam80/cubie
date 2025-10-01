"""Utilities for transforming array driver samples into CUDA interpolants."""

import math
from typing import Callable, Dict, Optional, Set, TYPE_CHECKING, Union

import numpy as np
from attrs import define, field, validators
from numba import cuda, int32
from numpy.typing import NDArray

from cubie.cuda_simsafe import selp

if TYPE_CHECKING:
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE

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
        represents sample spacing; optional ``"wrap"`` and
        ``"boundary_condition"`` flags to configure behaviour around the
        final sample; an optional ``"order"`` integer to override the spline
        order.
    order : int
        Polynomial order for the interpolation over each segment.
    wrap : bool
        Whether the vector should repeat or provide zero values
        outside of the sampled range.
    boundary_condition : {"natural", "periodic"}, optional
        Boundary condition for the spline interpolation. When omitted the
        solver falls back to the original local polynomial interpolation.
    t0 : float
        start time of driver samples; overwritten if time is supplied as an
        array (defaults to first time in that array).
    driver_array : numpy.ndarray
        Driver sample values. Extracted from ``drivers_dict``.
    dt : numpy.ndarray
        Sampling frequency. Extracted from ``drivers_dict``.
    """

    precision: PrecisionDtype = field(validator=precision_validator)
    _drivers_dict: dict[str, Union[float, bool, FloatArray]] = field(
        validator=validators.instance_of(dict),
    )
    order: int = field(default=3)
    wrap: bool = field(default=True)
    boundary_condition: Optional[str] = field(
        default=None,
        validator=validators.optional(
            validators.in_({"natural", "periodic"})
        ),
    )

    # These are generated from _drivers_dict after init
    driver_array: FloatArray = field(
        init=False,
        validator=float_array_validator,
    )
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
        self._update_order_wrap()

    def _update_order_wrap(self) -> None:
        if "order" in self._drivers_dict:
            self.order = self._drivers_dict["order"]
        if "wrap" in self._drivers_dict:
            self.wrap = self._drivers_dict["wrap"]
        if "boundary_condition" in self._drivers_dict:
            self.boundary_condition = self._drivers_dict["boundary_condition"]

    def _normalise_driver_array(self) -> None:
        """Promote the forcing array to the expected dimensionality.

        Raises
        ------
        ValueError
            Raised when the driver array fails dimensionality requirements or
            when the time array length does not match the driver samples.
        """
        special_keys = ["time", "dt", "wrap", "order", "boundary_condition"]
        for key, array in self._drivers_dict.items():
            if key not in special_keys:
                array = np.asarray(array)
                if array.ndim != 1:
                    raise ValueError(f"Forcing array {key} must be "
                                     f"one-dimensional.")
        forcing_vectors = tuple(
            array
            for key, array in self._drivers_dict.items()
            if key not in special_keys
        )
        if not all(
            array.shape[0] == forcing_vectors[0].shape[0]
            for array in forcing_vectors
        ):
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
        self._update_order_wrap()

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
        self,
        precision: PrecisionDtype,
        drivers_dict: Dict[str, FloatArray],
        order: int = 3,
        wrap: bool = False,
        boundary_condition: Optional[str] = None,
    ) -> None:
        super().__init__()
        config = DriverArrayConfig(
            precision=precision,
            drivers_dict=drivers_dict,
            order=order,
            wrap=wrap,
            boundary_condition=boundary_condition,
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
    def boundary_condition(self) -> Optional[str]:
        """Return the spline boundary condition to enforce, if any."""

        return self.compile_settings.boundary_condition

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

    @staticmethod
    def check_against_system(self,
                             drivers_dict: Dict[str, Union[float, bool, FloatArray]],
                             system: 'SymbolicODE'):
        driver_keys = [
            key
            for key in drivers_dict
            if key not in [
                "time",
                "dt",
                "wrap",
                "order",
                "boundary_condition",
            ]
        ]
        system_keys = set(system.indices.drivers.symbol_map.keys())
        if len(driver_keys) != system.num_drivers:
            raise ValueError(f"Number of drivers in drivers_dict "
                             f"({len(driver_keys)}) does not match number of "
                             f"drivers in system ({system.num_drivers}).")
        if set(driver_keys) != system_keys:
            raise ValueError(f"Driver symbols in drivers_dict ("
                             f"{set(driver_keys)}) do not match driver symbols "
                             f"in system ({system_keys}).")

    def _compute_coefficients(self) -> FloatArray:
        """Compute coefficients via local polynomial interpolation.

        Returns
        -------
        numpy.ndarray
            Segment-major coefficient array of shape ``(num_segments,
            num_drivers, order + 1)``.
        """
        boundary_condition = self.boundary_condition
        if boundary_condition is None:
            precision = self.precision
            num_segments = self.num_segments
            num_drivers = self.num_drivers
            order = self.order
            drivers = self.driver_array
            dt = self.dt
            times = (np.arange(num_segments + 1, dtype=precision) * dt +
                     self.t0)

            window_size = order + 1
            left_window = order // 2
            base_indices = np.arange(num_segments, dtype=np.int64) - left_window
            max_start = self.num_samples - window_size
            indices_start = np.clip(base_indices, 0, max_start)
            offsets = np.arange(window_size, dtype=np.int64)
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

        return self._compute_coefficients_with_boundary(boundary_condition)

    def _compute_coefficients_with_boundary(
        self,
        boundary_condition: str,
    ) -> FloatArray:
        """Return spline coefficients respecting the requested boundary.

        Parameters
        ----------
        boundary_condition : str
            Requested boundary condition. Accepted values are ``"natural"``
            and ``"periodic"``.

        Returns
        -------
        numpy.ndarray
            Coefficient array with shape ``(num_segments, num_drivers,
            order + 1)`` ordered for Horner evaluation.

        Raises
        ------
        ValueError
            Raised when periodic constraints are incompatible with the driver
            configuration or when an unknown boundary condition is supplied.
        """

        if boundary_condition not in {"natural", "periodic"}:
            raise ValueError(
                f"Unsupported boundary condition: {boundary_condition}."
            )

        precision = self.precision
        drivers = self.driver_array.astype(precision, copy=False)
        num_samples = self.num_samples
        num_segments = self.num_segments
        num_drivers = self.num_drivers
        order = self.order

        if boundary_condition == "periodic":
            if not self.wrap:
                raise ValueError(
                    "Periodic boundary conditions require wrap=True so that "
                    "the driver repeats after the final segment."
                )
            if not np.allclose(drivers[0], drivers[-1]):
                raise ValueError(
                    "Periodic boundary conditions require the first and "
                    "last samples to match."
                )

        num_coeffs = num_segments * (order + 1)
        matrix = np.zeros((num_coeffs, num_coeffs), dtype=precision)
        rhs = np.zeros((num_coeffs, num_drivers), dtype=precision)
        row_index = 0

        def coeff_index(segment: int, power: int) -> int:
            """Return the flattened coefficient index for ``segment``."""

            return segment * (order + 1) + power

        falling = np.zeros((order + 1, order + 1), dtype=precision)
        falling[:, 0] = precision(1.0)
        for derivative in range(1, order + 1):
            for power in range(derivative, order + 1):
                falling[power, derivative] = (
                    falling[power, derivative - 1]
                    * precision(power - (derivative - 1))
                )

        # Function value constraints at the left edge of each segment.
        for segment in range(num_segments):
            matrix[row_index, coeff_index(segment, 0)] = precision(1.0)
            rhs[row_index] = drivers[segment]
            row_index += 1

        # Function value constraints at the right edge of each segment.
        for segment in range(num_segments):
            base = coeff_index(segment, 0)
            for power in range(order + 1):
                matrix[row_index, base + power] = precision(1.0)
            rhs[row_index] = drivers[segment + 1]
            row_index += 1

        # Continuity of derivatives across interior knots.
        for segment in range(num_segments - 1):
            for derivative in range(1, order):
                base = coeff_index(segment, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base + power] = falling[power, derivative]
                next_index = coeff_index(segment + 1, derivative)
                matrix[row_index, next_index] -= falling[derivative, derivative]
                row_index += 1

        if boundary_condition == "natural":
            remaining = order - 1
            derivative = 2
            while remaining > 0 and derivative <= order:
                base_start = coeff_index(0, 0)
                matrix[row_index, base_start + derivative] = (
                    falling[derivative, derivative]
                )
                row_index += 1
                remaining -= 1
                if remaining == 0:
                    break
                base_end = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_end + power] = (
                        falling[power, derivative]
                    )
                row_index += 1
                remaining -= 1
                derivative += 1
        elif boundary_condition == "periodic":
            for derivative in range(1, order):
                base_last = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_last + power] = (
                        falling[power, derivative]
                    )
                base_first = coeff_index(0, derivative)
                matrix[row_index, base_first] -= falling[derivative, derivative]
                row_index += 1

        if row_index != num_coeffs:
            raise ValueError(
                "Failed to assemble a square spline system; "
                "please verify boundary condition handling."
            )

        solution = np.linalg.solve(matrix, rhs)
        coefficients = solution.reshape(num_segments, order + 1, num_drivers)
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
        zero_value = self.precision(0.0)

        @cuda.jit(device=True, inline=True)
        def evaluate_all(
            time,
            coefficients,
            out
        ) -> None:
            """Evaluate all driver polynomials at ``time`` on the device.

            Parameters
            ----------
            time : float
                Query time for evaluation.
            coefficients : device array
                Segment-major coefficients with trailing polynomial degrees.
            out : device array
                Output array to populate with evaluated driver values.
            """

            scaled = (time - start_time) * inv_resolution
            scaled_floor = math.floor(scaled)
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments_i32)
                tau = scaled - scaled_floor
                in_range = True
            else:
                in_range = (scaled >= 0.0) and (scaled <= num_segments)
                seg = selp(idx < 0, int32(0), idx)
                seg = selp(seg >= num_segments_i32,
                           int32(num_segments_i32 - 1), seg)
                tau = scaled - float(seg)

            # Evaluate polynomials using Horner's rule; compiler will unroll the k-loop
            for driver_index in range(num_drivers):
                acc = zero_value
                for k in range(order, -1, -1):
                    acc = acc * tau + coefficients[seg, driver_index, k]
                out[driver_index] = acc if in_range else zero_value

        return DriverArrayCache(
            device_function=evaluate_all,
            coefficients=coeffs_host,
         )

    def update(
        self,
        updates_dict: Optional[Dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Apply configuration updates and invalidate caches when needed.

        Parameters
        ----------
        updates_dict
            Mapping of configuration keys to their new values.
        silent
            When ``True``, suppress warnings about inapplicable keys.
        **kwargs
            Additional configuration updates supplied inline.

        Returns
        -------
        set
            Set of configuration keys that were recognized and updated.

        Raises
        ------
        KeyError
            Raised when an unknown key is provided while ``silent`` is False.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised


        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )

        return recognised

    @property
    def driver_function(self) -> Callable:
        """Return the device function for evaluating all drivers."""
        return self.get_cached_output("device_function")

    @property
    def coefficients(self) -> FloatArray:
        """Return the host-side coefficients array."""
        return self.get_cached_output("coefficients")
