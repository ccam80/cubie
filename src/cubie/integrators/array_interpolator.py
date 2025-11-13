"""Utilities for transforming array input samples into CUDA interpolants."""

import math
from typing import Callable, Dict, Optional, Set, TYPE_CHECKING, Union, Any, \
    Tuple

import numpy as np
from attrs import define, field, validators
from numba import cuda, int32
from numpy.typing import NDArray

from cubie.cuda_simsafe import selp

if TYPE_CHECKING:
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
    precision_converter,
    precision_validator,
)

FloatArray = NDArray[np.floating]


@define
class InterpolatorCache:
    """Cached device helpers emitted by :class:`ArrayInterpolator`."""

    evaluation_function: Optional[Callable] = field(default=None)
    driver_del_t: Optional[Callable] = field(default=None)


@define
class ArrayInterpolatorConfig:
    """Configuration describing an input-array interpolation problem.

    Attributes
    ----------
    precision : numpy.dtype
        Precision to be used when generating polynomial coefficients.
    order : int
        Polynomial order for the interpolation over each segment.
    wrap : bool
        Whether the vector should repeat or provide zero values
        outside of the sampled range.
    boundary_condition : {"natural", "periodic", "clamped", "not-a-knot"},
        optional boundary condition for the spline interpolation.
        defaults to 'not-a-knot' to match Scipy's CubicSpline.
    t0 : float
        start time of input samples
    dt : numpy.ndarray
        Sampling frequency.
    num_inputs : int
        Number of separate input vectors
    num_segments : int
        Number of polynomial segments in the coefficient table. For
        clamped, non-wrapping inputs this includes two ghost segments that
        transition from and to zero-valued padding samples.
    """

    precision: PrecisionDType = field(
        converter=precision_converter,
        validator=precision_validator,
    )
    order: int = field(
        default=3,
        validator=gttype_validator(int, 0),
    )
    wrap: bool = field(
        default=True,
        validator=validators.instance_of(bool),
    )
    boundary_condition: str = field(
        default="not-a-knot",
        validator=validators.optional(
            validators.in_({"natural", "periodic", "not-a-knot", "clamped"})
        ),
    )
    dt: FloatArray = field(
        init=False,
        default=1e-16,
        validator=getype_validator(float, 0)
    )
    t0: float = field(
        default=0.0,
        validator=getype_validator(float, 0)
    )
    num_inputs: int = field(
        init=False,
        default=0,
        validator=validators.instance_of(int),
    )
    num_segments: int = field(
        init=False,
        default=0,
        validator=validators.instance_of(int),
    )



class ArrayInterpolator(CUDAFactory):
    """Factory emitting CUDA device functions for interpolating array-driven
    forcing terms."""
    config_keys = ("wrap", "order", "boundary_condition")
    time_info = ("time", "dt", "t0")
    def __init__(
        self,
        precision: PrecisionDType,
        input_dict: Dict[str, FloatArray],
    ) -> None:
        super().__init__()
        config = ArrayInterpolatorConfig(
            precision=precision,
        )
        self.setup_compile_settings(config)
        self._coefficients = None
        self._input_array = None
        self.update_from_dict(input_dict)


    def update_from_dict(self, input_dict: Dict[str, Any]) -> bool:
        """Update the factory configuration from a user-supplied dictionary.

        Parameters
        ----------
        input_dict
            Dictionary containing input arrays and configuration options
            for the interpolated inputs.

        Returns
        -------
        bool
            ``True`` when the coefficients or configuration changed.

        Notes
        -----
        ## Input dictionary
        input_dict fields must include:

            - ``"time"``: 1D float array of sample times corresponding to
            input array values, or
                - ``"dt"``: uniform spacing between samples, and
                - ``"t0"``: starting time of the input samples.
            - ``[input_name]``: one-dimensional float array of samples for
            each input, where ``input_name`` is the name of the input signal
            as entered in the system definition.

            Fields may optionally include:

            - ``"order"``: polynomial order for spline interpolation,
            default 3.
            - ``"wrap"``: whether the input should wrap past the final
            value when the last time index is exceeded. When False the
            interpolator clamps to zero before ``t0`` and after the final
            sample.
            - ``"boundary_condition"``: boundary condition for splines.
            Defaults to ``"clamped"`` when ``"wrap"`` is False and to
            ``"periodic"`` when wrapping is enabled.

        The input arrays must all be one-dimensional and of the same length.

        The final interpolation result is an array of polynomial
        coefficients with shape (num_segments, num_inputs, order + 1),
        where num_segments is one less than the number of samples provided.

        ## Interpolation behaviour
        If ``"boundary_condition"`` is None, then spline coefficients are
        calculated in segments, with no continuity constraints. Otherwise,
        the spline coefficients are fit simultaneously for all segments,
        and end conditions are enforced according to the boundary condition:

        - ``"natural"``: second derivative at the ends of the curve is set
        to zero.
        - ``"periodic"``: the first and last segments are identical. For
        this condition, the first and last samples must match. This is the
        default when "wrap" is True, to avoid introducing a discontinuity on
        wrap.

        These boundary conditions are identical to those in [SciPy's
        CubicSpline interpolator]<https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.interpolate.CubicSpline.html>
        """

        config = {k: v for k, v in input_dict.items() if k in self.config_keys}
        inputs = {k: v for k, v in input_dict.items()
                   if k not in self.config_keys and k not in self.time_info}
        time = {k: v for k, v in input_dict.items() if k in self.time_info}

        # Update order first, for checks  in _normalise_input_array
        fn_changed = any(self.update_compile_settings(config))

        input_array = self._normalise_input_array(inputs)
        if np.array_equal(input_array, self.input_array):
            return False
        else:
            self._input_array = input_array

        dt, t0 = self._validate_time_inputs(time)
        base_segments = self.num_samples - 1
        config.update({'t0': t0,
                       'dt': dt,
                       'num_inputs': self.num_inputs})

        # Final update; invalidates cache if settings have changed.
        wrap_setting = config.get('wrap', self.wrap)
        if wrap_setting:
            if 'boundary_condition' not in config:
                config['boundary_condition'] = 'periodic'
            num_segments = base_segments
        elif 'boundary_condition' not in config:
            config['boundary_condition'] = 'clamped'
            num_segments = base_segments + 2
        else:
            boundary = config['boundary_condition']
            if boundary == 'clamped':
                num_segments = base_segments + 2
            else:
                num_segments = base_segments
        config['num_segments'] = num_segments
        fn_changed |= any(self.update_compile_settings(config))
        self._coefficients = self._compute_coefficients()

        return fn_changed

    def _normalise_input_array(
            self, input_dict: Dict[str, FloatArray]
    ) -> FloatArray:
        """Construct inputs array and check sizes.

        Parameters
        ----------
        input_dict
            Dictionary mapping input names to 1d arrays of samples.

        Returns
        -------
        np.ndarray of floats
            Input vectors stacked into a single array.
        Raises
        ------
        ValueError
            Raised when the input array is the wrong shape, type,
            or multiple arrys have different lengths.
        """

        for key, array in input_dict.items():
            try:
                array = np.asarray(array, dtype=self.precision)
            except ValueError:
                raise ValueError(
                    f"Forcing array {key} could not be converted "
                    f"to a NumPy array."
                )
            if array.ndim != 1:
                raise ValueError(f"Forcing array {key} must be "
                                 f"one-dimensional.")
            input_dict[key] = array
        input_vectors = list(input_dict.values())
        if not all(
            array.shape[0] == input_vectors[0].shape[0]
            for array in input_vectors
        ):
            raise ValueError(
                "All forcing vectors must have the same length / be sampled "
                "on the same grid",
            )
        input_array = np.column_stack(input_vectors)
        if input_array.shape[0] < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct"
                " splines.",
            )
        return input_array

    def _validate_time_inputs(self, time_dict: Dict[str, Any]
                              ) -> Tuple[float, float]:
        """Process and check time inputs.

        Parameters
        ----------
        time_dict
            Dictionary of time-related user inputs. If "dt" is provided,
            then this will be used and "t0" will be fetched from the dict or
            default to 0.0. If "time" is provided, dt will be calculated as the
             difference between samples, and t0 as time_dict['time'][0].
        Returns
        -------
        tuple (float, float)
            dt and t0, either obtained directly from time_dict or computed
            from a "time" array.
        Raises
        ------
        ValueError
            Raised if the time array is not strictly increasing or the
            spacing between samples is non-uniform.
        """

        if ("dt" in time_dict) and ("time" in time_dict):
            raise ValueError("Only one of dt or time should be provided.")
        if "dt" in time_dict:
            dt = time_dict["dt"]
            t0 = time_dict.get("t0", 0.0)
        elif "time" in time_dict:
            timeArray = time_dict["time"]
            if timeArray.ndim != 1:
                raise ValueError("Time array must be one-dimensional.")
            if timeArray.shape[0] != self.num_samples:
                raise ValueError("Time array length must match the number of"
                                 " samples in provided input vectors.")
            t0 = timeArray[0]
            time_differences = np.diff(timeArray)
            if np.any(time_differences <= 0.0):
                raise ValueError("Time array must be strictly increasing.")
            if not np.allclose(
                time_differences,
                np.full_like(time_differences,time_differences[0]),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError("Time array must be uniformly spaced.")
            dt = time_differences[0]
        else:
            raise ValueError("Either Time array or dt must be provided.")

        return dt, t0


    # ---------------------------------------------------------------------- #
    # Evaluation function machinery
    # ---------------------------------------------------------------------- #
    def build(self) -> Callable:
        """Compile device helpers and return them alongside host coefficients.

        Returns
        -------
        Callable
            Device function which evaluates input polynomials at a given time.
        """
        precision = self.precision
        order = self.order
        num_inputs = self.num_inputs
        resolution = precision(self.dt)
        inv_resolution = precision(precision(1.0) / resolution)
        start_time = precision(self.t0)
        num_segments = int32(self.num_segments)
        wrap = self.wrap
        boundary_condition = self.boundary_condition
        pad_clamped = (not wrap) and (boundary_condition == 'clamped')
        zero_value = self.precision(0.0)
        evaluation_start = precision(start_time - (
            resolution if pad_clamped else precision(0.0)))
        # no cover: start
        @cuda.jit(device=True, inline=True)
        def evaluate_all(
            time,
            coefficients,
            out
        ) -> None:
            """Evaluate all input polynomials at ``time`` on the device.

            Parameters
            ----------
            time : float
                Query time for evaluation.
            coefficients : device array
                Segment-major coefficients with trailing polynomial degrees.
            out : device array
                Output array to populate with evaluated input values.
            """

            scaled = (time - evaluation_start) * inv_resolution
            scaled_floor = math.floor(scaled)
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments)
                tau = precision(scaled - scaled_floor)
                in_range = True
            else:
                in_range = (scaled >= 0.0) and (scaled <= num_segments)
                seg = selp(idx < 0, int32(0), idx)
                seg = selp(seg >= num_segments,
                           int32(num_segments - 1), seg)
                tau = scaled - float(seg)

            # Evaluate polynomials using Horner's rule
            for input_index in range(num_inputs):
                acc = zero_value
                for k in range(order, -1, -1):
                    acc = acc * tau + coefficients[seg, input_index, k]
                out[input_index] = acc if in_range else zero_value
        # no cover: end

        # no cover: start
        @cuda.jit(device=True, inline=True)
        def evaluate_time_derivative(
            time,
            coefficients,
            out,
        ) -> None:
            """Evaluate the derivative of each driver polynomial."""

            scaled = (time - evaluation_start) * inv_resolution
            scaled_floor = math.floor(scaled)
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments)
                tau = precision(scaled - scaled_floor)
                in_range = True
            else:
                in_range = (scaled >= 0.0) and (scaled <= num_segments)
                seg = selp(idx < 0, int32(0), idx)
                seg = selp(seg >= num_segments,
                           int32(num_segments - 1), seg)
                tau = scaled - float(seg)

            for input_index in range(num_inputs):
                acc = zero_value
                for k in range(order, 0, -1):
                    acc = acc * tau + precision(k) * (
                        coefficients[seg, input_index, k]
                    )
                out[input_index] = (
                    acc * inv_resolution if in_range else zero_value
                )
        # no cover: end
        cache = InterpolatorCache(
            evaluation_function=evaluate_all,
            driver_del_t=evaluate_time_derivative,
        )
        return cache


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
    def evaluation_function(self) -> Callable:
        """Device function for evaluating all inputs."""
        return self.get_cached_output("evaluation_function")

    @property
    def driver_del_t(self) -> Callable:
        """Device function returning the interpolated driver time derivative."""

        return self.get_cached_output("driver_del_t")

    @property
    def coefficients(self) -> FloatArray:
        """Return the host-side coefficients array."""
        return self._coefficients

    # ---------------------------------------------------------------------- #
    # Inspection interface
    # ---------------------------------------------------------------------- #
    def get_input_array(self) -> FloatArray:
        """Return the input array.
        """
        return self._input_array

    def get_interpolated(
        self,
        eval_times: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the interpolated drivers on the device.

        Parameters
        ----------
        eval_times
            One-dimensional array of query times.

        Returns
        -------
        numpy.ndarray
            Interpolated driver values with shape ``(len(eval_times),
            num_inputs)``.

        Raises
        ------
        ValueError
            Raised when ``eval_times`` is not one-dimensional.
        RuntimeError
            Raised when interpolation coefficients are unavailable.
        """

        times = np.asarray(eval_times, dtype=self.precision)
        if times.ndim != 1:
            raise ValueError("eval_times must be one-dimensional.")

        num_points = times.size
        if num_points == 0:
            return np.empty((0, self.num_inputs), dtype=self.precision)

        coefficients = self.coefficients
        if coefficients is None:
            raise RuntimeError(
                "Interpolation coefficients have not been generated."
            )

        device_eval = self.evaluation_function

        # no cover: start
        @cuda.jit
        def _evaluate_kernel(times_device, coefficients_device, out_device):
            idx = cuda.grid(1)
            if idx < times_device.shape[0]:
                device_eval(
                    times_device[idx],
                    coefficients_device,
                    out_device[idx],
                )
        # no cover: end

        times_device = cuda.to_device(times)
        coefficients_device = cuda.to_device(coefficients)
        out_device = cuda.device_array(
            (num_points, self.num_inputs),
            dtype=self.precision,
        )

        threads_per_block = 128
        blocks_per_grid = (num_points + threads_per_block - 1) // (
            threads_per_block
        )
        _evaluate_kernel[blocks_per_grid, threads_per_block](
            times_device,
            coefficients_device,
            out_device,
        )
        cuda.synchronize()

        return out_device.copy_to_host()


    def plot_interpolated(
        self,
        eval_times: NDArray[np.floating],
    ) -> Tuple[Any, Any]: # pragma: no cover - optional dependency
        """Plot interpolated drivers against the sampled input data.

        Parameters
        ----------
        eval_times
            One-dimensional array of times at which to evaluate the
            interpolated drivers.

        Returns
        -------
        tuple
            Matplotlib figure and axes containing the plot.

        Raises
        ------
        ImportError
            Raised when :mod:`matplotlib` is not installed.
        ValueError
            Raised when ``eval_times`` is not one-dimensional.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Optional dependency matplotlib is required for plotting."
            ) from exc

        times = np.asarray(eval_times, dtype=self.precision)
        if times.ndim != 1:
            raise ValueError("eval_times must be one-dimensional.")

        interpolated = self.get_interpolated(times)

        sample_times = self.t0 + self.dt * np.arange(
            self.num_samples,
            dtype=self.precision,
        )
        sample_values = self.input_array.astype(self.precision, copy=False)

        if self.wrap and times.size:
            period = self.dt * self.num_samples
            min_eval = times.min()
            max_eval = times.max()
            repeats_before = int(
                math.ceil(max(0.0, (sample_times[0] - min_eval) / period))
            )
            repeats_after = int(
                math.ceil(max(0.0, (max_eval - sample_times[-1]) / period))
            )
            time_tiles = []
            value_tiles = []
            for step in range(repeats_before, 0, -1):
                time_tiles.append(sample_times - step * period)
                value_tiles.append(sample_values)
            time_tiles.append(sample_times)
            value_tiles.append(sample_values)
            for step in range(1, repeats_after + 1):
                time_tiles.append(sample_times + step * period)
                value_tiles.append(sample_values)
            marker_times = np.concatenate(time_tiles)
            marker_values = np.vstack(value_tiles)
        else:
            marker_times = sample_times
            marker_values = sample_values

        fig, ax = plt.subplots()
        for input_index in range(self.num_inputs):
            ax.plot(
                times,
                interpolated[:, input_index],
                label=f"Input {input_index}",
            )
            ax.plot(
                marker_times,
                marker_values[:, input_index],
                linestyle="None",
                marker="x",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Driver value")
        if self.num_inputs > 1:
            ax.legend()
        plt.show()
        return fig, ax

    # ---------------------------------------------------------------------- #
    # System-specific interface
    # ---------------------------------------------------------------------- #
    @staticmethod
    def check_against_system_drivers(
        inputs_dict: Dict[str, Union[float, bool, FloatArray]],
        system: "SymbolicODE",
    ):
        input_keys = [key for key in inputs_dict
            if key not in (
                ArrayInterpolator.config_keys + ArrayInterpolator.time_info
            )
        ]
        system_driver_keys = set(system.indices.drivers.symbol_map.keys())
        if len(input_keys) != system.num_drivers:
            raise ValueError(
                f"Number of inputs in inputs_dict "
                f"({len(input_keys)}) does not match number of "
                f"drivers in system ({system.num_drivers})."
            )
        if set(input_keys) != system_driver_keys:
            raise ValueError(f"input symbols in inputs_dict ("
                             f"{set(input_keys)}) do not match drivers "
                             f"symbols in system ({system_driver_keys}).")

    # ---------------------------------------------------------------------- #
    # Spline coefficient generation
    # ---------------------------------------------------------------------- #

    def _compute_coefficients(self) -> FloatArray:
        """Return spline coefficients respecting the requested boundary.

        Returns
        -------
        numpy.ndarray
            Segment-major coefficient array of shape ``(num_segments,
            num_inputs, order + 1)``.

        Raises
        ------
        ValueError
            Raised when periodic constraints are incompatible with the input
            configuration or when an unknown boundary condition is supplied.
        """
        boundary_condition = self.boundary_condition
        if boundary_condition not in {"natural",
                                      "periodic",
                                      "clamped",
                                      "not-a-knot",}:
            raise ValueError(
                f"Unsupported boundary condition: {boundary_condition}."
            )

        precision = self.precision
        base_inputs = self.input_array.astype(precision, copy=False)
        num_inputs = self.num_inputs
        order = self.order

        pad_with_zeros = (not self.wrap) and boundary_condition == "clamped"
        if pad_with_zeros:
            zero_row = np.zeros((1, num_inputs), dtype=precision)
            inputs = np.vstack((zero_row, base_inputs, zero_row))
        else:
            inputs = base_inputs

        num_segments = inputs.shape[0] - 1

        if boundary_condition == "periodic":
            if not self.wrap:
                raise ValueError(
                    "Periodic boundary conditions require wrap=True so that "
                    "the input repeats after the final segment."
                )
            if not np.allclose(inputs[0], inputs[-1]):
                raise ValueError(
                    "Periodic boundary conditions require the first and "
                    "last samples to match."
                )

        num_coeffs = num_segments * (order + 1)
        matrix = np.zeros((num_coeffs, num_coeffs), dtype=precision)
        rhs = np.zeros((num_coeffs, num_inputs), dtype=precision)
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
            rhs[row_index] = inputs[segment]
            row_index += 1

        # Function value constraints at the right edge of each segment.
        for segment in range(num_segments):
            base = coeff_index(segment, 0)
            for power in range(order + 1):
                matrix[row_index, base + power] = precision(1.0)
            rhs[row_index] = inputs[segment + 1]
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
                matrix[row_index, base_first] -= falling[
                    derivative, derivative
                ]
                row_index += 1

        elif boundary_condition == "clamped":
            remaining = order - 1
            derivative = 1
            while remaining > 0 and derivative <= order:
                base_start = coeff_index(0, 0)
                matrix[row_index, base_start + derivative] = falling[
                    derivative, derivative
                ]
                row_index += 1
                remaining -= 1
                if remaining == 0:
                    break
                base_end = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_end + power] = falling[
                        power, derivative
                    ]
                row_index += 1
                remaining -= 1
                derivative += 1

        elif boundary_condition == "not-a-knot":
            constraints_needed = order - 1
            constraints_added = 0
            highest_power = order
            for difference_order in range(1, order):
                if constraints_added >= constraints_needed:
                    break

                # Enforce vanishing forward difference at the start of the grid.
                start_row = row_index
                for offset in range(difference_order + 1):
                    coefficient = ((-1) ** (difference_order - offset))
                    coefficient *= math.comb(difference_order, offset)
                    segment = offset
                    matrix[start_row, coeff_index(segment, highest_power)] = (
                        precision(coefficient)
                    )
                row_index += 1
                constraints_added += 1
                if constraints_added >= constraints_needed:
                    break

                # Mirror the same finite-difference constraint at the end.
                end_row = row_index
                for offset in range(difference_order + 1):
                    coefficient = ((-1) ** (difference_order - offset))
                    coefficient *= math.comb(difference_order, offset)
                    segment = num_segments - 1 - (difference_order - offset)
                    matrix[end_row, coeff_index(segment, highest_power)] = (
                        precision(coefficient)
                    )
                row_index += 1
                constraints_added += 1

        if row_index != num_coeffs:
            raise ValueError(
                "Failed to assemble a square spline system; "
                "please verify boundary condition handling."
            )

        solution = np.linalg.solve(matrix, rhs)
        coefficients = solution.reshape(num_segments, order + 1, num_inputs)
        coefficients = np.transpose(coefficients, (0, 2, 1))
        return np.ascontiguousarray(coefficients, dtype=self.precision)

    # ---------------------------------------------------------------------- #
    # Getters and pass-through
    # ---------------------------------------------------------------------- #

    @property
    def num_inputs(self) -> int:
        """Return the number of input signals."""
        return self.input_array.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of samples available for interpolation."""
        return self.input_array.shape[0]

    @property
    def input_array(self) -> FloatArray:
        """Return the normalised input array."""
        return self._input_array

    @property
    def order(self) -> int:
        """Return the interpolating polynomial order."""
        return self.compile_settings.order

    @property
    def wrap(self) -> bool:
        """Return whether the input should wrap past the final sample."""
        return self.compile_settings.wrap

    @property
    def boundary_condition(self) -> Optional[str]:
        """Return the spline boundary condition to enforce, if any."""
        return self.compile_settings.boundary_condition

    @property
    def num_segments(self) -> int:
        """Return the number of polynomial segments."""
        return self.compile_settings.num_segments

    @property
    def precision(self) -> PrecisionDType:
        """Return the numerical precision used for the run."""
        return self.compile_settings.precision

    @property
    def t0(self) -> float:
        """Return the start time of the input samples."""
        return self.compile_settings.t0

    @property
    def dt(self) -> float:
        """Return the sample spacing."""
        return self.compile_settings.dt