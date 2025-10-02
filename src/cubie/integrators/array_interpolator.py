"""Utilities for transforming array input samples into CUDA interpolants."""

import math
from typing import Callable, Dict, Optional, Set, TYPE_CHECKING, Union, Any, \
    Tuple

import numpy as np
from attrs import define, field, validators
from numba import cuda, int32
from numpy.typing import NDArray

from cubie._utils import gttype_validator
from cubie.cuda_simsafe import selp

if TYPE_CHECKING:
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import (
    PrecisionDtype,
    getype_validator,
    precision_validator,
)

FloatArray = NDArray[np.floating]


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
        Number of polynomial segments, equal to the number of samples - 1.
    """

    precision: PrecisionDtype = field(validator=precision_validator)
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
        validator=getype_validator(float, 0)
    )
    t0: float = field(
        default=0.0,
        validator=getype_validator(float, 0)
    )
    num_inputs: int = field(
        init=False,
        validator=validators.instance_of(int),
    )
    num_segments: int = field(
        init=False,
        validator=validators.instance_of(int),
    )



class ArrayInterpolator(CUDAFactory):
    """Factory emitting CUDA device functions for interpolating array-driven
    forcing terms."""
    config_keys = ("wrap", "order", "boundary_condition")
    time_info = ("time", "dt", "t0")
    def __init__(
        self,
        precision: PrecisionDtype,
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


    def update_from_dict(self, input_dict: Dict[str, Any]):
        """Update the factory configuration from a user-supplied dictionary

        Parameters
        ----------
        input_dict
            Dictionary containing input arrays and configuration options
            for the interpolated inputs.

        Returns
        -------
        None

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
            value when the last time index is exceeded. Default is False,
            which means the input value will be zero before t0 and after
            the final sample.
            - ``"boundary_condition"``: boundary condition for splines.
            defaults to "None", if wrap is False, otherwise "periodic".

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
            return
        else:
            self._input_array = input_array

        dt, t0 = self._validate_time_inputs(time)
        config.update({'t0': t0,
                       'dt': dt,
                       'num_inputs': self.num_inputs,
                       'num_segments': self.num_samples - 1})

        # Final update; invalidates cache if settings have changed.
        if config.get('wrap', False):
            if 'boundary_condition' not in config:
                config['boundary_condition'] = 'periodic'
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

        order = self.order
        num_inputs = self.num_inputs
        resolution = self.dt
        inv_resolution = 1.0 / resolution
        start_time = self.t0
        num_segments = int32(self.num_segments)
        wrap = self.wrap
        zero_value = self.precision(0.0)

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

            scaled = (time - start_time) * inv_resolution
            scaled_floor = math.floor(scaled)
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments)
                tau = scaled - scaled_floor
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

        return evaluate_all


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
        """Return the device function for evaluating all inputs."""
        return self.device_function

    @property
    def coefficients(self) -> FloatArray:
        """Return the host-side coefficients array."""
        return self._coefficients

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
        inputs = self.input_array.astype(precision, copy=False)
        num_segments = self.num_segments
        num_inputs = self.num_inputs
        order = self.order

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
    def precision(self) -> PrecisionDtype:
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