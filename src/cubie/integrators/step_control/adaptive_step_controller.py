"""Shared infrastructure for adaptive step-size controllers."""

from abc import abstractmethod
from typing import Callable, Optional
from warnings import warn

import numpy as np
from attrs import Converter, define, field
from numpy.typing import ArrayLike

from cubie._utils import (
    PrecisionDType,
    clamp_factory,
    float_array_validator,
    getype_validator,
    inrangetype_validator,
)
from cubie.integrators.step_control.base_step_controller import (
    BaseStepController, BaseStepControllerConfig
)


def tol_converter(
    value: float | ArrayLike,
    self_: "AdaptiveStepControlConfig",
) -> np.ndarray:
    """Convert tolerance input into an array with controller precision.

    Parameters
    ----------
    value
        Scalar or array-like tolerance specification.
    self_
        Configuration instance providing precision and dimension information.

    Returns
    -------
    numpy.ndarray
        Tolerance array with one value per state variable.

    Raises
    ------
    ValueError
        Raised when ``value`` cannot be broadcast to the expected shape.
    """

    if np.isscalar(value):
        tol = np.asarray([value] * self_.n, dtype=self_.precision)
    else:
        tol = np.asarray(value, dtype=self_.precision)
        if tol.shape[0] != self_.n:
            raise ValueError("tol must have shape (n,).")
    return tol


@define
class AdaptiveStepControlConfig(BaseStepControllerConfig):
    """Configuration for adaptive step controllers.

    Notes
    -----
    Parameters influencing compilation should live here so that device
    functions are rebuilt when they change.
    """

    _dt_min: float = field(default=1e-6, validator=getype_validator(float, 0))
    _dt_max: Optional[float] = field(
        default=None, validator=getype_validator(float, 0)
    )
    atol: np.ndarray = field(
        default=np.asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True)
    )
    rtol: np.ndarray = field(
        default=np.asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True)
    )
    algorithm_order: int = field(default=1, validator=getype_validator(int, 1))
    _min_gain: float = field(
        default=0.3,
        validator=inrangetype_validator(float, 0, 1),
    )
    _max_gain: float = field(
        default=2.0,
        validator=getype_validator(float, 1),
    )
    _safety: float = field(
        default=0.9,
        validator=inrangetype_validator(float, 0, 1),
    )
    _deadband_min: float = field(
        default=1.0,
        validator=inrangetype_validator(float, 0, 1.0),
    )
    _deadband_max: float = field(
        default=1.2,
        validator=getype_validator(float, 1.0),
    )

    def __attrs_post_init__(self) -> None:
        """Ensure step limits are coherent after initialisation."""

        if self._dt_max is None:
            self._dt_max = self._dt_min * 100
        elif self._dt_max < self._dt_min:
            warn(
                (
                    f"dt_max ({self._dt_max}) < dt_min ({self._dt_min}). "
                    "Setting dt_max = dt_min * 100"
                )
            )
            self._dt_max = self._dt_min * 100

        if self._deadband_min > self._deadband_max:
            self._deadband_min, self._deadband_max = (
                self._deadband_max,
                self._deadband_min,
            )


    @property
    def dt_min(self) -> float:
        """Return the minimum permissible step size."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the maximum permissible step size."""
        value = self._dt_max
        if value is None:
            value = self._dt_min * 100
        return self.precision(value)

    @property
    def dt0(self) -> float:
        """Return the initial step size."""
        return self.precision(np.sqrt(self.dt_min * self.dt_max))

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because the controller adapts step size."""
        return True

    @property
    def min_gain(self) -> float:
        """Return the minimum gain factor."""
        return self.precision(self._min_gain)

    @property
    def max_gain(self) -> float:
        """Return the maximum gain factor."""
        return self.precision(self._max_gain)

    @property
    def safety(self) -> float:
        """Return the safety scaling factor."""
        return self.precision(self._safety)

    @property
    def deadband_min(self) -> float:
        """Return the lower gain threshold for the unity deadband."""

        return self.precision(self._deadband_min)

    @property
    def deadband_max(self) -> float:
        """Return the upper gain threshold for the unity deadband."""

        return self.precision(self._deadband_max)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update(
            {
                'dt_min': self.dt_min,
                'dt_max': self.dt_max,
                'atol': self.atol,
                'rtol': self.rtol,
                'algorithm_order': self.algorithm_order,
                'min_gain': self.min_gain,
                'max_gain': self.max_gain,
                'safety': self.safety,
                'deadband_min': self.deadband_min,
                'deadband_max': self.deadband_max,
                'dt': self.dt0,
            }
        )
        return settings_dict

class BaseAdaptiveStepController(BaseStepController):
    """Base class for adaptive step-size controllers."""

    def __init__(
        self,
        config: AdaptiveStepControlConfig,
    ) -> None:
        """Initialise the adaptive controller.

        Parameters
        ----------
        config
            Configuration for the controller.
        """
        super().__init__()
        self.setup_compile_settings(config)

    def build(self) -> Callable:
        """Construct the device function implementing the controller.

        Returns
        -------
        Callable
            Compiled CUDA device function for adaptive control.
        """
        return self.build_controller(
            precision=self.precision,
            clamp=clamp_factory(self.precision),
            min_gain=self.min_gain,
            max_gain=self.max_gain,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            n=self.compile_settings.n,
            atol=self.atol,
            rtol=self.rtol,
            algorithm_order=self.compile_settings.algorithm_order,
            safety=self.compile_settings.safety,
        )

    @abstractmethod
    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_gain: float,
        max_gain: float,
        dt_min: float,
        dt_max: float,
        n: int,
        atol: np.ndarray,
        rtol: np.ndarray,
        algorithm_order: int,
        safety: float,
    ) -> Callable:
        """Create the device function for the specific controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce values.
        clamp
            Callable that limits step updates.
        min_gain
            Minimum allowed gain when adapting the step size.
        max_gain
            Maximum allowed gain when adapting the step size.
        dt_min
            Minimum permissible step size.
        dt_max
            Maximum permissible step size.
        n
            Number of state variables handled by the controller.
        atol
            Absolute tolerance vector.
        rtol
            Relative tolerance vector.
        algorithm_order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.

        Returns
        -------
        Callable
            CUDA device function implementing the controller policy.
        """
        raise NotImplementedError

    # @property
    # def kp(self) -> float:
    #     """Returns proportional gain."""
    #     return self.compile_settings.kp
    #
    # @property
    # def ki(self) -> float:
    #     """Returns integral gain."""
    #     return self.compile_settings.ki

    @property
    def min_gain(self) -> float:
        """Return the minimum gain factor."""

        return self.compile_settings.min_gain

    @property
    def max_gain(self) -> float:
        """Return the maximum gain factor."""

        return self.compile_settings.max_gain

    @property
    def safety(self) -> float:
        """Return the safety scaling factor."""

        return self.compile_settings.safety

    @property
    def deadband_min(self) -> float:
        """Return the lower gain threshold for unity selection."""

        return self.compile_settings.deadband_min

    @property
    def deadband_max(self) -> float:
        """Return the upper gain threshold for unity selection."""

        return self.compile_settings.deadband_max

    @property
    def algorithm_order(self) -> int:
        """Return the integration algorithm order assumed by the controller."""

        return int(self.compile_settings.algorithm_order)

    @property
    def atol(self) -> np.ndarray:
        """Return absolute tolerance."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> np.ndarray:
        """Return relative tolerance."""
        return self.compile_settings.rtol

    @property
    @abstractmethod
    def local_memory_elements(self) -> int:
        """Return number of floats required for controller local memory."""
        raise NotImplementedError

