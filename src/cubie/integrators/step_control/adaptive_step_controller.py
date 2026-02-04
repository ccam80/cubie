"""Shared infrastructure for adaptive step-size controllers.

Published Classes
-----------------
:class:`AdaptiveStepControlConfig`
    Attrs configuration shared by all adaptive controllers.

    >>> from numpy import float64
    >>> config = AdaptiveStepControlConfig(precision=float64)
    >>> config.is_adaptive
    True

:class:`BaseAdaptiveStepController`
    Abstract factory base for adaptive controllers.

See Also
--------
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepController`
    Abstract base class for all controllers.
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepControllerConfig`
    Base configuration class.
"""

from abc import abstractmethod
from typing import Callable, Optional
from warnings import warn

from numpy import asarray, ndarray, sqrt
from attrs import Converter, define, field
from numpy.typing import ArrayLike

from cubie._utils import (
    PrecisionDType,
    clamp_factory,
    float_array_validator,
    getype_validator,
    inrangetype_validator,
    opt_getype_validator,
    tol_converter,
)
from cubie.integrators.step_control.base_step_controller import (
    BaseStepController,
    BaseStepControllerConfig,
    ControllerCache,
)


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
        default=1.0, validator=getype_validator(float, 0)
    )
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
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
        super().__attrs_post_init__()
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
    def dt(self) -> float:
        """Return the initial step size."""
        if self._dt is not None:
            return self.precision(self._dt)
        return self.precision(sqrt(self.dt_min * self.dt_max))

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
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "atol": self.atol,
                "rtol": self.rtol,
                "algorithm_order": self.algorithm_order,
                "min_gain": self.min_gain,
                "max_gain": self.max_gain,
                "safety": self.safety,
                "deadband_min": self.deadband_min,
                "deadband_max": self.deadband_max,
                "dt": self.dt,
            }
        )
        return settings_dict


class BaseAdaptiveStepController(BaseStepController):
    """Base class for adaptive step-size controllers."""

    _config_class = AdaptiveStepControlConfig

    def _resolve_step_params(self, dt: float, kwargs: dict) -> None:
        """Derive bounds from dt and track user-provided values.

        Parameters
        ----------
        dt
            Initial step size, or None if not provided.
        kwargs
            Mutable dict of keyword arguments. Modified in place.
        """
        # Track user-provided values BEFORE derivation
        if dt is not None:
            self._user_step_params["dt"] = dt
        if "dt_min" in kwargs:
            self._user_step_params["dt_min"] = kwargs["dt_min"]
        if "dt_max" in kwargs:
            self._user_step_params["dt_max"] = kwargs["dt_max"]

        # Derive missing bounds from dt
        if dt is not None:
            kwargs.setdefault("dt_min", dt / 100)
            kwargs.setdefault("dt_max", dt * 100)
            kwargs["dt"] = dt

    def _ensure_sane_bounds(self) -> None:
        """Fix bounds if constraints violated on non-user-provided params."""
        dt = self.dt
        dt_min = self.dt_min
        dt_max = self.dt_max
        fixes = {}

        if dt_max < dt_min and self._user_step_params.get("dt_max") is None:
            fixes["dt_max"] = dt * 100
        if dt < dt_min and self._user_step_params.get("dt_min") is None:
            fixes["dt_min"] = dt / 100
        if dt > dt_max and self._user_step_params.get("dt_max") is None:
            fixes["dt_max"] = dt * 100

        if fixes:
            self.update_compile_settings(fixes, silent=True)

    def build(self) -> ControllerCache:
        """Construct the device function implementing the controller.

        Returns
        -------
        ControllerCache
            Cache containing the compiled adaptive controller device
            function.
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
        atol: ndarray,
        rtol: ndarray,
        algorithm_order: int,
        safety: float,
    ) -> ControllerCache:
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
        ControllerCache
            Cache containing the compiled controller device function.
        """
        raise NotImplementedError

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
    def atol(self) -> ndarray:
        """Return absolute tolerance."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> ndarray:
        """Return relative tolerance."""
        return self.compile_settings.rtol

    @property
    @abstractmethod
    def local_memory_elements(self) -> int:
        """Return number of floats required for controller local memory."""
        raise NotImplementedError
