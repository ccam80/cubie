from abc import abstractmethod
from typing import Optional, Callable
from warnings import warn

import numpy as np
from attrs import define, field

from cubie._utils import getype_validator, inrangetype_validator, clamp_factory, \
    float_array_validator
from cubie.errornorms import get_norm_factory
from cubie.integrators.step_control.base_step_controller import (
    BaseStepController, BaseStepControllerConfig
)



@define
class AdaptiveStepControlConfig(BaseStepControllerConfig):
    """
    Configuration parameters for a general adaptive step controller.

    This class can be used as-is for simple adaptive step controllers,
    or subclassed for more complex controllers. BaseStepControllerConfig
    defines the interface with the integrator loop; This class provides
    default logic for API properties.

    Any parameters required for subclassed controllers should live in here
    if they are used as compile-time constants - this will trigger the
    device function to recompile when they change.
    """

    _dt_min: float = field(default=1e-6, validator=getype_validator(float, 0))
    _dt_max: Optional[float] = field(
        default=None, validator=getype_validator(float, 0)
    )
    atol: np.ndarray = field(
        default=np.asarray([1e-6]),
        validator=float_array_validator,
    )
    rtol: np.ndarray = field(
        default=np.asarray([1e-6]),
        validator=float_array_validator,
    )
    algorithm_order: int = field(default=1, validator=getype_validator(int, 1))
    min_gain: float = field(
        default=0.3, validator=inrangetype_validator(float, 0, 1)
    )
    max_gain: float = field(default=2.0, validator=getype_validator(float, 1))
    safety: float = field(
        default=0.9, validator=inrangetype_validator(float, 0, 1)
    )

    def __attrs_post_init__(self):
        self._validate_config()

    def _validate_config(self):
        if self._dt_max is None:
            self._dt_max = self.dt_min * 100
        if self.dt_max < self.dt_min:
            warn(
                "dt_max ({self.dt_max}) < dt_min ({self.dt_min}). Setting "
                "dt_max = dt_min * 100"
            )
            self._dt_max = self.dt_min * 100

    @property
    def dt_min(self) -> float:
        """
        Returns the minimum time step size.

        Returns
        -------
        float
            Minimum time step size from the loop step configuration.
        """
        return self._dt_min

    @property
    def dt_max(self) -> float:
        """Returns the maximum time step size."""
        return self._dt_max

    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return (self.dt_min + self.dt_max) / 2

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return True


class BaseAdaptiveStepController(BaseStepController):
    def __init__(
        self,
        config: AdaptiveStepControlConfig,
        norm_type: str,
        norm_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.setup_compile_settings(config)

        norm_factory = get_norm_factory(norm_type)
        if norm_kwargs is None:
            norm_kwargs = {}
        try:
            self.norm_func = norm_factory(config.precision,
                                          config.n,
                                          **norm_kwargs)
        except TypeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                "Invalid parameters for chosen norm: "
                f"{norm_kwargs}. Check the norm function for expected "
                "parameters."
            ) from exc

    def sanitise_tol_array(self, tol, n, precision):
        if isinstance(tol, float):
            tol = np.asarray([tol] * n, dtype=precision)
        else:
            tol = np.asarray(tol, dtype=precision)
            if tol.shape[0] != n:
                raise ValueError("atol must have shape (n,).")
        return tol

    def build(self):
        return self.build_controller(
                precision=self.precision,
                clamp=clamp_factory(self.precision),
                norm_func=self.norm_func,
                min_gain=self.min_gain,
                max_gain=self.max_gain,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                n=self.compile_settings.n,
                atol=self.atol,
                rtol=self.rtol,
                order=self.compile_settings.algorithm_order,
                safety=self.compile_settings.safety,
        )

    @abstractmethod
    def build_controller(self,
                         precision: type,
                         clamp: Callable,
                         norm_func: Callable,
                         min_gain: float,
                         max_gain: float,
                         dt_min: float,
                         dt_max: float,
                         n: int,
                         atol: np.ndarray,
                         rtol: np.ndarray,
                         order: np.ndarray,
                         safety: float):
        raise NotImplementedError

    @property
    def kp(self) -> float:
        """Returns proportional gain."""
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        """Returns integral gain."""
        return self.compile_settings.ki

    @property
    def min_gain(self) -> float:
        """Returns minimum gain."""
        return self.compile_settings.min_gain

    @property
    def max_gain(self) -> float:
        """Returns maximum gain."""
        return self.compile_settings.max_gain

    @property
    def atol(self) -> np.ndarray:
        """ Returns absolute tolerance."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> np.ndarray:
        """ Returns relative tolerance."""
        return self.compile_settings.rtol

    @property
    @abstractmethod
    def local_memory_required(self) -> int:
        return NotImplementedError

