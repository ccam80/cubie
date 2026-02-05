"""Abstract interfaces for step-size controller configuration and
factories.

Published Classes
-----------------
:class:`ControllerCache`
    Cache container for compiled controller device functions.

:class:`BaseStepControllerConfig`
    Abstract attrs configuration shared by all controllers.

:class:`BaseStepController`
    Abstract factory base compiling CUDA step-size controllers.

Constants
---------
:data:`ALL_STEP_CONTROLLER_PARAMETERS`
    Union of all keyword arguments accepted across controller types.

Notes
-----
Concrete controllers extend these classes to compile CUDA device
functions that implement specific control strategies. Fixed and
adaptive controllers share the configuration and buffer registration
interfaces defined here.

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory providing compilation and cache management.
:mod:`cubie.integrators.step_control`
    Package-level entry point and controller registry.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import warnings

from attrs import define, field, validators

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
)
from cubie._utils import (
    getype_validator,
    opt_getype_validator,
    build_config,
    PrecisionDType,
)
from cubie.buffer_registry import buffer_registry

ALL_STEP_CONTROLLER_PARAMETERS = {
    "precision",
    "n",
    "step_controller",
    "dt",
    "dt_min",
    "dt_max",
    "atol",
    "rtol",
    "algorithm_order",
    "min_gain",
    "max_gain",
    "safety",
    "kp",
    "ki",
    "kd",
    "deadband_min",
    "deadband_max",
    "gamma",
    "newton_max_iters",
    "timestep_memory_location",
}
"""All keyword arguments accepted by step controllers.

These parameters can be passed as keyword arguments to any step
controller constructor or to :func:`get_controller`. The set is used
by parent components to filter kwargs before forwarding them.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``precision``
     - :class:`BaseStepControllerConfig`
     - Floating-point dtype for controller computations.
   * - ``n``
     - :class:`BaseStepControllerConfig`
     - Number of state variables controlled per step.
   * - ``step_controller``
     - :func:`~cubie.integrators.step_control.get_controller`
     - Controller type string (``'fixed'``, ``'i'``, ``'pi'``,
       ``'pid'``, ``'gustafsson'``).
   * - ``dt``
     - :class:`~cubie.integrators.step_control.fixed_step_controller.FixedStepControlConfig`
     - Fixed step size.
   * - ``dt_min``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Minimum permissible step size.
   * - ``dt_max``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Maximum permissible step size.
   * - ``atol``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Absolute tolerance vector.
   * - ``rtol``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Relative tolerance vector.
   * - ``algorithm_order``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Order of the integration algorithm.
   * - ``min_gain``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Minimum allowed gain factor.
   * - ``max_gain``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Maximum allowed gain factor.
   * - ``safety``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Safety scaling factor for step-size proposals.
   * - ``kp``
     - :class:`~cubie.integrators.step_control.adaptive_PI_controller.PIStepControlConfig`
     - Proportional gain.
   * - ``ki``
     - :class:`~cubie.integrators.step_control.adaptive_PI_controller.PIStepControlConfig`
     - Integral gain.
   * - ``kd``
     - :class:`~cubie.integrators.step_control.adaptive_PID_controller.PIDStepControlConfig`
     - Derivative gain.
   * - ``deadband_min``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Lower gain threshold for the unity deadband.
   * - ``deadband_max``
     - :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
     - Upper gain threshold for the unity deadband.
   * - ``gamma``
     - :class:`~cubie.integrators.step_control.gustafsson_controller.GustafssonStepControlConfig`
     - Damping factor for the Gustafsson predictor.
   * - ``newton_max_iters``
     - :class:`~cubie.integrators.step_control.gustafsson_controller.GustafssonStepControlConfig`
     - Maximum Newton iterations considered by the predictor.
   * - ``timestep_memory_location``
     - :class:`BaseStepControllerConfig`
     - Memory location for the timestep buffer (``'local'`` or
       ``'shared'``).
"""


@define
class ControllerCache(CUDADispatcherCache):
    """Cache container for compiled step-controller device functions.

    Attributes
    ----------
    device_function
        Compiled CUDA device function, or ``-1`` before compilation.
    """

    device_function: Union[Callable, int] = field(default=-1)


@define
class BaseStepControllerConfig(CUDAFactoryConfig, ABC):
    """Configuration interface for step-size controllers.

    Attributes
    ----------
    precision
        Precision used for controller calculations.
    n
        Number of state variables controlled per step.
    """

    n: int = field(default=1, validator=getype_validator(int, 0))
    _dt: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    timestep_memory_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    @abstractmethod
    def dt_min(self) -> float:
        """Return the minimum supported step size."""

    @property
    @abstractmethod
    def dt_max(self) -> float:
        """Return the maximum supported step size."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Return the initial step size used when integration starts."""

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        """Return ``True`` when the controller adapts its step size."""

    @property
    @abstractmethod
    def settings_dict(self) -> dict[str, object]:
        """Return a dictionary of configuration settings."""

        return {
            "n": self.n,
        }


class BaseStepController(CUDAFactory):
    """Factory interface for compiling CUDA step-size controllers."""

    _config_class = None  # Subclasses must override

    def __init__(
        self,
        precision: PrecisionDType,
        dt: float = None,
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise the step controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        dt
            Step size or initial step size.
        n
            Number of state variables.
        **kwargs
            Additional parameters passed to the config class.
        """
        super().__init__()
        self._user_step_params = {}
        self._resolve_step_params(dt, kwargs)
        config = build_config(
            self._config_class,
            required={"precision": precision, "n": n},
            **kwargs,
        )
        self.setup_compile_settings(config)
        self._ensure_sane_bounds()
        self.register_buffers()

    def _resolve_step_params(self, dt: float, kwargs: dict) -> None:
        """Resolve step parameters and track user-provided values.

        Subclasses override to implement controller-specific translation
        and set entries in ``self._user_step_params`` for user-provided
        values.

        Parameters
        ----------
        dt
            Step size, or None if not provided.
        kwargs
            Mutable dict of keyword arguments. Modified in place.
        """
        pass

    def _ensure_sane_bounds(self) -> None:
        """Ensure step bounds satisfy constraints.

        Called during __init__ and after update(). Subclasses override
        to validate bounds and fix constraint violations on
        non-user-provided parameters.
        """
        pass

    def register_buffers(self) -> None:
        """Register controller buffers with the central buffer registry.

        Registers the timestep_buffer using size from
        ``local_memory_elements`` and location from
        ``compile_settings.timestep_memory_location``.
        """
        config = self.compile_settings
        precision = config.precision
        size = self.local_memory_elements

        # Register timestep buffer
        buffer_registry.register(
            "timestep_buffer",
            self,
            size,
            config.timestep_memory_location,
            persistent=True,
            precision=precision,
        )

    @abstractmethod
    def build(self) -> ControllerCache:
        """Compile and return the CUDA device controller.

        Returns
        -------
        ControllerCache
            Cache containing the compiled controller device function.
        """

    @property
    def n(self) -> int:
        """Return the number of controlled state variables."""

        return self.compile_settings.n

    @property
    def dt_min(self) -> float:
        """Return the minimum supported step size."""

        return self.compile_settings.dt_min

    @property
    def dt_max(self) -> float:
        """Return the maximum supported step size."""

        return self.compile_settings.dt_max

    @property
    def dt(self) -> float:
        """Return the initial step size."""

        return self.compile_settings.dt

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if the controller is adaptive."""

        return self.compile_settings.is_adaptive

    @property
    @abstractmethod
    def local_memory_elements(self) -> int:
        """Return the number of local scratch elements required."""

        return 0

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the compile-time settings as a dictionary."""
        return self.compile_settings.settings_dict

    def update(
        self,
        updates_dict: Optional[dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> set[str]:
        """Propagate configuration updates to the compiled controller.

        Parameters
        ----------
        updates_dict
            Dictionary of configuration values to update.
        silent
            When ``True`` suppress warnings for recognised but unused
            controller parameters.
        **kwargs
            Additional configuration key-value pairs to update.

        Returns
        -------
        set[str]
            Names of parameters that were applied successfully.

        Raises
        ------
        KeyError
            Raised when an update references parameters that are not defined
            for any controller.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        # Track newly user-set step params
        for key in ("dt", "dt_min", "dt_max"):
            if key in updates_dict:
                self._user_step_params[key] = updates_dict[key]

        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised

        # Check if unrecognized parameters are valid step controller parameters
        # but not applicable to this specific controller
        valid_but_inapplicable = unrecognised & ALL_STEP_CONTROLLER_PARAMETERS
        truly_invalid = unrecognised - ALL_STEP_CONTROLLER_PARAMETERS

        # Mark valid controller parameters as recognized to prevent error
        # propagation
        recognised |= valid_but_inapplicable

        if valid_but_inapplicable:
            controller_type = self.__class__.__name__
            params_str = ", ".join(sorted(valid_but_inapplicable))
            warnings.warn(
                (
                    f"Parameters {{{params_str}}} are not recognized by "
                    f"{controller_type}; updates have been ignored."
                ),
                UserWarning,
                stacklevel=2,
            )

        if not silent and truly_invalid:
            raise KeyError(
                f"Unrecognized parameters in update: {truly_invalid}. "
                "These parameters were not updated.",
            )

        self._ensure_sane_bounds()
        return recognised
