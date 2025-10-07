"""Interfaces for step-size controller configuration and factories.

Notes
-----
The abstract configuration and factory interfaces defined here encapsulate
shared behaviour for fixed and adaptive step controllers. Concrete
controllers extend these classes to compile CUDA device functions that
implement specific control strategies.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import warnings

import numba
from numpy import float32
from attrs import define, field

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import PrecisionDType, getype_validator, precision_converter, \
    precision_validator
from cubie.cuda_simsafe import from_dtype as simsafe_dtype

# Define all possible step controller parameters across all controller types
ALL_STEP_CONTROLLER_PARAMETERS = {
    'precision', 'n', 'step_controller', 'dt',
    'dt_min', 'dt_max', 'atol', 'rtol', 'algorithm_order',
    'min_gain', 'max_gain', 'safety',
    'kp', 'ki', 'kd', 'deadband_min', 'deadband_max',
    'gamma', 'max_newton_iters'
}

@define
class BaseStepControllerConfig(ABC):
    """Configuration interface for step-size controllers.

    Attributes
    ----------
    precision
        Precision used for controller calculations.
    n
        Number of state variables controlled per step.
    """

    precision: PrecisionDType = field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    n: int = field(default=1, validator=getype_validator(int, 0))

    @property
    def numba_precision(self) -> type:
        """Return the Numba compatible precision object."""

        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator compatible precision object."""
        return simsafe_dtype(self.precision)

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
    def dt0(self) -> float:
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
            'n': self.n,
        }


class BaseStepController(CUDAFactory):
    """Factory interface for compiling CUDA step-size controllers."""

    def __init__(self) -> None:
        """Initialise the base controller factory."""

        super().__init__()

    @abstractmethod
    def build(self) -> Callable:
        """Compile and return the CUDA device controller.

        Returns
        -------
        Callable
            Device function implementing the controller policy.
        """

    @property
    def precision(self) -> PrecisionDType:
        """Return the host precision used for computations."""

        return self.compile_settings.precision

    @property
    def numba_precision(self) -> type:
        """Return the Numba precision used for compilation."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator compatible precision."""

        return self.compile_settings.simsafe_precision

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
    def dt0(self) -> float:
        """Return the initial step size."""

        return self.compile_settings.dt0

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
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

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

        return recognised
