"""Base classes for step-size controllers.

This module defines abstract interfaces used by the various step-size
controllers. Concrete implementations inherit from these classes to provide
specific control strategies.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from numpy import float32, float16, float64
from attrs import define, field, validators

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import getype_validator

@define
class BaseStepControllerConfig(ABC):
    """Configuration interface for step-size controllers.

    Attributes
    ----------
    precision
        Numerical precision for controller calculations.
    n
        Number of state variables the controller operates on.
    """

    precision: type = field(
        default=float32, validator=validators.in_([float16, float32, float64])
    )
    n: int = field(default=1, validator=getype_validator(int, 0))

    @abstractmethod
    def _validate_config(self) -> None:
        """Check for internal consistency of configuration."""

    @property
    @abstractmethod
    def dt_min(self) -> float:
        """Returns worst-case minimum step for calculating max iterations"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dt_max(self) -> float:
        """Returns best-case maximum step for calculating max iterations"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        """Return whether the step controller is adaptive."""
        raise NotImplementedError


class BaseStepController(CUDAFactory):
    """Abstract base class for step-size controllers."""

    def __init__(self) -> None:
        """Initialise the step controller."""
        super().__init__()

    @abstractmethod
    def build(self) -> Callable:
        """Build the step control device function.

        The returned device function has signature

        ``status = control_fn(dt, state, state_tmp, accept, error_integral)``.

        Returns
        -------
        Callable
            Compiled device function implementing the controller.
        """
        raise NotImplementedError

    @property
    def precision(self) -> type:
        """Returns the precision type for numerical computations."""
        return self.compile_settings.precision

    @property
    def dt_min(self) -> float:
        """Returns worst-case minimum step for calculating max iterations"""
        return self.compile_settings.dt_min

    @property
    def dt_max(self) -> float:
        """Returns best-case maximum step for calculating max iterations"""
        return self.compile_settings.dt_max

    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return self.compile_settings.dt0

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return self.compile_settings.is_adaptive

    def update(self,
               updates_dict : Optional[dict] = None,
               silent: bool = False,
               **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface.

        This method will invalidate the cache if an update is successful.
        Use silent=True when doing bulk updates with other component parameters
        to suppress warnings about unrecognized keys.

        Parameters
        ----------
        updates_dict
            Dictionary of parameters to update.
        silent
            If True, suppress warnings about unrecognized parameters.
        **kwargs
            Parameter updates to apply as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.
        """
        if updates_dict is None:
            updates_dict = {}
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
