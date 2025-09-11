from abc import abstractmethod
from typing import Callable

from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.step_control.base_step_controller_config import BaseStepControllerConfig
class BaseStepController(CUDAFactory):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self)-> Callable:
        """Build the step control function.

        Device function signature should be:
        TODO"""
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