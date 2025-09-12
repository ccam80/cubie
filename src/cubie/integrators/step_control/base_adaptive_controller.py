from abc import abstractmethod
from typing import Optional

from numba import cuda, int32
from numba import from_dtype

from cubie import clamp_factory
from cubie.errornorms import get_norm_factory
from cubie.integrators.step_control.adaptive_PI_config import \
    PIStepControlConfig
from cubie.integrators.step_control.adaptive_step_config import \
    AdaptiveStepControlConfig
from cubie.integrators.step_control.base_step_controller import \
    BaseStepController


class BaseAdaptiveStepController(BaseStepController):


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
    def atol(self) -> float:
        """ Returns absolute tolerance."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> float:
        """ Returns relative tolerance."""
        return self.compile_settings.rtol

    @property
    @abstractmethod
    def local_memory_required(self) -> int:
        return NotImplementedError

