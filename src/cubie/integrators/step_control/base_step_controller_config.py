from abc import abstractmethod, ABC

from attrs import define, field, validators
from numpy import float32, float16, float64

@define
class BaseStepControllerConfig(ABC):
    precision: type = field(
            validator=validators.in_([float32, float64, float16]),
    )

    @abstractmethod
    def _validate_config(self):
        """Check for internal consistency, eg dt_min < dt_max"""

    @abstractmethod
    @property
    def dt_min(self) -> float:
        """Returns worst-case minimum step for calculating max iterations"""
        raise NotImplementedError

    @abstractmethod
    @property
    def dt_max(self)-> float:
        """Returns best-case maximum step for calculating max iterations"""
        raise NotImplementedError

    @abstractmethod
    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        raise NotImplementedError

    @abstractmethod
    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        raise NotImplementedError