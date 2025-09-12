from abc import abstractmethod, ABC

from attrs import define, validators, field
from numpy import float32, float16, float64

from cubie._utils import getype_validator

@define
class BaseStepControllerConfig(ABC):
    precision: type = field(
            default=float32,
            validator=validators.in_([float16, float32, float64])
    )
    n: int = field(
           default=1,
           validator=getype_validator(int, 0)
    )

    @abstractmethod
    def _validate_config(self):
        """Check for internal consistency, eg dt_min < dt_max"""

    @property
    @abstractmethod
    def dt_min(self) -> float:
        """Returns worst-case minimum step for calculating max iterations"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dt_max(self)-> float:
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
        """Returns whether the step controller is adaptive."""
        raise NotImplementedError