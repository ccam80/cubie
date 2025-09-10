from abc import abstractmethod, ABC

from attrs import define

@define
class BaseStepControllerConfig(ABC):

    @abstractmethod
    @property
    def dt_min(self) -> float:
        """Returns worst-case minimum step for calculating max iterations"""
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