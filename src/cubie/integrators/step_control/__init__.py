"""Step size controller interfaces."""

from .adaptive_I_controller import AdaptiveIController
from .adaptive_PI_controller import AdaptivePIController
from .adaptive_PID_controller import AdaptivePIDController
from .fixed_step_controller import FixedStepController
from .gustafsson_controller import GustafssonController
from .base_step_controller import BaseStepController

__all__ = [
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "GustafssonController",
    "get_controller",
]


def get_controller(kind: str, **kwargs) -> BaseStepController:
    """Return a controller instance based on ``kind``.

    Parameters
    ----------
    kind
        Simplified name of the controller (``"i"``, ``"pi"``, ``"pid"``,
        ``"gustafsson"``).
    **kwargs
        Arguments passed to the controller constructor. These are set by the
        signature of the controller.

    Returns
    -------
    BaseStepController
        Instance of the requested controller.
    """
    kind = kind.lower()
    if kind == "fixed":
        return FixedStepController(**kwargs)
    if kind == "i":
        return AdaptiveIController(**kwargs)
    if kind == "pi":
        return AdaptivePIController(**kwargs)
    if kind == "pid":
        return AdaptivePIDController(**kwargs)
    if kind == "gustafsson":
        return GustafssonController(**kwargs)
    raise ValueError(f"Unknown controller type: {kind}")
