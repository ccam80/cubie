"""Public entry points for step-size controllers."""

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
    "FixedStepController",
    "get_controller",
]


def get_controller(kind: str, **kwargs: object) -> BaseStepController:
    """Return a controller instance based on ``kind``.

    Parameters
    ----------
    kind
        Simplified name of the controller (``"fixed"``, ``"i"``, ``"pi"``,
        ``"pid"``, ``"gustafsson"``).
    **kwargs
        Arguments passed to the controller constructor. These are set by the
        signature of the controller.

    Returns
    -------
    BaseStepController
        Instance of the requested controller.

    Raises
    ------
    ValueError
        Raised when ``kind`` does not match a known controller type.
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
