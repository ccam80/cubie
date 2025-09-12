"""Step size controller interfaces."""

from .adaptive_I_controller import AdaptiveIController
from .adaptive_PI_controller import AdaptivePIController
from .adaptive_PID_controller import AdaptivePIDController
from .gustafsson_controller import GustafssonController

__all__ = [
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "GustafssonController",
    "get_controller",
]


def get_controller(kind: str, **kwargs):
    """Return a controller instance based on ``kind``.

    Parameters
    ----------
    kind
        Simplified name of the controller (``"i"``, ``"pi"``, ``"pid"``,
        ``"gustafsson"``).
    **kwargs
        Arguments passed to the controller constructor.
    """
    kind = kind.lower()
    if kind == "i":
        return AdaptiveIController(**kwargs)
    if kind == "pi":
        return AdaptivePIController(**kwargs)
    if kind == "pid":
        return AdaptivePIDController(**kwargs)
    if kind in {"gustafsson", "predictive"}:
        return GustafssonController(**kwargs)
    raise ValueError(f"Unknown controller type: {kind}")
