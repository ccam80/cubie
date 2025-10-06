"""Public entry points for step-size controllers."""

import warnings
from typing import Any, Dict, Mapping, Optional, Type

from cubie._utils import split_applicable_settings

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

_CONTROLLER_REGISTRY: Dict[str, Type[BaseStepController]] = {
    "fixed": FixedStepController,
    "i": AdaptiveIController,
    "pi": AdaptivePIController,
    "pid": AdaptivePIDController,
    "gustafsson": GustafssonController,
}

def get_controller(
    kind: str,
    settings: Optional[Mapping[str, Any]] = None,
    warn_on_unused: bool = True,
    **kwargs: Any,
) -> BaseStepController:
    """Return a controller instance based on ``kind``.

    Parameters
    ----------
    kind
        Simplified name of the controller (``"fixed"``, ``"i"``, ``"pi"``,
        ``"pid"``, ``"gustafsson"``).
    settings
        Mapping of step control keyword arguments supplied by the caller.
        Values in ``settings`` are merged with ``kwargs``.
    warn_on_unused
        Emit a warning when unused settings remain after filtering.
    **kwargs
        Additional keyword arguments that override entries in
        ``settings``.

    Returns
    -------
    BaseStepController
        Instance of the requested controller.

    Raises
    ------
    ValueError
        Raised when ``kind`` does not match a known controller type or when
        required configuration keys are missing.
    """

    controller_key = kind.lower()
    try:
        controller_type = _CONTROLLER_REGISTRY[controller_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown controller type: {kind}") from exc

    merged_settings = {}
    if settings is not None:
        merged_settings.update(settings)
    merged_settings.update(kwargs)

    filtered, missing, unused = split_applicable_settings(
        controller_type,
        merged_settings,
        warn_on_unused=True
    )
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"{controller_type.__name__} requires settings for: "
            f"{missing_keys}"
        )

    return controller_type(**filtered)
