"""Public entry points for step-size controllers."""

import warnings
from typing import Any, Dict, Mapping, Optional, Type

from cubie._utils import PrecisionDType, split_applicable_settings

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
    "_CONTROLLER_REGISTRY",
]

_CONTROLLER_REGISTRY: Dict[str, Type[BaseStepController]] = {
    "fixed": FixedStepController,
    "i": AdaptiveIController,
    "pi": AdaptivePIController,
    "pid": AdaptivePIDController,
    "gustafsson": GustafssonController,
}

def get_controller(
    precision: PrecisionDType,
    settings: Optional[Mapping[str, Any]] = None,
    warn_on_unused: bool = True,
    **kwargs: Any,
) -> BaseStepController:
    """Return a controller instance from a settings mapping.

    Parameters
    ----------
    precision
        Floating-point dtype used when compiling the controller.
    settings
        Mapping of step control keyword arguments supplied by the caller.
        Values in ``settings`` are merged with ``kwargs``. The mapping must
        include ``"step_controller"`` to identify the controller factory.
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
        Raised when ``step_controller`` does not match a known controller type
        or when required configuration keys are missing.
    """

    controller_settings = {}
    if settings is not None:
        controller_settings.update(settings)
    controller_settings.update(kwargs)

    step_controller_value = controller_settings.pop("step_controller", None)
    if step_controller_value is None:
        raise ValueError("No step controller specified")
    controller_key = step_controller_value.lower()

    try:
        controller_type = _CONTROLLER_REGISTRY[controller_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Unknown controller type: {step_controller_value}"
        ) from exc

    controller_settings["precision"] = precision

    filtered, missing, unused = split_applicable_settings(
        controller_type,
        controller_settings,
        warn_on_unused=warn_on_unused
    )
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"{controller_type.__name__} requires settings for: "
            f"{missing_keys}"
        )

    controller = controller_type(**filtered)
    return controller
