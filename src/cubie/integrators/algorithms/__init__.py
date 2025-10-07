"""Factories for explicit and implicit algorithm step implementations."""

from typing import Any, Dict, Mapping, Optional, Type

from cubie._utils import split_applicable_settings

from .base_algorithm_step import BaseStepConfig, BaseAlgorithmStep
from .ode_explicitstep import ExplicitStepConfig, ODEExplicitStep
from .ode_implicitstep import ImplicitStepConfig, ODEImplicitStep
from .explicit_euler import ExplicitEulerStep
from .backwards_euler import BackwardsEulerStep
from .crank_nicolson import CrankNicolsonStep
from .backwards_euler_predict_correct import BackwardsEulerPCStep


__all__ = [
    "get_algorithm_step",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
    "_ALGORITHM_REGISTRY",
]

_ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithmStep]] = {
    "euler": ExplicitEulerStep,
    "backwards_euler": BackwardsEulerStep,
    "backwards_euler_pc": BackwardsEulerPCStep,
    "crank_nicolson": CrankNicolsonStep,
}


def get_algorithm_step(
    precision: type,
    settings: Optional[Mapping[str, Any]] = None,
    warn_on_unused: bool = True,
    **kwargs: Any,
) -> BaseAlgorithmStep:
    """Thin factory which filters arguments and instantiates an algorithm.

    Parameters
    ----------
    precision
        Floating-point datatype to use for computation
    settings
        Dictionary of settings to apply to the algorithm, can include any
        keywords from ``ALL_ALGORITHM_STEP_PARAMETERS``
    warn_on_unused
        If True, issue a warning for any values in ``settings`` that are not
        part of the requested algorithm's init signature.
    **kwargs
        Can be any additional keywords from ``ALL_ALGORITHM_STEP_PARAMETERS``.
        These will override any values in ``settings``.

    Returns
    -------
    BaseAlgorithmStep
        The requested step instance.

    Raises
    ------
    ValueError
        Raised when settings['algorithm'] does not match a known algorithm
        type or when required configuration keys are missing.
    """

    algorithm_settings: Dict[str, Any] = {}
    if settings is not None:
        algorithm_settings.update(settings)
    algorithm_settings.update(kwargs)

    algorithm_value = algorithm_settings.pop("algorithm", None)
    if algorithm_value is None:
        raise ValueError("Algorithm settings must include 'algorithm'.")
    algorithm_key = str(algorithm_value).lower()

    try:
        algorithm_type = _ALGORITHM_REGISTRY[algorithm_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown algorithm '{algorithm_value}'.") from exc

    algorithm_settings["precision"] = precision

    filtered, missing, unused = split_applicable_settings(
        algorithm_type,
        algorithm_settings,
        warn_on_unused=warn_on_unused,
    )
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"{algorithm_type.__name__} requires settings for: {missing_keys}"
        )

    return algorithm_type(**filtered)
