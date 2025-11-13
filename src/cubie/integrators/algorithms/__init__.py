"""Factories for explicit and implicit algorithm step implementations."""

from typing import Any, Mapping, Optional, Tuple, Type

from cubie._utils import split_applicable_settings

from .base_algorithm_step import BaseAlgorithmStep, BaseStepConfig, ButcherTableau
from .ode_explicitstep import ExplicitStepConfig, ODEExplicitStep
from .ode_implicitstep import ImplicitStepConfig, ODEImplicitStep
from .backwards_euler import BackwardsEulerStep
from .backwards_euler_predict_correct import BackwardsEulerPCStep
from .crank_nicolson import CrankNicolsonStep
from .explicit_euler import ExplicitEulerStep
from .generic_dirk import DIRKStep
from .generic_dirk_tableaus import DIRK_TABLEAU_REGISTRY, DIRKTableau
from .generic_firk import FIRKStep
from .generic_firk_tableaus import FIRK_TABLEAU_REGISTRY, FIRKTableau
from .generic_erk import ERKStep, ERKTableau
from .generic_erk_tableaus import ERK_TABLEAU_REGISTRY
from .generic_rosenbrock_w import GenericRosenbrockWStep
from .generic_rosenbrockw_tableaus import ROSENBROCK_TABLEAUS, RosenbrockTableau


__all__ = [
    "get_algorithm_step",
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
    "DIRKStep",
    "FIRKStep",
    "ERKStep",
    "GenericRosenbrockWStep",
    "_ALGORITHM_REGISTRY",
    "DIRKTableau",
    "DIRK_TABLEAU_REGISTRY",
    "FIRKTableau",
    "FIRK_TABLEAU_REGISTRY",
    "ERKTableau",
    "ERK_TABLEAU_REGISTRY",
]

_ALGORITHM_REGISTRY = {
    "euler": ExplicitEulerStep,
    "backwards_euler": BackwardsEulerStep,
    "backwards_euler_pc": BackwardsEulerPCStep,
    "crank_nicolson": CrankNicolsonStep,
    "dirk": DIRKStep,
    "firk": FIRKStep,
    "erk": ERKStep,
    "rosenbrock": GenericRosenbrockWStep,
}

_TABLEAU_REGISTRY_BY_ALGORITHM = {
    key: (constructor, None)
    for key, constructor in _ALGORITHM_REGISTRY.items()
}

for alias, tableau in ERK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (ERKStep, tableau)

for alias, tableau in DIRK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (DIRKStep, tableau)

for alias, tableau in FIRK_TABLEAU_REGISTRY.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (FIRKStep, tableau)

for alias, tableau in ROSENBROCK_TABLEAUS.items():
    _TABLEAU_REGISTRY_BY_ALGORITHM[alias] = (
        GenericRosenbrockWStep,
        tableau,
    )


def resolve_alias(alias: str) -> Tuple[Type[BaseAlgorithmStep], Optional[ButcherTableau]]:
    """Return the step constructor and tableau associated with ``alias``."""

    key = alias.lower()
    if key not in _TABLEAU_REGISTRY_BY_ALGORITHM:
        raise KeyError(alias)
    return _TABLEAU_REGISTRY_BY_ALGORITHM[key]


def resolve_supplied_tableau(
    tableau: ButcherTableau,
) -> Tuple[Type[BaseAlgorithmStep], ButcherTableau]:
    """Return the step constructor matching ``tableau``."""

    if isinstance(tableau, ERKTableau):
        return ERKStep, tableau
    if isinstance(tableau, DIRKTableau):
        return DIRKStep, tableau
    if isinstance(tableau, FIRKTableau):
        return FIRKStep, tableau
    if isinstance(tableau, RosenbrockTableau):
        return GenericRosenbrockWStep, tableau
    raise TypeError(
        "Received tableau of type "
        f"{type(tableau).__name__} which does not match known algorithms."
    )


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
        Floating-point dtype used when compiling the step implementation.
    settings
        Mapping of settings applied to the algorithm. Must include
        ``"algorithm"`` and can contain any keywords from
        ``ALL_ALGORITHM_STEP_PARAMETERS``.
    warn_on_unused
        If ``True``, issue a warning for settings that the selected algorithm
        does not accept.
    **kwargs
        Additional keywords from ``ALL_ALGORITHM_STEP_PARAMETERS``. These
        override entries provided in ``settings``.

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

    algorithm_settings = {}
    if settings is not None:
        algorithm_settings.update(settings)
    algorithm_settings.update(kwargs)

    algorithm_value = algorithm_settings.pop("algorithm", None)
    if algorithm_value is None:
        raise ValueError("Algorithm settings must include 'algorithm'.")

    if isinstance(algorithm_value, str):
        try:
            algorithm_type, resolved_tableau = resolve_alias(algorithm_value)
        except KeyError as exc:
            raise ValueError(f"Unknown algorithm '{algorithm_value}'.") from exc
    elif isinstance(algorithm_value, ButcherTableau):
        algorithm_type, resolved_tableau = resolve_supplied_tableau(
            algorithm_value
        )
    else:
        raise TypeError(
            "Expected algorithm name or ButcherTableau instance, "
            f"received {type(algorithm_value).__name__}."
        )

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

    if resolved_tableau is not None:
        filtered["tableau"] = resolved_tableau

    return algorithm_type(**filtered)
