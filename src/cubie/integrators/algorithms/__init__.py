"""Factories for explicit and implicit algorithm step implementations."""

from typing import Any

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
]


def get_algorithm_step(name: str, **kwargs: Any) -> BaseAlgorithmStep:
    """Instantiate an algorithm step implementation.

    Parameters
    ----------
    name
        Identifier for the desired algorithm implementation. Supported
        values are ``"euler"``, ``"backwards_euler"``,
        ``"backwards_euler_pc"``, and ``"crank_nicolson"``.
    **kwargs
        Configuration parameters forwarded to the selected step class.

    Returns
    -------
    BaseAlgorithmStep
        Instance of the requested algorithm step implementation.

    Raises
    ------
    ValueError
        Raised when ``name`` does not match a known algorithm identifier.
    """

    if name == "euler":
        return ExplicitEulerStep(**kwargs)
    if name == "backwards_euler":
        return BackwardsEulerStep(**kwargs)
    if name == "backwards_euler_pc":
        return BackwardsEulerPCStep(**kwargs)
    if name == "crank_nicolson":
        return CrankNicolsonStep(**kwargs)
    raise ValueError(f"Unknown algorithm '{name}'.")
