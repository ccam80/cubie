"""Algorithm step implementations."""

from typing import Optional, Callable

from .base_algorithm_step import BaseStepConfig, BaseAlgorithmStep
from .ode_explicitstep import ExplicitStepConfig, ODEExplicitStep
from .ode_implicitstep import ImplicitStepConfig, ODEImplicitStep
from .explicit_euler import ExplicitEulerStep
from .backwards_euler import BackwardsEulerStep
from .crank_nicolson import CrankNicolsonStep
from .backwards_euler_predict_correct import BackwardsEulerPCStep


__all__ = [
    "get_algorithm_step"
    "ExplicitStepConfig",
    "ImplicitStepConfig",
    "ExplicitEulerStep",
    "BackwardsEulerStep",
    "BackwardsEulerPCStep",
    "CrankNicolsonStep",
]

def get_algorithm_step(name,
                       **kwargs) -> BaseAlgorithmStep:
    """Return an algorith step instance based on ``name`` with parameters
    given by kwargs"""
    if name == "euler":
        return ExplicitEulerStep(**kwargs)
    elif name == "backwards_euler":
        return BackwardsEulerStep(**kwargs)
    elif name == "backwards_euler_pc":
        return BackwardsEulerPCStep(**kwargs)
    elif name == "crank_nicolson":
        return CrankNicolsonStep(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm '{name}'.")