"""Runtime configuration settings for numerical integration algorithms.

This module provides the :class:`IntegratorRunSettings` class which manages
timing, tolerance, and controller configuration for ODE integration runs.
It performs light dependency injection by instantiating algorithm step
objects and step-size controllers used by the modular IVP loop.
"""

from __future__ import annotations

from typing import Dict

import attrs
from attrs import setters
import numba
import numpy as np
from numpy import float32

# from cubie.integrators.algorithms.LoopStepConfig import LoopStepConfig


_ALGORITHM_ALIASES: Dict[str, str] = {
    "euler": "explicit_euler",
    "explicit_euler": "explicit_euler",
    "explicit": "explicit_euler",
    "backward_euler": "backwards_euler",
    "backwards_euler": "backwards_euler",
    "crank_nicolson": "crank_nicolson",
}

_CONTROLLER_ALIASES: Dict[str, str] = {
    "fixed": "fixed",
    "i": "i",
    "pi": "pi",
    "pid": "pid",
    "gustafsson": "gustafsson",
}

_KNOWN_ALGORITHM_KEYS: set[str] = {
    "dt",
    "linsolve_tolerance",
    "max_linear_iters",
    "linear_correction_type",
    "nonlinear_tolerance",
    "max_newton_iters",
    "newton_damping",
    "newton_max_backtracks",
    "norm_type",
    "preconditioner_order",
}

_KNOWN_CONTROLLER_KEYS: set[str] = {
    "algorithm_order",
    "atol",
    "dt",
    "dt_max",
    "dt_min",
    "kd",
    "ki",
    "kp",
    "max_gain",
    "min_gain",
    "norm",
    "norm_kwargs",
    "order",
    "rtol",
}


def _normalise_algorithm(name: str) -> str:
    """Return the canonical algorithm identifier."""

    try:
        return _ALGORITHM_ALIASES[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown integrator algorithm '{name}'.") from exc


def _normalise_controller(kind: str) -> str:
    """Return the canonical controller identifier."""

    try:
        return _CONTROLLER_ALIASES[kind.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown step controller '{kind}'.") from exc


@attrs.define
class IntegratorRunSettings:
    """Container for runtime/timing settings grouped for IVP loops.

    Parameters
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller_kind
        Name of the step-size controller.
    """

    precision: type = attrs.field(
        default=float32,
        validator=attrs.validators.in_([np.float32, np.float64, np.float16]),
    )
    algorithm: str = attrs.field(
        default="explicit_euler",
        converter=_normalise_algorithm,
        on_setattr=setters.convert,
    )
    step_controller_kind: str = attrs.field(
        default="fixed",
        converter=_normalise_controller,
        on_setattr=setters.convert,
    )

    def __attrs_post_init__(self) -> None:
        """Validate configuration after initialisation."""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def numba_precision(self) -> type:
        """Return the Numba-compatible precision."""

        return numba.from_dtype(self.precision)
