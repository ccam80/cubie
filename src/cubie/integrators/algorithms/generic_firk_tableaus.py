"""Fully implicit Runge--Kutta tableau definitions."""

from typing import Dict

import attrs

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.define(frozen=True)
class FIRKTableau(ButcherTableau):
    """Coefficient tableau describing a fully implicit RK scheme."""


SQRT3 = 3 ** 0.5

GAUSS_LEGENDRE_2_TABLEAU = FIRKTableau(
    a=(
        (0.25, 0.25 - SQRT3 / 6.0),
        (0.25 + SQRT3 / 6.0, 0.25),
    ),
    b=(0.5, 0.5),
    c=(0.5 - SQRT3 / 6.0, 0.5 + SQRT3 / 6.0),
    order=4,
)

DEFAULT_FIRK_TABLEAU = GAUSS_LEGENDRE_2_TABLEAU


FIRK_TABLEAU_REGISTRY: Dict[str, FIRKTableau] = {
    "firk_gauss_legendre_2": GAUSS_LEGENDRE_2_TABLEAU,
}

