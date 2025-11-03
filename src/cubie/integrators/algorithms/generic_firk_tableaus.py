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

# Radau IIA 5th-order method (3 stages)
SQRT6 = 6 ** 0.5
RADAU_IIA_5_TABLEAU = FIRKTableau(
    a=(
        ((88 - 7 * SQRT6) / 360.0, (296 - 169 * SQRT6) / 1800.0, (-2 + 3 * SQRT6) / 225.0),
        ((296 + 169 * SQRT6) / 1800.0, (88 + 7 * SQRT6) / 360.0, (-2 - 3 * SQRT6) / 225.0),
        ((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    ),
    b=((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    b_hat=(-1.0530884977290216, 0.4222222222222222, 3.6308662755068006),
    c=((4 - SQRT6) / 10.0, (4 + SQRT6) / 10.0, 1.0),
    order=5,
)

DEFAULT_FIRK_TABLEAU = GAUSS_LEGENDRE_2_TABLEAU


FIRK_TABLEAU_REGISTRY: Dict[str, FIRKTableau] = {
    "firk_gauss_legendre_2": GAUSS_LEGENDRE_2_TABLEAU,
    "radau_iia_5": RADAU_IIA_5_TABLEAU,
    "radau": RADAU_IIA_5_TABLEAU,
}
