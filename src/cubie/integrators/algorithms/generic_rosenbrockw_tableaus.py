"""Rosenbrock-W method tableaus and registry utilities."""

from math import sqrt
from typing import Dict, Tuple

import attrs

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.define(frozen=True)
class RosenbrockTableau(ButcherTableau):
    """Coefficient tableau describing a Rosenbrock-W integration scheme.

    Parameters
    ----------
    a
        Lower-triangular matrix of stage coupling coefficients.
    b
        Weights applied to the stage increments when forming the solution.
    c
        Stage abscissae expressed as fractions of the step size.
    order
        Classical order of the Rosenbrock-W method.
    b_hat
        Optional embedded weights that deliver an error estimate.
    C
        Lower-triangular matrix containing Jacobian update coefficients.
    gamma
        Diagonal shift applied to the stage Jacobian solves.
    gamma_stages
        Optional per-stage diagonal shifts applied to the Jacobian solves.

    """

    C: Tuple[Tuple[float, ...], ...] = attrs.field(factory=tuple)
    gamma: float = attrs.field(default=0.25)
    gamma_stages: Tuple[float, ...] = attrs.field(factory=tuple)

    def typed_gamma_stages(
        self,
        numba_precision: type,
    ) -> Tuple[float, ...]:
        """Return stage-specific gamma shifts typed to ``numba_precision``."""

        return self.typed_vector(self.gamma_stages, numba_precision)


def _ros3p_tableau() -> RosenbrockTableau:
    """Return the three-stage third-order ROS3P tableau."""

    gamma = 0.5 + sqrt(3.0) / 6.0
    igamma = 1.0 / gamma
    c_matrix = (
        (0.0, 0.0, 0.0),
        (-igamma ** 2, 0.0, 0.0),
        (
            -igamma * (1.0 + igamma * (2.0 - 0.5 * igamma)),
            -igamma * (2.0 - 0.5 * igamma),
            0.0,
        ),
    )
    b_aux = igamma * (2.0 / 3.0 - (1.0 / 6.0) * igamma)
    tableau = RosenbrockTableau(
        a=(
            (0.0, 0.0, 0.0),
            (igamma, 0.0, 0.0),
            (igamma, 0.0, 0.0),
        ),
        C=c_matrix,
        b=(
            igamma * (1.0 + b_aux),
            b_aux,
            igamma / 3.0,
        ),
        b_hat=(
            2.113248654051871,
            1.0,
            0.4226497308103742,
        ),
        c=(0.0, 1.0, 1.0),
        order=3,
        gamma=gamma,
        gamma_stages=(gamma, -0.2113248654051871, 1/2 - 2*gamma),
    )
    return tableau


# ROSENBROCK_W6S4OS_TABLEAU = RosenbrockTableau(
#     ...existing code...
# )

ROS3P_TABLEAU = _ros3p_tableau()

# Rosenbrock 2(3) method used by MATLAB ode23s (Shampine & Reichelt, 1997)
r23_gamma = 1.0 / (2.0 + 2.0 ** 0.5)
ROSENBROCK_23_TABLEAU = RosenbrockTableau(
    a=(
        (0.0, 0.0),
        (1.0, 0.0),
    ),
    C=(
        (1.0 / (2.0 + 2.0 ** 0.5), 0.0),
        (-1.0 / (2.0 + 2.0 ** 0.5), 1.0 / (2.0 + 2.0 ** 0.5)),
    ),
    b=(0.5, 0.5),
    b_hat=(1.0, 0.0),
    c=(0.0, 1.0),
    order=2,
    gamma=1.0 / (2.0 + 2.0 ** 0.5),
    gamma_stages=(r23_gamma,r23_gamma),
)

ROSENBROCK_TABLEAUS: Dict[str, RosenbrockTableau] = {
    "ros3p": ROS3P_TABLEAU,
    "rosenbrock23": ROSENBROCK_23_TABLEAU,
    "ode23s": ROSENBROCK_23_TABLEAU,
    # "rosenbrock_w6s4os": ROSENBROCK_W6S4OS_TABLEAU,
}

DEFAULT_ROSENBROCK_TABLEAU_NAME = "ros3p"
DEFAULT_ROSENBROCK_TABLEAU = ROSENBROCK_TABLEAUS[
    DEFAULT_ROSENBROCK_TABLEAU_NAME
]


__all__ = [
    "RosenbrockTableau",
    # "ROSENBROCK_W6S4OS_TABLEAU",
    "ROS3P_TABLEAU",
    "ROSENBROCK_TABLEAUS",
    "DEFAULT_ROSENBROCK_TABLEAU",
    "DEFAULT_ROSENBROCK_TABLEAU_NAME",
]
