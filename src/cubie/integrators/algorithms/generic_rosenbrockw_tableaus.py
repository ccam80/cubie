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
        gamma_stages=(gamma, gamma, gamma),
    )
    return tableau


ROSENBROCK_W6S4OS_TABLEAU = RosenbrockTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.5812383407115008,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.9039624413714670,
            1.8615191555345010,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.0765797196750000,
            0.1884255381414796,
            1.8701589674910320,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.4355506384843120,
            5.4571817986101890,
            4.6163507880689300,
            3.1181119524023610,
            0.0,
            0.0,
        ),
        (
            10.791701698483260,
            -10.056915225841310,
            14.995644854284190,
            5.2743399543909430,
            1.4297308712611900,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -2.661294105131369,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -3.128450202373838,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -6.920335474535658,
            -1.202675288266817,
            -9.733561811413620,
            0.0,
            0.0,
            0.0,
        ),
        (
            -28.095306291026950,
            20.371262954793770,
            -41.043752753028690,
            -19.663731756208950,
            0.0,
            0.0,
        ),
        (
            9.7998186780974000,
            11.935792886603180,
            3.6738749290132010,
            14.807828541095500,
            0.8318583998690680,
            0.0,
        ),
    ),
    b=(
        6.4562170746532350,
        -4.8531413177680530,
        9.7653183340692600,
        2.0810841772787230,
        0.6603936866352417,
        0.6000000000000000,
    ),
    b_hat=(
        6.2062170746532350,
        -4.9368104361973420,
        9.7108464717176250,
        2.4213131495143094,
        0.6266285278012887,
        0.6903074267618540,
    ),
    c=(
        0.0,
        0.1453095851778752,
        0.3817422770256738,
        0.6367813704374599,
        0.7560744496323561,
        0.9271047239875670,
    ),
    order=4,
    gamma=0.25,
)

ROS3P_TABLEAU = _ros3p_tableau()

ROSENBROCK_TABLEAUS: Dict[str, RosenbrockTableau] = {
    "ros3p": ROS3P_TABLEAU,
    "rosenbrock_w6s4os": ROSENBROCK_W6S4OS_TABLEAU,
}

DEFAULT_ROSENBROCK_TABLEAU_NAME = "ros3p"
DEFAULT_ROSENBROCK_TABLEAU = ROSENBROCK_TABLEAUS[
    DEFAULT_ROSENBROCK_TABLEAU_NAME
]


__all__ = [
    "RosenbrockTableau",
    "ROSENBROCK_W6S4OS_TABLEAU",
    "ROS3P_TABLEAU",
    "ROSENBROCK_TABLEAUS",
    "DEFAULT_ROSENBROCK_TABLEAU",
    "DEFAULT_ROSENBROCK_TABLEAU_NAME",
]
