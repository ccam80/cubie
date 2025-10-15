"""Explicit Runge--Kutta tableau definitions used by generic ERK steps.

The tableaus collected here provide reusable coefficients for explicit
Runge--Kutta methods with classical order of at least two. Each tableau lists
the literature source describing the coefficients so integrators can reference
the original derivations when validating implementations.
"""

from typing import Dict

import attrs

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.define(frozen=True)
class ERKTableau(ButcherTableau):
    """Coefficient tableau describing an explicit Runge--Kutta scheme."""


#: Heun's improved Euler method offering second-order accuracy.
#: Heun, K. "Neue Methoden zur approximativen Integration der
#: Differentialgleichungen einer unabhängigen Veränderlichen." *Z.
#: Math. Phys.* 45 (1900).
HEUN_21_TABLEAU = ERKTableau(
    a=((0.0, 0.0), (1.0, 0.0)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
)

#: Ralston's third-order, three-stage explicit Runge--Kutta method.
#: Ralston, A. "Runge-Kutta methods with minimum error bounds." *Math. Comp.*
#: 16.80 (1962).
RALSTON_33_TABLEAU = ERKTableau(
    a=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.75, 0.0)),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0),
    c=(0.0, 1.0 / 2.0, 3.0 / 4.0),
    order=3,
)

#: Bogacki--Shampine 3(2) tableau with an embedded error estimate.
#: Bogacki, P. and Shampine, L. F. "An efficient Runge-Kutta (4,5) pair." *J.
#: Comput. Appl. Math.* 46.1 (1993).
BOGACKI_SHAMPINE_32_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.75, 0.0, 0.0),
        (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    ),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    b_hat=(7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0),
    c=(0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0),
    order=3,
)

#: Dormand--Prince 5(4) tableau with an embedded error estimate.
#: Dormand, J. R. and Prince, P. J. "A family of embedded Runge-Kutta
#: formulae." *Journal of Computational and Applied Mathematics* 6.1 (1980).
DORMAND_PRINCE_54_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            44.0 / 45.0,
            -56.0 / 15.0,
            32.0 / 9.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
            0.0,
        ),
        (
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ),
    ),
    b=(
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ),
    b_hat=(
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ),
    c=(
        0.0,
        1.0 / 5.0,
        3.0 / 10.0,
        4.0 / 5.0,
        8.0 / 9.0,
        1.0,
        1.0,
    ),
    order=5,
)

#: Classical four-stage Runge--Kutta method introduced by Kutta (1901).
#: Kutta, W. "Beitrag zur näherungsweisen Integration totaler
#: Differentialgleichungen." *Zeitschrift für Mathematik und Physik* 46 (1901).
CLASSICAL_RK4_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (1.0 / 2.0, 0.0, 0.0, 0.0),
        (0.0, 1.0 / 2.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    ),
    b=(
        1.0 / 6.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 6.0,
    ),
    c=(0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0),
    order=4,
)

#: Cash--Karp 5(4) tableau with an embedded error estimate.
#: Cash, J. R. and Karp, A. H. "A variable order Runge-Kutta method for initial
#: value problems with rapidly varying right-hand sides." *ACM Transactions on
#: Mathematical Software* 16.3 (1990).
CASH_KARP_54_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0),
        (
            3.0 / 10.0,
            -9.0 / 10.0,
            6.0 / 5.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -11.0 / 54.0,
            5.0 / 2.0,
            -70.0 / 27.0,
            35.0 / 27.0,
            0.0,
            0.0,
        ),
        (
            1631.0 / 55296.0,
            175.0 / 512.0,
            575.0 / 13824.0,
            44275.0 / 110592.0,
            253.0 / 4096.0,
            0.0,
        ),
    ),
    b=(
        37.0 / 378.0,
        0.0,
        250.0 / 621.0,
        125.0 / 594.0,
        0.0,
        512.0 / 1771.0,
    ),
    b_hat=(
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        1.0 / 4.0,
    ),
    c=(
        0.0,
        1.0 / 5.0,
        3.0 / 10.0,
        3.0 / 5.0,
        1.0,
        7.0 / 8.0,
    ),
    order=5,
)

#: Runge--Kutta--Fehlberg 5(4) tableau with an embedded error estimate.
#: Fehlberg, E. "Low-order classical Runge-Kutta formulas with stepsize control
#: and their application to some heat transfer problems." *NASA Technical
#: Report* 315 (1969).
FEHLBERG_45_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0),
        (
            1932.0 / 2197.0,
            -7200.0 / 2197.0,
            7296.0 / 2197.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            439.0 / 216.0,
            -8.0,
            3680.0 / 513.0,
            -845.0 / 4104.0,
            0.0,
            0.0,
        ),
        (
            -8.0 / 27.0,
            2.0,
            -3544.0 / 2565.0,
            1859.0 / 4104.0,
            -11.0 / 40.0,
            0.0,
        ),
    ),
    b=(
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    ),
    b_hat=(
        25.0 / 216.0,
        0.0,
        1408.0 / 2565.0,
        2197.0 / 4104.0,
        -1.0 / 5.0,
        0.0,
    ),
    c=(
        0.0,
        1.0 / 4.0,
        3.0 / 8.0,
        12.0 / 13.0,
        1.0,
        1.0 / 2.0,
    ),
    order=5,
)

DEFAULT_ERK_TABLEAU = DORMAND_PRINCE_54_TABLEAU
"""Default tableau used when constructing generic ERK integrators."""

ERK_TABLEAU_REGISTRY: Dict[str, ERKTableau] = {
    "heun-21": HEUN_21_TABLEAU,
    "ralston-33": RALSTON_33_TABLEAU,
    "bogacki-shampine-32": BOGACKI_SHAMPINE_32_TABLEAU,
    "dormand-prince-54": DORMAND_PRINCE_54_TABLEAU,
    "dopri54": DORMAND_PRINCE_54_TABLEAU,
    "classical-rk4": CLASSICAL_RK4_TABLEAU,
    "cash-karp-54": CASH_KARP_54_TABLEAU,
    "fehlberg-45": FEHLBERG_45_TABLEAU,
}
"""Mapping from human readable identifiers to ERK tableaus."""
