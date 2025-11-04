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

#: Dormand--Prince 8(5,3) explicit Runge--Kutta method (DOP853).
#: Hairer, Nørsett and Wanner (1993), "Solving Ordinary Differential Equations I", p. 178.
DORMAND_PRINCE_853_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.05260015195876773,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0197250569845379,
            0.0591751709536137,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (0.02958758547680685, 0.0, 0.08876275643042055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.2413651341592667,
            0.0,
            -0.8845494793282861,
            0.924834003261792,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.037037037037037035,
            0.0,
            0.0,
            0.17082860872947387,
            0.12546768756682242,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.037109375,
            0.0,
            0.0,
            0.17025221101954404,
            0.06021653898045596,
            -0.017578125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.03709200011850479,
            0.0,
            0.0,
            0.17038392571223999,
            0.10726203044637328,
            -0.015319437748624402,
            0.008273789163814023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.6241109587160757,
            0.0,
            0.0,
            -3.360892629446941,
            -0.868219346841726,
            27.592099699446706,
            20.154067550477893,
            -43.48988418106996,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.47766253643826435,
            0.0,
            0.0,
            -2.4881146199716676,
            -0.590290826836843,
            21.230051448181194,
            15.279233632882424,
            -33.28821096898486,
            -0.020331201708508627,
            0.0,
            0.0,
            0.0,
        ),
        (
            -0.9371424300859873,
            0.0,
            0.0,
            5.1863724288440636,
            1.0914373489967296,
            -8.149787010746926,
            -18.52006565999696,
            22.739487099350504,
            2.4936055526796526,
            -3.0467644718982197,
            0.0,
            0.0,
        ),
        (
            2.2733101475165383,
            0.0,
            0.0,
            -10.53449546673725,
            -2.0008720582248626,
            -17.958931863118799,
            27.94888452941996,
            -2.8589982771350237,
            -8.87285693353063,
            12.360567175794303,
            0.6433927460157635,
            0.0,
        ),
    ),
    b=(
        0.05429373411656876,
        0.0,
        0.0,
        0.0,
        0.0,
        4.450312892752409,
        1.8915178993145004,
        -5.801203960010585,
        0.3111643669578199,
        -0.15216094966251608,
        0.20136540080403035,
        0.04471061572777259,
    ),
    b_hat=(
        0.04117368912237387,
        0.0,
        0.0,
        0.0,
        0.0,
        5.675469339128613,
        2.3872768489717506,
        -7.465581142465572,
        0.6614932157077936,
        -0.4863400683755336,
        0.20136540080403035,
        0.04471061572777259,
    ),
    c=(
        0.0,
        0.05260015195876773,
        0.0789002279381516,
        0.1183503419072274,
        0.2816496580927726,
        0.3333333333333333,
        0.25,
        0.3076923076923077,
        0.6512820512820513,
        0.6,
        0.8571428571428571,
        1.0,
    ),
    order=8,
)

#: Tsitouras 5(4) tableau (Tsit5) with an embedded error estimate.
#: Tsitouras, Ch. (2011). "Runge–Kutta pairs of order 5(4) satisfying only the
#: first column simplifying assumption." Applied Numerical Mathematics, 56(10–11).
TSITOURAS_54_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.008480655492356989, 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0.0, 0.0, 0.0, 0.0),
        (5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0.0, 0.0, 0.0),
        (5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0.0, 0.0),
        (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0),
    ),
    b=(0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0),
    b_hat=(0.001780011052226, 0.000816434459657, -0.007880878010262, 0.144711007173263, -0.582357165452555, 0.458082105929187, 1.0/66.0),
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    order=5,
)

#: Verner 7(6) explicit Runge--Kutta tableau (Vern7) with embedded error estimate.
#: Verner, J. H. "Explicit Runge-Kutta methods with estimates of the local truncation error." (1997).
VERNER_76_TABLEAU = ERKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.0069931640625, 0.0876382652528226, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.03125, 0.0, 0.09375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.312, 0.0, -3.75, 3.75, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.04791013711111111, 0.0, 0.0, 0.2122707316666667, 0.9480738848888889, -0.4582536388888889, 0.0, 0.0, 0.0),
        (0.02660133245082994, 0.0, 0.0, 0.09736968614291808, 0.5331197042399694, 0.4417629791869485, -0.1, 0.0, 0.0),
        (0.03733650364325708, 0.0, 0.0, 0.17127704248123783, 0.6127033533509607, -0.1324544946682509, 0.3114709480497196, 0.0, 0.0),
        (0.04217361013027329, 0.0, 0.0, 0.0, 0.4831465145206436, -0.06410968794871542, 0.3215559617003662, 0.21736140686000956, 0.0),
    ),
    b=(0.04217361013027329, 0.0, 0.0, 0.0, 0.4831465145206436, -0.06410968794871542, 0.3215559617003662, 0.21736140686000956, 0.0),
    b_hat=(0.04589409398360933, 0.0, 0.0, 0.0, 0.48029210080816724, -0.05664509822456289, 0.32064601332793096, 0.19612113201708098, 0.01369185715374882),
    c=(0.0, 0.05, 0.08064516129032258, 0.125, 0.3125, 0.75, 0.9, 1.0, 1.0),
    order=7,
)

DEFAULT_ERK_TABLEAU = DORMAND_PRINCE_54_TABLEAU
"""Default tableau used when constructing generic ERK integrators."""

ERK_TABLEAU_REGISTRY: Dict[str, ERKTableau] = {
    "heun-21": HEUN_21_TABLEAU,
    "ralston-33": RALSTON_33_TABLEAU,
    "bogacki-shampine-32": BOGACKI_SHAMPINE_32_TABLEAU,
    "rk23": BOGACKI_SHAMPINE_32_TABLEAU,
    "ode23": BOGACKI_SHAMPINE_32_TABLEAU,
    "dormand-prince-54": DORMAND_PRINCE_54_TABLEAU,
    "dopri54": DORMAND_PRINCE_54_TABLEAU,
    "rk45": DORMAND_PRINCE_54_TABLEAU,
    "ode45": DORMAND_PRINCE_54_TABLEAU,
    "classical-rk4": CLASSICAL_RK4_TABLEAU,
    "cash-karp-54": CASH_KARP_54_TABLEAU,
    "fehlberg-45": FEHLBERG_45_TABLEAU,
    "dormand-prince-853": DORMAND_PRINCE_853_TABLEAU,
    "dop853": DORMAND_PRINCE_853_TABLEAU,
    "tsit5": TSITOURAS_54_TABLEAU,
    "Tsit5": TSITOURAS_54_TABLEAU,
    "vern7": VERNER_76_TABLEAU,
    "Vern7": VERNER_76_TABLEAU,
}
"""Mapping from human readable identifiers to ERK tableaus."""
