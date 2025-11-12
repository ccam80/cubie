"""Tableaus for diagonally implicit Runge--Kutta (DIRK) methods."""

from typing import Dict, Tuple

import attrs
import math

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.define(frozen=True)
class DIRKTableau(ButcherTableau):
    """Coefficient tableau describing a diagonally implicit RK scheme.

    The tableau stores the Runge--Kutta coefficients required by
    diagonally implicit methods, including singly diagonally implicit
    (SDIRK) and explicit-first-stage diagonally implicit (ESDIRK) variants.

    Methods
    -------
    diagonal(precision)
        Return the diagonal elements of the :math:`A` matrix as a
        precision-typed tuple.

    References
    ----------
    Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
    Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
    Springer.
    """

    def diagonal(self, precision: type) -> Tuple[float, ...]:
        """Return the diagonal entries of the tableau."""

        diagonal_entries = tuple(
            self.a[idx][idx] for idx in range(self.stage_count)
        )
        return self.typed_vector(diagonal_entries, precision)


IMPLICIT_MIDPOINT_TABLEAU = DIRKTableau(
    a=((0.5,),),
    b=(1.0,),
    c=(0.5,),
    order=2,
)
"""DIRK tableau for the implicit midpoint rule (second order).

The method is singly diagonally implicit with a single stage whose
coefficient equals :math:`1/2`. It is symplectic and A-stable, making it
useful for Hamiltonian systems.

References
----------
Sanz-Serna, J. M. (1988). Runge--Kutta schemes for Hamiltonian systems.
*BIT Numerical Mathematics*, 28(4), 877-883.
"""

TRAPEZOIDAL_DIRK_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0),
        (0.5, 0.5),
    ),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
)
"""DIRK tableau for the Crank--Nicolson (trapezoidal) rule.

The first stage is explicit while the second stage is implicit, placing
this scheme in the ESDIRK family. It is A-stable and time-reversible,
which makes it a popular choice for moderately stiff problems.

References
----------
Crank, J., & Nicolson, P. (1947). A practical method for numerical
solution of partial differential equations of the heat-conduction type.
*Mathematical Proceedings of the Cambridge Philosophical Society*,
43(1), 50-67.
"""

LOBATTO_IIIC_3_TABLEAU = DIRKTableau(
    a=(
        (1.0 / 6.0, 0.0, 0.0),
        (2.0 / 3.0, 1.0 / 6.0, 0.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ),
    b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 1.0),
    order=4,
)
"""Three-stage Lobatto IIIC DIRK tableau of order four.

All stages share the same diagonal coefficient, so the tableau may be
solved sequentially without resorting to coupled implicit systems. The
method is symplectic and stiffly accurate, making it attractive for high
accuracy integrations.

References
----------
Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical
Integration* (2nd ed.). Springer.
"""

SQRT2 = 2 ** 0.5
SQRT6 = 6 ** 0.5

ARCTAN_TERM = math.atan(SQRT2 / 4.0) / 3.0
L_STABLE_DIRK3_GAMMA = (
    -SQRT2 * math.cos(ARCTAN_TERM) / 2.0
    + SQRT6 * math.sin(ARCTAN_TERM) / 2.0
    + 1.0
)
SDIRK_2_2_TABLEAU = DIRKTableau(
    a=(
        ((2.0 - SQRT2) / 2.0, 0.0),
        (1.0 - (2.0 - SQRT2) / 2.0, (2.0 - SQRT2) / 2.0),
    ),
    b=(0.5, 0.5),
    b_hat=(-0.5, 0.5),
    c=((2.0 - SQRT2) / 2.0, 1.0),
    order=2,
)
"""Two-stage, second-order SDIRK tableau by Alexander.

The tableau is L-stable and singly diagonally implicit with diagonal
coefficient :math:`1 - \\tfrac{1}{\\sqrt{2}}`. The embedded weights provide
an error estimate suitable for adaptive step controllers.

References
----------
Alexander, R. (1977). Diagonally implicit Runge--Kutta methods for
stiff ODEs. *SIAM Journal on Numerical Analysis*, 14(6), 1006-1021.
"""

L_STABLE_DIRK3_TABLEAU = DIRKTableau(
    a=(
        (L_STABLE_DIRK3_GAMMA, 0.0, 0.0),
        ((1.0 - L_STABLE_DIRK3_GAMMA) / 2.0, L_STABLE_DIRK3_GAMMA, 0.0),
        (
            (
                -6.0 * L_STABLE_DIRK3_GAMMA ** 2
                + 16.0 * L_STABLE_DIRK3_GAMMA
                - 1.0
            )
            / 4.0,
            (
                6.0 * L_STABLE_DIRK3_GAMMA ** 2
                - 20.0 * L_STABLE_DIRK3_GAMMA
                + 5.0
            )
            / 4.0,
            L_STABLE_DIRK3_GAMMA,
        ),
    ),
    b=(
        (
            -6.0 * L_STABLE_DIRK3_GAMMA ** 2
            + 16.0 * L_STABLE_DIRK3_GAMMA
            - 1.0
        )
        / 4.0,
        (
            6.0 * L_STABLE_DIRK3_GAMMA ** 2
            - 20.0 * L_STABLE_DIRK3_GAMMA
            + 5.0
        )
        / 4.0,
        L_STABLE_DIRK3_GAMMA,
    ),
    c=(
        L_STABLE_DIRK3_GAMMA,
        (1.0 + L_STABLE_DIRK3_GAMMA) / 2.0,
        1.0,
    ),
    order=3,
)
"""Three-stage, third-order L-stable DIRK method with stiff accuracy.

The tableau follows the coefficients published in MOOSE's
``LStableDirk3`` time integrator, derived from Alexander's family of
L-stable singly diagonally implicit schemes. All stages share the
diagonal value :math:`\\gamma`, and the last row equals the weight
vector, so the method is stiffly accurate.

References
----------
MOOSE Framework documentation. "LStableDirk3" time integrator.
https://mooseframework.inl.gov/source/timeintegrators/LStableDirk3.html
"""

QUARTER = 0.25
L_STABLE_SDIRK4_TABLEAU = DIRKTableau(
    a=(
        (QUARTER, 0.0, 0.0, 0.0, 0.0),
        (0.5, QUARTER, 0.0, 0.0, 0.0),
        (17.0 / 50.0, -1.0 / 25.0, QUARTER, 0.0, 0.0),
        (
            371.0 / 1360.0,
            137.0 / 2720.0,
            15.0 / 544.0,
            QUARTER,
            0.0,
        ),
        (
            25.0 / 24.0,
            -49.0 / 48.0,
            125.0 / 16.0,
            -85.0 / 12.0,
            QUARTER,
        ),
    ),
    b=(
        25.0 / 24.0,
        -49.0 / 48.0,
        125.0 / 16.0,
        -85.0 / 12.0,
        QUARTER,
    ),
    b_hat=(
        59.0 / 48.0,
        -17.0 / 96.0,
        225.0 / 32.0,
        -85.0 / 12.0,
        0.0
    ),
    c=(
        QUARTER,
        3.0 / 4.0,
        11.0 / 20.0,
        0.5,
        1.0,
    ),
    order=4,
)
"""Hairer--Wanner L-stable SDIRK tableau of order four.

The five-stage scheme delivers fourth-order accuracy with stiff
accuracy, reusing :math:`\\gamma = 1/4` on the diagonal. The tableau
matches the coefficients tabulated in Hairer and Wanner's *Solving
Ordinary Differential Equations II* (Table 6.5).

References
----------
Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
Springer.
"""

DIRK_TABLEAU_REGISTRY: Dict[str, DIRKTableau] = {
    "implicit_midpoint": IMPLICIT_MIDPOINT_TABLEAU,
    "trapezoidal_dirk": TRAPEZOIDAL_DIRK_TABLEAU,
    "ode23t": TRAPEZOIDAL_DIRK_TABLEAU,
    "lobatto_iiic_3": LOBATTO_IIIC_3_TABLEAU,
    "sdirk_2_2": SDIRK_2_2_TABLEAU,
    "l_stable_dirk_3": L_STABLE_DIRK3_TABLEAU,
    "l_stable_sdirk_4": L_STABLE_SDIRK4_TABLEAU,
}
"""Registry of named DIRK tableaus available to the integrator."""

DEFAULT_DIRK_TABLEAU_NAME = "lobatto_iiic_3"
DEFAULT_DIRK_TABLEAU = DIRK_TABLEAU_REGISTRY[DEFAULT_DIRK_TABLEAU_NAME]
