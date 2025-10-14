"""Tableaus for diagonally implicit Runge--Kutta (DIRK) methods."""

from typing import Dict, Tuple

import attrs

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
coefficient :math:`1 - \tfrac{1}{\sqrt{2}}`. The embedded weights provide
an error estimate suitable for adaptive step controllers.

References
----------
Alexander, R. (1977). Diagonally implicit Runge--Kutta methods for
stiff ODEs. *SIAM Journal on Numerical Analysis*, 14(6), 1006-1021.
"""

DIRK_TABLEAU_REGISTRY: Dict[str, DIRKTableau] = {
    "implicit_midpoint": IMPLICIT_MIDPOINT_TABLEAU,
    "trapezoidal_dirk": TRAPEZOIDAL_DIRK_TABLEAU,
    "lobatto_iiic_3": LOBATTO_IIIC_3_TABLEAU,
    "sdirk_2_2": SDIRK_2_2_TABLEAU,
}
"""Registry of named DIRK tableaus available to the integrator."""

DEFAULT_DIRK_TABLEAU_NAME = "sdirk_2_2"
DEFAULT_DIRK_TABLEAU = DIRK_TABLEAU_REGISTRY[DEFAULT_DIRK_TABLEAU_NAME]
