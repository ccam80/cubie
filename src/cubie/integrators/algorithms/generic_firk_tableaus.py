"""Fully implicit Runge–Kutta tableau definitions.

Published Classes
-----------------
:class:`FIRKTableau`
    Extends :class:`~base_algorithm_step.ButcherTableau` (no
    additional methods; serves as a type tag for dispatch).

Module-Level Functions
----------------------
:func:`compute_embedded_weights_radauIIA`
    Compute embedded weights for Radau IIA collocation nodes via
    moment conditions.

Constants
---------
:data:`GAUSS_LEGENDRE_2_TABLEAU`
    Two-stage, fourth-order Gauss–Legendre tableau.

:data:`RADAU_IIA_5_TABLEAU`
    Three-stage, fifth-order Radau IIA tableau with second-order
    embedded error estimate.

:data:`FIRK_TABLEAU_REGISTRY`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_FIRK_TABLEAU`
    Default tableau (Gauss–Legendre 2-stage).

See Also
--------
:class:`~cubie.integrators.algorithms.generic_firk.FIRKStep`
    Step factory consuming these tableaus.
:class:`~cubie.integrators.algorithms.base_algorithm_step.ButcherTableau`
    Parent tableau class.
"""

from typing import Dict

from attrs import define
from numpy import (
    array as np_array,
    asarray as np_asarray,
    linalg as np_linalg,
    sqrt as np_sqrt,
    vander as np_vander,
)

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@define
class FIRKTableau(ButcherTableau):
    """Coefficient tableau describing a fully implicit RK scheme."""


SQRT3 = np_sqrt(3)

GAUSS_LEGENDRE_2_TABLEAU = FIRKTableau(
    a=(
        (0.25, 0.25 - SQRT3 / 6.0),
        (0.25 + SQRT3 / 6.0, 0.25),
    ),
    b=(0.5, 0.5),
    c=(0.5 - SQRT3 / 6.0, 0.5 + SQRT3 / 6.0),
    order=4,
)


def compute_embedded_weights_radauIIA(c, order=None):
    """Compute embedded weights for Radau IIA collocation nodes.

    Solves the moment conditions
    :math:`\\sum_i b^*_i \\, c_i^{k-1} = 1/k` for
    :math:`k = 1, \\ldots, \\text{order}`.

    Parameters
    ----------
    c : array_like, shape (s,)
        Collocation nodes.
    order : int, optional
        Order of the embedded method (must be ``<= s``). When
        ``None``, defaults to ``s``.

    Returns
    -------
    ndarray, shape (s,)
        Embedded weights satisfying the moment conditions.

    Raises
    ------
    ValueError
        If ``order`` exceeds the number of stages.
    """
    c = np_asarray(c)
    s = len(c)

    if order is None:
        order = s
    if order > s:
        raise ValueError(f"Cannot achieve order {order} with {s} stages")

    # Build Vandermonde-like system: M[k-1,i] = c[i]^(k-1)
    M = np_vander(c, N=order, increasing=True).T

    # RHS: 1/k for k=1..order
    r = np_array([1.0 / k for k in range(1, order + 1)])

    # Solve (use lstsq for underdetermined case)
    if order == s:
        b_star = np_linalg.solve(M, r)
    else:
        b_star = np_linalg.lstsq(M, r, rcond=None)[0]

    return b_star


SQRT6 = np_sqrt(6)
_RADAU_IIA_5_c = ((4 - SQRT6) / 10.0, (4 + SQRT6) / 10.0, 1.0)
_RADAU_IIA_5_b_hat = tuple(
    compute_embedded_weights_radauIIA(_RADAU_IIA_5_c, order=2).tolist()
)

RADAU_IIA_5_TABLEAU = FIRKTableau(
    a=(
        (
            (88 - 7 * SQRT6) / 360.0,
            (296 - 169 * SQRT6) / 1800.0,
            (-2 + 3 * SQRT6) / 225.0,
        ),
        (
            (296 + 169 * SQRT6) / 1800.0,
            (88 + 7 * SQRT6) / 360.0,
            (-2 - 3 * SQRT6) / 225.0,
        ),
        ((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    ),
    b=((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    b_hat=_RADAU_IIA_5_b_hat,
    c=_RADAU_IIA_5_c,
    order=5,
)

DEFAULT_FIRK_TABLEAU = GAUSS_LEGENDRE_2_TABLEAU


FIRK_TABLEAU_REGISTRY: Dict[str, FIRKTableau] = {
    "firk_gauss_legendre_2": GAUSS_LEGENDRE_2_TABLEAU,
    "radau_iia_5": RADAU_IIA_5_TABLEAU,
    "radau": RADAU_IIA_5_TABLEAU,
}
