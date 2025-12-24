"""Fully implicit Runge--Kutta tableau definitions."""

from typing import Dict

import attrs
import numpy as np

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


def compute_embedded_weights_radauIIA(c, order=None):
    """
    Compute embedded weights b* for Radau IIA given collocation nodes c.

    Uses moment conditions: sum(b*_i * c_i^(k-1)) = 1/k for k=1..order

    Parameters:
    -----------
    c : array_like, shape (s,)
        Collocation nodes
    order : int, optional
        Order of embedded method (must be <= s). If None, uses s.

    Returns:
    --------
    b_star : ndarray, shape (s,)
        Embedded weights
    """
    c = np.asarray(c)
    s = len(c)

    if order is None:
        order = s
    if order > s:
        raise ValueError(f"Cannot achieve order {order} with {s} stages")

    # Build Vandermonde-like system: M[k-1,i] = c[i]^(k-1)
    M = np.vander(c, N=order, increasing=True).T

    # RHS: 1/k for k=1..order
    r = np.array([1.0 / k for k in range(1, order + 1)])

    # Solve (use lstsq for underdetermined case)
    if order == s:
        b_star = np.linalg.solve(M, r)
    else:
        b_star = np.linalg.lstsq(M, r, rcond=None)[0]

    return b_star


# Hairer's RadauIIA5 collocation nodes
c = np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1.0])

# Main weights (order 5)
b = np.array([(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1.0 / 9])

# Compute embedded weights (choose order 2 or 3)
b_star = compute_embedded_weights_radauIIA(c, order=2)


# Radau IIA 5th-order method (3 stages)
SQRT6 = 6 ** 0.5
RADAU_IIA_5_c = ((4 - SQRT6) / 10.0, (4 + SQRT6) / 10.0, 1.0)
RADAU_IIA_5_b_hat = compute_embedded_weights_radauIIA(RADAU_IIA_5_c,
                                                      order=2).tolist()

# print("b_hat =", RADAU_IIA_5_b_hat)

RADAU_IIA_5_TABLEAU = FIRKTableau(
    a=(
        ((88 - 7 * SQRT6) / 360.0, (296 - 169 * SQRT6) / 1800.0, (-2 + 3 * SQRT6) / 225.0),
        ((296 + 169 * SQRT6) / 1800.0, (88 + 7 * SQRT6) / 360.0, (-2 - 3 * SQRT6) / 225.0),
        ((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    ),
    b=((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    b_hat=tuple(RADAU_IIA_5_b_hat),
    c=((4 - SQRT6) / 10.0, (4 + SQRT6) / 10.0, 1.0),
    order=5,
)

DEFAULT_FIRK_TABLEAU = GAUSS_LEGENDRE_2_TABLEAU


FIRK_TABLEAU_REGISTRY: Dict[str, FIRKTableau] = {
    "firk_gauss_legendre_2": GAUSS_LEGENDRE_2_TABLEAU,
    "radau_iia_5": RADAU_IIA_5_TABLEAU,
    "radau": RADAU_IIA_5_TABLEAU,
}
