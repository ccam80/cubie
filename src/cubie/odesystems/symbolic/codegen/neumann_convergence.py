"""Check whether the Neumann series preconditioner will converge.

The Neumann series approximates ``(betaI - gamma h (A (x) J))^-1`` via
the truncated expansion ``beta^-1 (I + T + T^2 + ...)`` where
``T = (gamma h / beta)(A (x) J)``. This converges if and only if the
spectral radius ``rho(T) < 1``.

Since ``T`` scales linearly with ``h``, a more fundamental
(h-independent) diagnostic is the spectral radius of the Jacobi
iteration matrix ``N = I - D^-1 J`` where ``D = diag(J)``; if
``rho(N) >= 1`` the ODE Jacobian is not diagonally dominant and the
Neumann series diverges for all but the tiniest step sizes.

Published Functions
-------------------
:func:`check_neumann_convergence`
    Evaluate convergence diagnostics for the Neumann preconditioner and
    emit a warning when divergence is likely.
"""

import logging
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
import sympy as sp

from cubie.odesystems.symbolic.codegen.jacobian import generate_jacobian
from cubie.odesystems.symbolic.parsing.parser import (
    ParsedEquations,
    TIME_SYMBOL,
)
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.sym_utils import topological_sort

logger = logging.getLogger(__name__)


def check_neumann_convergence(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, sp.Expr]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, sp.Expr]]] = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    t0: float = 0.0,
) -> Dict[str, object]:
    """Evaluate whether the Neumann preconditioner is likely to converge.

    Computes the symbolic ODE Jacobian, substitutes initial-state and
    constant values, and checks diagonal dominance / spectral radius of
    the Jacobi iteration matrix ``N = I - D^-1 J``.

    Parameters
    ----------
    equations
        Parsed ODE equations (the same object passed to codegen).
    index_map
        Index maps (states, constants, etc.) with default values.
    stage_coefficients
        Butcher-tableau ``A`` matrix for the FIRK method. When provided,
        the full staged system ``betaI - gamma(A (x) J)`` is analysed
        (un-scaled by ``h`` so the result is h-independent).
    stage_nodes
        Butcher-tableau ``c`` nodes (unused for convergence, reserved
        for future driver interpolation).
    beta, gamma
        Transformation parameters from the FIRK formulation.
    t0
        Time at which to evaluate the Jacobian (default 0).

    Returns
    -------
    dict
        ``rho_N`` -- spectral radius of the Jacobi iteration matrix.
        ``max_ratio`` -- worst-case row off-diag/diag ratio.
        ``worst_rows`` -- list of (row_index, state_name, ratio) tuples
        for rows where ratio > 1.
        ``converges`` -- ``True`` if ``rho(N) < 1``.
        ``J_numeric`` -- the evaluated numeric Jacobian (for debugging).
    """
    # 1. Symbolic Jacobian.
    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
    )
    state_count = len(index_map.states.index_map)

    # 2. Substitution map: states -> defaults, constants -> values.
    subs = {}
    for sym, default in index_map.states.default_values.items():
        subs[sym] = float(default)
    for sym, default in index_map.constants.default_values.items():
        subs[sym] = float(default)
    subs[TIME_SYMBOL] = float(t0)
    if hasattr(index_map, "parameters"):
        for sym, default in index_map.parameters.default_values.items():
            subs[sym] = float(default)
    if hasattr(index_map, "drivers"):
        for sym in index_map.drivers.index_map:
            subs[sym] = 0.0

    # 3. Evaluate the Jacobian numerically. Jacobian entries reference
    # auxiliary variables defined in the equation list; evaluate all
    # auxiliaries in topological order so every intermediate resolves
    # before we hit the J entries.
    eq_list = equations.to_equation_list()
    sorted_eqs = topological_sort(eq_list)
    eval_subs = dict(subs)
    dx_symbols = set(index_map.dxdt.index_map.keys())
    for lhs, rhs in sorted_eqs:
        if lhs in dx_symbols:
            continue
        try:
            val = complex(rhs.xreplace(eval_subs))
            eval_subs[lhs] = val.real if val.imag == 0 else val
        except (TypeError, ValueError, AttributeError):
            pass  # skip if it cannot be evaluated

    n_failed = 0
    J_num = np.zeros((state_count, state_count), dtype=np.float64)
    for i in range(state_count):
        for j in range(state_count):
            entry = jac[i, j]
            if entry is sp.S.Zero or entry == 0:
                continue
            try:
                val = complex(entry.xreplace(eval_subs))
                J_num[i, j] = val.real
            except (TypeError, ValueError, AttributeError):
                # Treat unevaluable entries as large (conservative).
                J_num[i, j] = 1e30
                n_failed += 1
    if n_failed > 0:
        logger.warning(
            "Could not evaluate %d Jacobian entries numerically; "
            "treating them as large for convergence check.",
            n_failed,
        )

    # 4. Build the system matrix and analyse. Map row/col indices back
    # to state names for reporting.
    idx_to_name = {
        v: str(k) for k, v in index_map.states.index_map.items()
    }

    if stage_coefficients is not None:
        # Full h-independent staged matrix M_0 = beta*I - gamma*(A (x) J)
        # (factor out h so the diagnostic is step-size independent).
        A = np.array(
            [[float(c) for c in row] for row in stage_coefficients],
            dtype=np.float64,
        )
        s = A.shape[0]
        n = state_count
        M_0 = beta * np.eye(s * n) - gamma * np.kron(A, J_num)
        D_0 = np.diag(M_0)
    else:
        # Single-stage fallback: M_0 = beta*I - gamma*J.
        n = state_count
        s = 1
        M_0 = beta * np.eye(n) - gamma * J_num
        D_0 = np.diag(M_0)

    sn = s * n

    # Guard against zero diagonal entries.
    D_safe = np.where(np.abs(D_0) > 1e-30, D_0, 1e-30)
    D_inv = np.diag(1.0 / D_safe)

    # Jacobi iteration matrix: N = I - D^-1 * M_0.
    N = np.eye(sn) - D_inv @ M_0
    eigvals = np.linalg.eigvals(N)
    rho_N = float(np.max(np.abs(eigvals)))

    # Per-row diagonal dominance ratio.
    abs_M = np.abs(M_0)
    diag_abs = np.abs(D_0)
    off_diag_sum = np.sum(abs_M, axis=1) - diag_abs
    ratios = off_diag_sum / np.maximum(diag_abs, 1e-30)
    max_ratio = float(np.max(ratios))

    # Identify worst rows (map back to state names for staged system).
    worst_rows = []
    for idx in range(sn):
        if ratios[idx] > 1.0:
            stage_idx = idx // n
            state_idx = idx % n
            name = idx_to_name.get(state_idx, f"state_{state_idx}")
            label = f"stage{stage_idx}:{name}" if s > 1 else name
            worst_rows.append((idx, label, float(ratios[idx])))
    worst_rows.sort(key=lambda x: -x[2])

    converges = rho_N < 1.0

    result = {
        "rho_N": rho_N,
        "max_ratio": max_ratio,
        "worst_rows": worst_rows,
        "converges": converges,
        "J_numeric": J_num,
    }

    # 5. Emit a warning if divergent.
    if not converges:
        n_bad = len(worst_rows)
        top_rows = worst_rows[:5]
        row_details = ", ".join(
            f"{label} (ratio={r:.1f})" for _, label, r in top_rows
        )
        msg = (
            "Neumann preconditioner will likely DIVERGE for this "
            f"system. Spectral radius rho(I - D^-1 M) = {rho_N:.2f} "
            ">= 1.0 (need < 1 for convergence). "
            f"{n_bad}/{sn} rows violate diagonal dominance. "
            f"Worst: [{row_details}]. Consider a diagonal (Jacobi) "
            "preconditioner or a smaller step size."
        )
        warnings.warn(msg, stacklevel=3)
        logger.warning(msg)

    return result
