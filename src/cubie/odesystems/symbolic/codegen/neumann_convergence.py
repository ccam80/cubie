"""Check whether the Neumann series preconditioner will converge.

The Neumann series approximates ``(betaI - gamma h (A (x) J))^-1`` via
the truncated expansion ``beta^-1 (I + T + T^2 + ...)`` where
``T = (gamma h / beta)(A (x) J)``. This converges if and only if the
spectral radius ``rho(T) < 1``.

Since ``T`` scales linearly with ``h``, a more fundamental
(h-independent) diagnostic is the spectral radius of the Jacobi
iteration matrix ``N = I - D^-1 M`` where ``M = beta I - gamma (A (x) J)``
and ``D = diag(M)``; if ``rho(N) >= 1`` the operator is not diagonally
dominant and the Neumann series diverges for all but the tiniest step
sizes.

The Jacobian is evaluated at the initial state by central finite
differences of the guarded right-hand side rather than by substituting
into the analytic Jacobian matrix. The analytic derivative of a CellML
gating term such as ``(V - E) / (exp((V - E) / k) - 1)`` is ``0 / 0`` at
the resting potential even though the term itself is finite there;
finite-differencing the guarded RHS takes that removable-singularity
limit numerically and yields a finite Jacobian.

Published Functions
-------------------
:func:`build_rhs_evaluator`
    Build a cached finite-difference Jacobian evaluator for a system.
:func:`neumann_spectral_radius`
    Pure-numeric Jacobi spectral-radius diagnostic as a function of the
    Jacobian and the ``beta``/``gamma``/tableau parameters.
:func:`check_neumann_convergence`
    Evaluate convergence diagnostics for the Neumann preconditioner and
    emit a warning when divergence is likely.
"""

import logging
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
import sympy as sp

from cubie.odesystems.symbolic.parsing.parser import TIME_SYMBOL
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing.parser import ParsedEquations
from cubie.odesystems.symbolic.sym_utils import topological_sort

logger = logging.getLogger(__name__)

# Central-difference step: cube-root of machine epsilon balances round-off
# against truncation error for a double-precision second-order stencil.
_FD_STEP = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)


class NeumannRHSEvaluator:
    """Finite-difference Jacobian evaluator for a symbolic system.

    Resolves the auxiliary chain into the state derivatives once and
    compiles the guarded right-hand side to a NumPy callable. The
    resulting object is cached on the owning :class:`SymbolicODE`; each
    call reads current state/constant/parameter values from the index
    map, so value changes are reflected without rebuilding.
    """

    def __init__(
        self,
        rhs_callable,
        argument_symbols: Sequence[sp.Symbol],
        state_symbols: Sequence[sp.Symbol],
        driver_symbols: Sequence[sp.Symbol],
    ) -> None:
        self._rhs = rhs_callable
        self._argument_symbols = list(argument_symbols)
        self._state_symbols = list(state_symbols)
        self._driver_symbols = set(driver_symbols)
        self._state_count = len(state_symbols)

    def _value_map(
        self,
        index_map: IndexedBases,
        t0: float,
    ) -> Dict[sp.Symbol, float]:
        """Collect current numeric values for every RHS symbol."""
        values: Dict[sp.Symbol, float] = {}
        for sym, val in index_map.states.default_values.items():
            values[sym] = float(val)
        for sym, val in index_map.constants.default_values.items():
            values[sym] = float(val)
        if hasattr(index_map, "parameters"):
            for sym, val in index_map.parameters.default_values.items():
                values[sym] = float(val)
        for sym in self._driver_symbols:
            values[sym] = 0.0
        values[TIME_SYMBOL] = float(t0)
        return values

    def jacobian(
        self,
        index_map: IndexedBases,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Return the state Jacobian at the initial state.

        Parameters
        ----------
        index_map
            Index maps supplying current state/constant/parameter values.
        t0
            Time at which to evaluate the right-hand side.

        Returns
        -------
        numpy.ndarray
            The ``state_count x state_count`` Jacobian. Entries that
            cannot be evaluated are returned as ``nan``.
        """
        values = self._value_map(index_map, t0)
        base_args = np.array(
            [values.get(sym, 0.0) for sym in self._argument_symbols],
            dtype=np.float64,
        )
        n = self._state_count

        def evaluate(args):
            with np.errstate(all="ignore"):
                try:
                    return np.asarray(self._rhs(*args), dtype=np.float64)
                except (
                    ZeroDivisionError,
                    FloatingPointError,
                    OverflowError,
                    ValueError,
                ):
                    return np.full(n, np.nan)

        jacobian = np.zeros((n, n), dtype=np.float64)
        for col in range(n):
            step = _FD_STEP * max(abs(base_args[col]), 1.0)
            forward = base_args.copy()
            backward = base_args.copy()
            forward[col] += step
            backward[col] -= step
            f_plus = evaluate(forward)
            f_minus = evaluate(backward)
            jacobian[:, col] = (f_plus - f_minus) / (2.0 * step)
        return jacobian


def _host_guarded(func):
    """Wrap a user callable so host evaluation degrades to ``nan``.

    User device functions are host-callable under the CUDA simulator
    but raise when called from host code on a real GPU; the guard
    turns that into a ``nan`` result so the convergence diagnostic is
    skipped gracefully instead of raising.
    """

    def wrapped(*args):
        try:
            return float(func(*args))
        except Exception:
            return float("nan")

    return wrapped


def build_rhs_evaluator(
    equations: ParsedEquations,
    index_map: IndexedBases,
    user_functions: Optional[Dict[str, object]] = None,
) -> NeumannRHSEvaluator:
    """Compile a finite-difference Jacobian evaluator for a system.

    Resolves the auxiliary assignments into each state derivative and
    lambdifies the guarded right-hand side. This is the expensive step;
    the returned evaluator is cheap to call and should be cached.

    Parameters
    ----------
    equations
        Parsed ODE equations (the same object passed to codegen).
    index_map
        Index maps (states, constants, drivers, ...) for the system.
    user_functions
        Callables resolving user function names appearing in the
        right-hand side, keyed by the printed name. Each is wrapped so
        a host-side call failure yields ``nan`` instead of raising.

    Returns
    -------
    NeumannRHSEvaluator
        Callable evaluator producing the state Jacobian by central
        finite differences of the guarded right-hand side.
    """
    state_symbols = list(index_map.states.index_map.keys())
    dxdt_symbols = list(index_map.dxdt.index_map.keys())
    dxdt_set = set(dxdt_symbols)

    # Resolve the auxiliary chain into the derivatives so the compiled
    # right-hand side depends only on states/constants/drivers/time. The
    # guarded Piecewise gating terms survive substitution and keep the
    # RHS finite at removable singularities.
    sorted_equations = topological_sort(equations.to_equation_list())
    auxiliary_subs: Dict[sp.Symbol, sp.Expr] = {}
    derivative_exprs: Dict[sp.Symbol, sp.Expr] = {}
    for lhs, rhs in sorted_equations:
        resolved = rhs.xreplace(auxiliary_subs)
        if lhs in dxdt_set:
            derivative_exprs[lhs] = resolved
        else:
            auxiliary_subs[lhs] = resolved

    constant_symbols = list(index_map.constants.default_values.keys())
    parameter_symbols = []
    if hasattr(index_map, "parameters"):
        parameter_symbols = list(
            index_map.parameters.default_values.keys()
        )
    driver_symbols = []
    if hasattr(index_map, "drivers"):
        driver_symbols = list(index_map.drivers.index_map.keys())

    argument_symbols = (
        state_symbols
        + constant_symbols
        + parameter_symbols
        + driver_symbols
        + [TIME_SYMBOL]
    )
    rhs_vector = [
        derivative_exprs.get(sym, sp.S.Zero) for sym in dxdt_symbols
    ]
    # Evaluate with the ``math`` module (scalar ternaries for Piecewise,
    # builtin min/max) rather than NumPy: the finite-difference stencil
    # feeds scalars, and NumPy's ``select`` requires array conditions.
    guarded_functions = {
        name: _host_guarded(func)
        for name, func in (user_functions or {}).items()
    }
    rhs_callable = sp.lambdify(
        argument_symbols,
        rhs_vector,
        modules=[guarded_functions, "math"],
        cse=True,
    )
    return NeumannRHSEvaluator(
        rhs_callable,
        argument_symbols,
        state_symbols,
        driver_symbols,
    )


def neumann_spectral_radius(
    jacobian: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, sp.Expr]]]
    ] = None,
) -> Dict[str, object]:
    """Spectral radius of the Jacobi iteration matrix ``N``.

    Pure-numeric diagnostic: builds ``M = beta I - gamma (A (x) J)``
    (or ``beta I - gamma J`` when ``stage_coefficients`` is ``None``) and
    returns the spectral radius and per-row diagonal-dominance ratios.

    Parameters
    ----------
    jacobian
        The ``n x n`` state Jacobian.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    stage_coefficients
        Butcher-tableau ``A`` matrix. When provided the flattened staged
        operator is analysed; otherwise the single-stage operator is
        used.

    Returns
    -------
    dict
        ``rho_N`` -- spectral radius of the Jacobi iteration matrix.
        ``max_ratio`` -- worst-case row off-diag/diag ratio.
        ``ratios`` -- per-row diagonal-dominance ratios.
        ``converges`` -- ``True`` if ``rho(N) < 1``.
        ``n_states`` / ``n_stages`` -- flattened-system dimensions.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    n = jacobian.shape[0]

    if stage_coefficients is not None:
        coefficients = np.array(
            [[float(c) for c in row] for row in stage_coefficients],
            dtype=np.float64,
        )
        s = coefficients.shape[0]
        operator = beta * np.eye(s * n) - gamma * np.kron(
            coefficients, jacobian
        )
    else:
        s = 1
        operator = beta * np.eye(n) - gamma * jacobian

    diag = np.diag(operator)

    # Guard against zero diagonal entries while preserving sign so the
    # replacement never flips D_inv on a near-zero negative diagonal.
    signs = np.where(diag >= 0.0, 1.0, -1.0)
    diag_safe = np.where(np.abs(diag) > 1e-30, diag, signs * 1e-30)
    diag_inv = np.diag(1.0 / diag_safe)

    iteration_matrix = np.eye(s * n) - diag_inv @ operator
    eigenvalues = np.linalg.eigvals(iteration_matrix)
    rho_n = float(np.max(np.abs(eigenvalues)))

    off_diagonal = np.sum(np.abs(operator), axis=1) - np.abs(diag)
    ratios = off_diagonal / np.maximum(np.abs(diag), 1e-30)

    return {
        "rho_N": rho_n,
        "max_ratio": float(np.max(ratios)),
        "ratios": ratios,
        "converges": rho_n < 1.0,
        "n_states": n,
        "n_stages": s,
    }


def check_neumann_convergence(
    equations: ParsedEquations,
    index_map: IndexedBases,
    evaluator: Optional[NeumannRHSEvaluator] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, sp.Expr]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, sp.Expr]]] = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    t0: float = 0.0,
    user_functions: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Evaluate whether the Neumann preconditioner is likely to converge.

    Evaluates the state Jacobian at the initial state by finite
    differences and checks the spectral radius of the Jacobi iteration
    matrix ``N = I - D^-1 M``.

    Parameters
    ----------
    equations
        Parsed ODE equations (the same object passed to codegen).
    index_map
        Index maps (states, constants, etc.) with default values.
    evaluator
        Prebuilt finite-difference Jacobian evaluator. Built from
        ``equations``/``index_map`` when omitted.
    stage_coefficients
        Butcher-tableau ``A`` matrix for the FIRK method. When provided,
        the full staged system ``beta I - gamma (A (x) J)`` is analysed
        (un-scaled by ``h`` so the result is h-independent).
    stage_nodes
        Butcher-tableau ``c`` nodes. These influence the Jacobian only
        through the ``O(h)`` per-stage time/state offset, which this
        h-independent diagnostic neglects, so they are not used.
    beta, gamma
        Transformation parameters from the FIRK formulation.
    t0
        Time at which to evaluate the Jacobian.
    user_functions
        Callables resolving user function names appearing in the
        right-hand side, keyed by the printed name. Used only when
        ``evaluator`` is omitted and forwarded to
        :func:`build_rhs_evaluator`.

    Returns
    -------
    dict
        The :func:`neumann_spectral_radius` result extended with
        ``worst_rows`` (offending ``(index, label, ratio)`` tuples) and
        ``J_numeric``. ``converges`` is ``None`` when the Jacobian could
        not be evaluated.
    """
    if evaluator is None:
        evaluator = build_rhs_evaluator(
            equations, index_map, user_functions=user_functions
        )

    jacobian = evaluator.jacobian(index_map, t0=t0)

    if not np.isfinite(jacobian).all():
        n_bad = int(np.count_nonzero(~np.isfinite(jacobian)))
        logger.warning(
            "Neumann preconditioner convergence not verified: the "
            "Jacobian at the initial state has %d non-finite entries "
            "(likely a genuine singularity at t0).",
            n_bad,
        )
        n_stages = (
            len(stage_coefficients)
            if stage_coefficients is not None
            else 1
        )
        return {
            "rho_N": float("nan"),
            "max_ratio": float("nan"),
            "converges": None,
            "n_states": jacobian.shape[0],
            "n_stages": n_stages,
            "worst_rows": [],
            "J_numeric": jacobian,
        }

    result = neumann_spectral_radius(
        jacobian, beta=beta, gamma=gamma,
        stage_coefficients=stage_coefficients,
    )

    n = result["n_states"]
    s = result["n_stages"]
    ratios = result["ratios"]
    idx_to_name = {
        v: str(k) for k, v in index_map.states.index_map.items()
    }
    worst_rows = []
    for idx in range(s * n):
        if ratios[idx] > 1.0:
            stage_idx = idx // n
            state_idx = idx % n
            name = idx_to_name.get(state_idx, f"state_{state_idx}")
            label = f"stage{stage_idx}:{name}" if s > 1 else name
            worst_rows.append((idx, label, float(ratios[idx])))
    worst_rows.sort(key=lambda row: -row[2])

    result["worst_rows"] = worst_rows
    result["J_numeric"] = jacobian
    del result["ratios"]

    if not result["converges"]:
        rho_n = result["rho_N"]
        top_rows = worst_rows[:5]
        row_details = ", ".join(
            f"{label} (ratio={ratio:.1f})"
            for _, label, ratio in top_rows
        )
        message = (
            "Neumann preconditioner will likely DIVERGE for this "
            f"system. Spectral radius rho(I - D^-1 M) = {rho_n:.2f} "
            ">= 1.0 (need < 1 for convergence). "
            f"{len(worst_rows)}/{s * n} rows violate diagonal "
            f"dominance. Worst: [{row_details}]. Consider a diagonal "
            "(Jacobi) preconditioner or a smaller step size."
        )
        warnings.warn(message, stacklevel=3)
        logger.warning(message)

    return result
