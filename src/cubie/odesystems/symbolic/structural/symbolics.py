"""SymPy primitives backing the structural simplification passes.

These replace the Symbolics.jl operations used by
ModelingToolkitTearing: structural linear expansion, linear solving,
fixpoint substitution, total time derivatives over derivative-symbol
maps, and the derivative-symbol registry that stands in for MTK's
``Differential`` terms (cubie states are plain SymPy symbols, so
derivatives are represented by registered companion symbols).

Published Classes
-----------------
:class:`DerivativeRegistry`
    Creates and tracks derivative symbols (``x`` -> ``x_t`` ->
    ``x_tt`` ...) with collision-safe naming.

Published Functions
-------------------
:func:`linear_expansion`
    Decompose ``expr`` as ``a*var + b`` with ``a``, ``b`` free of
    ``var``, or report nonlinearity.

:func:`solve_linear`
    Solve a linear equation for a variable.

:func:`fixpoint_sub`
    Structural substitution applied until a fixed point.

:func:`total_derivative`
    Total time derivative of an expression under a derivative map.

:func:`as_small_int`
    Extract an integer with ``|value| <= 127`` from a SymPy number.
"""

from typing import Dict, Iterable, Optional, Set, Tuple

import sympy as sp


def linear_expansion(
    expr: sp.Expr, var: sp.Symbol
) -> Tuple[sp.Expr, sp.Expr, bool]:
    """Decompose ``expr`` into ``a*var + b`` when possible.

    Returns ``(a, b, islinear)``. When ``islinear`` is true, ``a`` and
    ``b`` contain no occurrence of ``var`` and
    ``expr == a*var + b`` holds structurally. Mirrors
    ``Symbolics.linear_expansion``: expressions where ``var`` appears
    inside a nonlinear or non-polynomial construct report
    ``islinear = False``.
    """

    zero = sp.S.Zero
    if expr == var:
        return (sp.S.One, zero, True)
    if var not in expr.free_symbols:
        return (zero, expr, True)
    if expr.is_Add:
        a_total = zero
        b_total = zero
        for arg in expr.args:
            a, b, islinear = linear_expansion(arg, var)
            if not islinear:
                return (zero, zero, False)
            a_total += a
            b_total += b
        return (a_total, b_total, True)
    if expr.is_Mul:
        var_factor = None
        rest = []
        for arg in expr.args:
            if var in arg.free_symbols:
                if var_factor is not None:
                    return (zero, zero, False)
                var_factor = arg
            else:
                rest.append(arg)
        a, b, islinear = linear_expansion(var_factor, var)
        if not islinear:
            return (zero, zero, False)
        rest_prod = sp.Mul(*rest)
        return (a * rest_prod, b * rest_prod, True)
    # Pow, functions, Piecewise, ... containing var: nonlinear.
    return (zero, zero, False)


def solve_linear(
    lhs: sp.Expr, rhs: sp.Expr, var: sp.Symbol
) -> Optional[sp.Expr]:
    """Solve ``lhs == rhs`` for ``var`` assuming linearity.

    Returns the solution expression, or ``None`` when the equation is
    singular in ``var`` (zero coefficient).
    """

    residual = lhs - rhs
    a, b, islinear = linear_expansion(residual, var)
    if not islinear:
        raise ValueError(
            f"cannot solve nonlinear equation for {var}: {residual}"
        )
    if a == sp.S.Zero:
        return None
    return -b / a


def fixpoint_sub(
    expr: sp.Expr,
    sub_map: Dict[sp.Symbol, sp.Expr],
    maxiters: Optional[int] = None,
) -> sp.Expr:
    """Apply structural substitution until the expression stabilises.

    Substitution keys are plain symbols so :meth:`~sympy.Expr.xreplace`
    (exact-node replacement) is sufficient; it is applied repeatedly
    because substituted expressions may themselves contain keys.
    """

    if not sub_map:
        return expr
    if maxiters is None:
        maxiters = len(sub_map) + 10
    for _ in range(maxiters):
        new_expr = expr.xreplace(sub_map)
        if new_expr == expr:
            return new_expr
        expr = new_expr
    raise ValueError(
        "fixpoint substitution failed to converge; the substitution "
        "map is likely cyclic"
    )


def total_derivative(
    expr: sp.Expr,
    deriv_map: Dict[sp.Symbol, sp.Symbol],
    time_symbol: sp.Symbol,
    known_derivative_map: Optional[Dict[sp.Symbol, sp.Expr]] = None,
) -> sp.Expr:
    """Total time derivative of ``expr``.

    Parameters
    ----------
    expr
        Expression to differentiate.
    deriv_map
        Map from unknown symbols to their derivative symbols. Unknowns
        absent from the map cannot be differentiated and raise.
    time_symbol
        The independent variable; explicit dependence differentiates
        through :func:`sympy.diff`.
    known_derivative_map
        Derivative expressions for known time-dependent quantities
        (drivers). Known symbols absent from this map differentiate
        to zero, matching MTK's default for time-dependent
        parameters.

    Notes
    -----
    Computed as ``sum(diff(expr, v) * dv) + diff(expr, t)`` over the
    mapped symbols occurring in ``expr``.
    """

    if known_derivative_map is None:
        known_derivative_map = {}
    result = sp.diff(expr, time_symbol)
    for sym in expr.free_symbols:
        if sym == time_symbol:
            continue
        dsym = deriv_map.get(sym)
        if dsym is not None:
            result += sp.diff(expr, sym) * dsym
            continue
        known = known_derivative_map.get(sym)
        if known is not None:
            result += sp.diff(expr, sym) * known
        # Symbols that are neither unknowns nor known time-dependent
        # quantities are constants/parameters: derivative zero.
    return result


def as_small_int(value: sp.Expr) -> Optional[int]:
    """Return ``int(value)`` when it is integral with ``|v| <= 127``.

    Mirrors StateSelection's small-integer coefficient gate for the
    integer-linear subsystem; returns ``None`` otherwise.
    """

    if not value.is_number:
        return None
    if isinstance(value, sp.Integer):
        iv = int(value)
    elif isinstance(value, sp.Rational):
        if value.q != 1:
            return None
        iv = int(value)
    elif isinstance(value, sp.Float):
        fv = float(value)
        if not fv.is_integer():
            return None
        iv = int(fv)
    else:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return None
        if not fv.is_integer():
            return None
        iv = int(fv)
    if -127 <= iv <= 127:
        return iv
    return None


def lower_varname(
    base_name: str, order: int, reserved: Set[str]
) -> str:
    """User-visible name for a dummy-derivative variable.

    Produces ``x_t``, ``x_tt``, ... (the plain-symbol analogue of
    MTK's ``xˍt`` naming), appending underscores until the name is
    free in ``reserved``. The chosen name is added to ``reserved``.
    """

    name = f"{base_name}_{'t' * order}"
    while name in reserved:
        name = name + "_"
    reserved.add(name)
    return name


class DerivativeRegistry:
    """Factory and index for derivative symbols.

    Derivative symbols stand in for MTK's ``Differential`` terms and
    are plain real SymPy symbols with mangled internal names
    (``_cubie_D<order>_<base>``); they are renamed to user-visible
    ``x_t`` forms during reassembly when state selection turns them
    into ordinary algebraic variables. The registry records
    base/order relations so higher-order chains can be walked
    symbolically as well as through the integer
    :class:`~cubie.odesystems.symbolic.structural.diffgraph.DiffGraph`.

    Parameters
    ----------
    reserved_names
        Names that generated symbols must not collide with (all user
        symbols in the system).
    """

    def __init__(self, reserved_names: Iterable[str]) -> None:
        self.reserved = set(reserved_names)
        self._to_base = {}
        self._to_derivative = {}

    def register_known(
        self, base: sp.Symbol, derivative: sp.Symbol
    ) -> None:
        """Record a pre-existing base/derivative symbol pair."""

        self._to_derivative[base] = derivative
        self._to_base[derivative] = base
        self.reserved.add(derivative.name)

    def derivative(self, var: sp.Symbol) -> sp.Symbol:
        """Return (creating if needed) the derivative symbol of ``var``."""

        existing = self._to_derivative.get(var)
        if existing is not None:
            return existing
        base, order = self.base_and_order(var)
        name = f"_cubie_D{order + 1}_{base.name}"
        while name in self.reserved:
            name = name + "_"
        dsym = sp.Symbol(name, real=True)
        self.reserved.add(name)
        self._to_derivative[var] = dsym
        self._to_base[dsym] = var
        return dsym

    def lower_order(self, var: sp.Symbol) -> Optional[sp.Symbol]:
        """Return the symbol ``var`` is the derivative of, if known."""

        return self._to_base.get(var)

    def base_and_order(self, var: sp.Symbol) -> Tuple[sp.Symbol, int]:
        """Return the underived base symbol and derivative order."""

        order = 0
        base = var
        while True:
            lower = self._to_base.get(base)
            if lower is None:
                return (base, order)
            base = lower
            order += 1

    def is_derivative(self, var: sp.Symbol) -> bool:
        """Whether ``var`` is a registered derivative symbol."""

        return var in self._to_base

    def deriv_map(self) -> Dict[sp.Symbol, sp.Symbol]:
        """Return a copy of the base-to-derivative map."""

        return dict(self._to_derivative)

    def rename(self, old: sp.Symbol, new: sp.Symbol) -> None:
        """Rebind a registered symbol to a new symbol.

        Chain relations (its base and its derivative, if any) are
        preserved under the new symbol.
        """

        lower = self._to_base.pop(old, None)
        if lower is not None:
            self._to_base[new] = lower
            self._to_derivative[lower] = new
        upper = self._to_derivative.pop(old, None)
        if upper is not None:
            self._to_derivative[new] = upper
            self._to_base[upper] = new
        self.reserved.add(new.name)
