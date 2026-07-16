"""One-time conversion of parsed systems into engine IR.

``ParsedEquations`` and ``IndexedBases`` stay SymPy-facing (parsing,
hashing, GUIs); codegen converts them here, once per system, into a
:class:`SystemIR` bundle. Every generator consumes the bundle, so
SymPy objects never enter the compute-heavy generation paths.

Published Classes
-----------------
:class:`SystemIR`
    IR equations plus the ordered symbol tables generators need.

Published Functions
-------------------
:func:`system_ir`
    Memoised ``(ParsedEquations, IndexedBases) -> SystemIR``.
"""

from typing import Dict, List, Tuple

import attrs

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import (
    convert_assignments,
    derivative_name_map,
    from_sympy,
)

__all__ = ["SystemIR", "system_ir"]


@attrs.frozen(eq=False)
class SystemIR:
    """IR view of a parsed system.

    Parameters
    ----------
    equations
        All equations in evaluation order as IR pairs.
    state_symbols
        State symbols ordered by state index.
    dxdt_symbols
        Derivative-output symbols ordered by output index.
    observable_symbols
        Observable symbols ordered by observable index.
    driver_symbols
        Driver symbols ordered by driver index.
    state_index, dxdt_index, driver_index
        Symbol-to-position lookups for the ordered collections.
    arrayrefs
        Printer symbol map: scalar name to :class:`~.expr.Arr`.
    constant_names
        Factory-scope constant names (exponent aliasing).
    function_aliases
        Renamed user-function name to printable original name.
    derivative_names
        User-function name to derivative placeholder print name.
    time_symbol
        The IR time symbol (``t``).
    """

    equations: Tuple[Tuple[ir.Expr, ir.Expr], ...]
    state_symbols: Tuple[ir.Sym, ...]
    dxdt_symbols: Tuple[ir.Sym, ...]
    observable_symbols: Tuple[ir.Sym, ...]
    driver_symbols: Tuple[ir.Sym, ...]
    state_index: Dict[ir.Sym, int]
    dxdt_index: Dict[ir.Sym, int]
    driver_index: Dict[ir.Sym, int]
    arrayrefs: Dict[str, ir.Expr]
    constant_names: Tuple[str, ...]
    function_aliases: Dict[str, str]
    derivative_names: Dict[str, str]
    time_symbol: ir.Sym

    @property
    def observable_set(self) -> frozenset:
        """Return the observable symbols as a frozenset."""
        return frozenset(self.observable_symbols)

    def non_observable_equations(
        self,
    ) -> List[Tuple[ir.Expr, ir.Expr]]:
        """Return equations whose outputs are not observables."""
        observables = self.observable_set
        return [
            (lhs, rhs)
            for lhs, rhs in self.equations
            if lhs not in observables
        ]


_CACHE: Dict[Tuple[int, int], Tuple[object, object, SystemIR]] = {}


def _ordered_syms(index_map) -> Tuple[Tuple[ir.Sym, ...], Dict]:
    """Convert a SymPy ``symbol -> position`` map to ordered IR."""
    ordered = sorted(index_map.items(), key=lambda item: item[1])
    symbols = tuple(ir.sym(str(sym)) for sym, _ in ordered)
    index = {symbol: pos for pos, symbol in enumerate(symbols)}
    return symbols, index


def system_ir(equations, index_map) -> SystemIR:
    """Return the memoised IR view of a parsed system.

    Parameters
    ----------
    equations
        ``ParsedEquations`` from the parser.
    index_map
        ``IndexedBases`` bundle for the same system.

    Returns
    -------
    SystemIR
        Converted equations and symbol tables. Repeated calls with
        the same objects return the same instance.
    """
    key = (id(equations), id(index_map))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is equations and hit[1] is index_map:
        return hit[2]

    memo: Dict = {}
    eq_list = equations.to_equation_list()
    if eq_list and isinstance(eq_list[0][0], ir.Expr):
        # Parser output is already IR; the pairs pass through and
        # derivative placeholder names ride on the container.
        eq_pairs = tuple(eq_list)
        derivative_names = dict(
            getattr(equations, "derivative_names", None) or {}
        )
    else:
        eq_pairs = tuple(convert_assignments(eq_list, memo))
        derivative_names = derivative_name_map(eq_list)

    states, state_index = _ordered_syms(index_map.states.index_map)
    dxdt, dxdt_index = _ordered_syms(index_map.dxdt.index_map)
    drivers, driver_index = _ordered_syms(
        index_map.drivers.index_map
    )
    observables = tuple(
        ir.sym(str(sym))
        for sym in index_map.observables.index_map.keys()
    )

    arrayrefs: Dict[str, ir.Expr] = {}
    aliases: Dict[str, str] = {}
    for sym_key, ref in index_map.all_arrayrefs.items():
        if sym_key == "__function_aliases__" or str(
            sym_key
        ) == "__function_aliases__":
            if isinstance(ref, dict):
                aliases = dict(ref)
            continue
        arrayrefs[str(sym_key)] = from_sympy(ref, memo)

    result = SystemIR(
        equations=eq_pairs,
        state_symbols=states,
        dxdt_symbols=dxdt,
        observable_symbols=observables,
        driver_symbols=drivers,
        state_index=state_index,
        dxdt_index=dxdt_index,
        driver_index=driver_index,
        arrayrefs=arrayrefs,
        constant_names=tuple(index_map.constants.symbol_map),
        function_aliases=aliases,
        derivative_names=derivative_names,
        time_symbol=ir.sym("t"),
    )
    _CACHE[key] = (equations, index_map, result)
    return result
