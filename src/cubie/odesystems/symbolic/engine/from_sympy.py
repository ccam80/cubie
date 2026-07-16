"""Convert SymPy expressions into engine IR nodes.

This is the only module in the engine that imports SymPy. Parsing
(and structural simplification) still produce SymPy expressions;
codegen converts them once, here, and every downstream compute step
(differentiation, substitution, CSE, printing) runs on the IR.

Published Functions
-------------------
:func:`from_sympy`
    Convert one SymPy expression (memoised).
:func:`convert_assignments`
    Convert an iterable of ``(lhs, rhs)`` SymPy assignment pairs.
:func:`derivative_name_map`
    Recover user-function derivative placeholder names from the
    dynamic ``fdiff`` classes built during parsing.

Notes
-----
Symbols whose *names* carry an index (``sp.Symbol("out[3]")``) and
``sp.Indexed`` leaves both convert to :class:`~.expr.Arr` — the
engine's own lightweight indexed reference — so the IR has exactly
one way to spell an array element.
"""

import math
import re
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from cubie.odesystems.symbolic.engine import expr as ir

__all__ = [
    "from_sympy",
    "to_sympy",
    "convert_assignments",
    "derivative_name_map",
    "ConversionError",
]

_BRACKET_NAME = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]$"
)

_REL_OPS = {
    sp.StrictLessThan: "<",
    sp.LessThan: "<=",
    sp.StrictGreaterThan: ">",
    sp.GreaterThan: ">=",
    sp.Eq: "==",
    sp.Ne: "!=",
}


class ConversionError(TypeError):
    """Raised when a SymPy construct has no IR equivalent."""


def _convert_number(node: sp.Expr) -> ir.Expr:
    if isinstance(node, sp.Integer):
        return ir.num(int(node))
    if isinstance(node, sp.Rational):
        return ir.num(Fraction(int(node.p), int(node.q)))
    if isinstance(node, sp.Float):
        return ir.num(float(node))
    if node is sp.pi:
        return ir.num(math.pi)
    if node is sp.E:
        return ir.num(math.e)
    if node is sp.EulerGamma:
        return ir.num(0.5772156649015329)
    if isinstance(node, sp.NumberSymbol):
        return ir.num(float(node))
    raise ConversionError(f"unsupported numeric node: {node!r}")


def from_sympy(
    node: sp.Basic,
    memo: Optional[Dict[sp.Basic, ir.Expr]] = None,
) -> ir.Expr:
    """Convert a SymPy expression to an IR expression.

    Parameters
    ----------
    node
        SymPy expression to convert.
    memo
        Optional memo dictionary; pass one dictionary across calls to
        share conversion work between expressions of one system.

    Returns
    -------
    ir.Expr
        Equivalent interned IR expression.

    Raises
    ------
    ConversionError
        When ``node`` contains a construct outside the supported
        vocabulary (e.g. ``Derivative``, infinities, complex units).
    """
    if memo is None:
        memo = {}

    def walk(current: sp.Basic) -> ir.Expr:
        cached = memo.get(current)
        if cached is not None:
            return cached
        result = _convert(current, walk)
        memo[current] = result
        return result

    return walk(node)


def _convert(current: sp.Basic, walk) -> ir.Expr:
    """Convert one SymPy node, recursing through ``walk``."""
    if current is sp.oo or current is -sp.oo or current is sp.zoo:
        raise ConversionError(
            f"infinite quantity in equations: {current!r}. If this "
            "arose from a symbol named like a SymPy constant "
            "(e.g. 'zoo', 'oo'), declare the symbol before use."
        )
    if current is sp.nan:
        raise ConversionError("NaN in equations")
    if isinstance(current, sp.Symbol):
        match = _BRACKET_NAME.match(current.name)
        if match is not None:
            return ir.arr(
                match.group("name"), int(match.group("index"))
            )
        return ir.sym(current.name)
    if isinstance(current, (sp.Number, sp.NumberSymbol)):
        if current is sp.oo or current is -sp.oo or current is sp.zoo:
            raise ConversionError(
                f"infinite quantity in equations: {current!r}"
            )
        if current is sp.nan:
            raise ConversionError("NaN in equations")
        return _convert_number(current)
    if isinstance(current, sp.Indexed):
        base_name = str(current.base.label)
        if len(current.indices) != 1:
            raise ConversionError(
                f"only 1-D indexed references supported: {current!r}"
            )
        index = current.indices[0]
        if not (
            isinstance(index, sp.Integer)
            or (
                isinstance(index, (int,))
                and not isinstance(index, bool)
            )
        ):
            raise ConversionError(
                f"array index must be a fixed integer: {current!r}"
            )
        return ir.arr(base_name, int(index))
    if isinstance(current, sp.Add):
        return ir.add(*(walk(a) for a in current.args))
    if isinstance(current, sp.Mul):
        return ir.mul(*(walk(a) for a in current.args))
    if isinstance(current, sp.Pow):
        return ir.pow_(walk(current.base), walk(current.exp))
    if isinstance(current, sp.Piecewise):
        pairs: List[Tuple[ir.Expr, ir.Expr]] = []
        for value, cond in current.args:
            pairs.append((walk(value), walk(cond)))
        return ir.piecewise(*pairs)
    if isinstance(current, BooleanTrue):
        return ir.TRUE
    if isinstance(current, BooleanFalse):
        return ir.FALSE
    for rel_cls, op in _REL_OPS.items():
        if isinstance(current, rel_cls):
            return ir.rel(op, walk(current.lhs), walk(current.rhs))
    if isinstance(current, sp.And):
        return ir.bool_op("and", *(walk(a) for a in current.args))
    if isinstance(current, sp.Or):
        return ir.bool_op("or", *(walk(a) for a in current.args))
    if isinstance(current, sp.Not):
        return ir.bool_op("not", walk(current.args[0]))
    if isinstance(current, sp.Abs):
        return ir.call("Abs", walk(current.args[0]))
    if isinstance(current, sp.Heaviside):
        argument = walk(current.args[0])
        if len(current.args) > 1:
            at_zero = walk(current.args[1])
        else:
            at_zero = ir.num(Fraction(1, 2))
        return ir.piecewise(
            (ir.ZERO, ir.rel("<", argument, ir.ZERO)),
            (at_zero, ir.rel("==", argument, ir.ZERO)),
            (ir.ONE, ir.TRUE),
        )
    if isinstance(current, sp.Min):
        return ir.call("Min", *(walk(a) for a in current.args))
    if isinstance(current, sp.Max):
        return ir.call("Max", *(walk(a) for a in current.args))
    if isinstance(current, sp.Derivative):
        raise ConversionError(
            "unresolved Derivative reached codegen; structural "
            f"simplification should have removed it: {current!r}"
        )
    if isinstance(current, AppliedUndef):
        name = type(current).__name__
        return ir.call(name, *(walk(a) for a in current.args))
    if isinstance(current, sp.Function):
        # Known SymPy functions print by their class name, matching
        # the CUDA_FUNCTIONS lookup keys used by the printer.
        name = type(current).__name__
        return ir.call(name, *(walk(a) for a in current.args))
    raise ConversionError(
        f"unsupported SymPy node in codegen: {type(current).__name__}"
        f" ({current!r})"
    )


def convert_assignments(
    assignments: Iterable[Tuple[sp.Symbol, sp.Basic]],
    memo: Optional[Dict[sp.Basic, ir.Expr]] = None,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Convert ``(lhs, rhs)`` SymPy pairs to IR pairs.

    Parameters
    ----------
    assignments
        Assignment pairs whose left-hand sides are SymPy symbols
        (bracket-named symbols convert to :class:`~.expr.Arr`).
    memo
        Optional shared conversion memo.

    Returns
    -------
    list of tuple
        IR assignment pairs in the input order.
    """
    if memo is None:
        memo = {}
    return [
        (from_sympy(lhs, memo), from_sympy(rhs, memo))
        for lhs, rhs in assignments
    ]


_TO_SYMPY_RELS = {
    "<": sp.StrictLessThan,
    "<=": sp.LessThan,
    ">": sp.StrictGreaterThan,
    ">=": sp.GreaterThan,
    "==": sp.Eq,
    "!=": sp.Ne,
}

_TO_SYMPY_FUNCS = {
    "Abs": sp.Abs,
    "Min": sp.Min,
    "Max": sp.Max,
    "ceiling": sp.ceiling,
    "floor": sp.floor,
    "sign": sp.sign,
}


def to_sympy(node: ir.Expr) -> sp.Basic:
    """Convert an IR expression back to SymPy.

    Primarily a verification utility: tests convert engine results to
    SymPy to compare against ground-truth expressions. Array
    references become bracket-named symbols (``out[0]``), the naming
    :func:`from_sympy` maps back to :class:`~.expr.Arr`.

    Parameters
    ----------
    node
        IR expression to convert.

    Returns
    -------
    sympy.Basic
        Equivalent SymPy expression.
    """
    if isinstance(node, ir.Num):
        value = node.value
        if isinstance(value, int):
            return sp.Integer(value)
        if isinstance(value, Fraction):
            return sp.Rational(value.numerator, value.denominator)
        return sp.Float(value)
    if isinstance(node, ir.Sym):
        return sp.Symbol(node.name, real=True)
    if isinstance(node, ir.Arr):
        return sp.Symbol(f"{node.name}[{node.index}]", real=True)
    if isinstance(node, ir.Add):
        return sp.Add(*(to_sympy(a) for a in node.args))
    if isinstance(node, ir.Mul):
        return sp.Mul(*(to_sympy(a) for a in node.args))
    if isinstance(node, ir.Pow):
        return sp.Pow(to_sympy(node.base), to_sympy(node.exp))
    if isinstance(node, ir.Piecewise):
        return sp.Piecewise(
            *(
                (to_sympy(value), to_sympy(cond))
                for value, cond in node.pairs
            )
        )
    if isinstance(node, ir.Rel):
        return _TO_SYMPY_RELS[node.op](
            to_sympy(node.lhs), to_sympy(node.rhs)
        )
    if isinstance(node, ir.BoolOp):
        args = [to_sympy(a) for a in node.args]
        if node.kind == "and":
            return sp.And(*args)
        if node.kind == "or":
            return sp.Or(*args)
        return sp.Not(args[0])
    if isinstance(node, ir.BoolConst):
        return sp.true if node.value else sp.false
    if isinstance(node, ir.Call):
        mapped = _TO_SYMPY_FUNCS.get(node.name)
        if mapped is None:
            mapped = getattr(sp, node.name, None)
        if mapped is None or not callable(mapped):
            mapped = sp.Function(node.name)
        return mapped(*(to_sympy(a) for a in node.args))
    raise ConversionError(f"unknown IR node: {node!r}")


def derivative_name_map(
    equations: Iterable[Tuple[sp.Symbol, sp.Basic]],
) -> Dict[str, str]:
    """Recover derivative placeholder names for user functions.

    The parser wraps device functions (and user functions with
    derivative helpers) in dynamic ``sp.Function`` subclasses whose
    ``fdiff`` emits a placeholder call. This inspects each applied
    user function once and records the placeholder's printed name so
    IR differentiation reproduces it exactly.

    Parameters
    ----------
    equations
        SymPy assignment pairs to scan.

    Returns
    -------
    dict
        Mapping from applied-function name to derivative placeholder
        name.
    """
    names: Dict[str, str] = {}
    for _, rhs in equations:
        if not isinstance(rhs, sp.Basic):
            continue
        # The parser's dynamic device-function classes are *defined*
        # Function subclasses (they implement eval/fdiff), so scan
        # every applied function, not just AppliedUndef.
        for node in rhs.atoms(sp.Function):
            func_name = type(node).__name__
            if func_name in names:
                continue
            fdiff = getattr(node, "fdiff", None)
            if fdiff is None:
                continue
            try:
                placeholder = node.fdiff(1)
            except Exception:
                continue
            if not isinstance(placeholder, AppliedUndef):
                # Default fdiff yields Derivative/Subs, not a
                # placeholder call; only the parser's dynamic
                # classes produce an applied function here.
                continue
            names[func_name] = type(placeholder).__name__
    return names
