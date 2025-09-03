import re
from typing import Dict, Iterable, Optional, Tuple

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

CUDA_FUNCTIONS: Dict[str, str] = {
    'exp': 'math.exp',
    'log': 'math.log',
    'sin': 'math.sin',
    'cos': 'math.cos',
    'tan': 'math.tan',
    'sqrt': 'math.sqrt',
    'Abs': 'math.Abs',
    'Min': 'math.Min',
    'Max': 'math.Max',
    'sign': 'math.sign',
}


class CUDAPrinter(PythonCodePrinter):
    """SymPy printer for CUDA code generation with symbol substitutions and optimizations."""

    def __init__(self, symbol_map=None, *args, **kwargs):
        """
        Initialize CUDA printer.

        Args:
            symbol_map: Dictionary mapping Symbol instances to IndexedBase references
        """
        super().__init__(*args, **kwargs)
        self.symbol_map = symbol_map or {}

    def doprint(self, expr, **kwargs):
        """Main printing method that applies all transformations."""
        assign_to = kwargs.get("assign_to", None)
        # Force outer assignment for Piecewise to avoid assignments inside ternaries
        if assign_to is not None and isinstance(expr, sp.Piecewise):
            rhs = self._print(expr)
            lhs = self._print(assign_to)
            result = f"{lhs} = {rhs}"
        else:
            result = super().doprint(expr, **kwargs)
        result = self._replace_powers_with_multiplication(result)
        # result = self._ifelse_to_selp(result)
        return result

    def _print_Symbol(self, expr):
        """Print Symbol, applying symbol-to-IndexedBase mapping if available."""
        if expr in self.symbol_map:
            return self._print(self.symbol_map[expr])
        return super()._print_Symbol(expr)

    def _print_Piecewise(self, expr: sp.Piecewise):
        """Always render Piecewise as a pure expression (nested ternaries).

        This avoids generating assignments inside conditional expressions when
        ``assign_to`` is provided to ``doprint``. The outer assignment will be
        handled by ``doprint`` itself ("lhs = <expr>").
        """
        # expr.args is a tuple of (expr_i, cond_i). The last cond may be True.
        pieces = list(expr.args)
        # Build nested ternary from the end to the start.
        # Start with the last expression (which should have a True condition or be the fallback).
        last_expr, _ = pieces[-1]
        rendered = self._print(last_expr)
        # Process in reverse, skipping the last fallback.
        for e, c in reversed(pieces[:-1]):
            cond = self._print(c)
            val = self._print(e)
            rendered = f"({val} if {cond} else ({rendered}))"
        return rendered

    def _replace_powers_with_multiplication(self, expr_str):
        """Replace **2 and **3 with explicit multiplications for efficiency."""
        expr_str = self._replace_square_powers(expr_str)
        expr_str = self._replace_cube_powers(expr_str)
        return expr_str

    def _replace_square_powers(self, expr_str):
        """Replace x**2 with x*x, handling spaces around the ** operator."""
        return re.sub(r"(\w+(?:\[[^]]+])*)\s*\*\*\s*2\b", r"\1*\1", expr_str)

    def _replace_cube_powers(self, expr_str):
        """Replace x**3 with x*x*x, handling spaces around the ** operator."""
        return re.sub(r'(\w+(?:\[[^]]+])*)\s*\*\*\s*3\b', r'\1*\1*\1',
                      expr_str)

    def _ifelse_to_selp(self, expr_str):
        """Replace if-else statements with select statements."""
        return re.sub(
            r"\s+(.+?)\sif\s+(.+?)\s+else\s+(.+)",
            r"cuda.selp(\2, \1, \3)",
            expr_str,
        )

    # TODO: Singularity skips from Chaste codegen, piecewise blend if required
    # TODO: Add translation to CUDA-native functions


def print_cuda(expr: sp.Expr,
               symbol_map: Optional[Dict] = None,
               **kwargs):
    """
    Convenience function to print SymPy expressions as CUDA-optimized code.

    Args:
        expr: SymPy expression to print
        symbol_map: Dictionary mapping Symbol instances to IndexedBase references
        **kwargs: Additional arguments passed to the printer

    Returns:
        String representation of the expression optimized for CUDA
    """
    printer = CUDAPrinter(symbol_map=symbol_map, **kwargs)
    return printer.doprint(expr)

def print_cuda_multiple(exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
                        symbol_map=None,
                        **kwargs):
    """
    Convenience function to print SymPy expressions as CUDA-optimized code.

    Parameters
    ----------
        exprs: iterable of tuples of (sp.Symbol, sp.Expr)
            Iterable of symbol, expression pairs for "symbol = expression"
            assignments
        symbol_map: dict of sp.Symbol to IndexedBase or str
            Dictionary mapping algebraic symbols to their array or variable
            names
        **kwargs: Additional arguments passed to the printer

    Returns:
        String representation of the expression optimized for CUDA
    """
    printer = CUDAPrinter(symbol_map=symbol_map, **kwargs)
    lines = []
    for assign_to, expr in exprs:
        line = printer.doprint(expr, assign_to=assign_to)
        lines.append(line)

    return lines
