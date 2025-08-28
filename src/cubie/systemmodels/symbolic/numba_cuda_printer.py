import re
from typing import Dict, Iterable, Optional, Tuple

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter


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
        result = super().doprint(expr, **kwargs)
        result = self._replace_powers_with_multiplication(result)
        return result

    def _print_Symbol(self, expr):
        """Print Symbol, applying symbol-to-IndexedBase mapping if available."""
        if expr in self.symbol_map:
            return self._print(self.symbol_map[expr])
        return super()._print_Symbol(expr)

    def _replace_powers_with_multiplication(self, expr_str):
        """Replace **2 and **3 with explicit multiplications for efficiency."""
        expr_str = self._replace_square_powers(expr_str)
        expr_str = self._replace_cube_powers(expr_str)
        return expr_str

    def _replace_square_powers(self, expr_str):
        """Replace x**2 with x*x, handling spaces around the ** operator."""
        return re.sub(r"(\w+(?:\[[^]]+\])*)\s*\*\*\s*2\b", r"\1*\1", expr_str)

    def _replace_cube_powers(self, expr_str):
        """Replace x**3 with x*x*x, handling spaces around the ** operator."""
        return re.sub(r'(\w+(?:\[[^]]+\])*)\s*\*\*\s*3\b', r'\1*\1*\1',
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
    for i in range(len(exprs)):
        assign_to, expr = exprs[i]
        line = printer.doprint(expr, assign_to=assign_to)
        lines.append(line)

    return lines
