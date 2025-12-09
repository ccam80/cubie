"""SymPy printer utilities that emit CUDA-friendly Python code snippets."""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

# Map SymPy function names to CUDA/Python math equivalents for printing
# Keys should match expr.func.__name__ from SymPy expressions
CUDA_FUNCTIONS: Dict[str, str] = {
    # Elementary trig
    'sin': 'math.sin',
    'cos': 'math.cos',
    'tan': 'math.tan',
    'asin': 'math.asin',
    'acos': 'math.acos',
    'atan': 'math.atan',
    'atan2': 'math.atan2',

    # Hyperbolic
    'sinh': 'math.sinh',
    'cosh': 'math.cosh',
    'tanh': 'math.tanh',
    'asinh': 'math.asinh',
    'acosh': 'math.acosh',
    'atanh': 'math.atanh',

    # Exponential / Logarithmic
    'exp': 'math.exp',
    'expm1': 'math.expm1',
    'log': 'math.log',
    'log2': 'math.log2',
    'log10': 'math.log10',
    'log1p': 'math.log1p',

    # Special functions
    'erf': 'math.erf',
    'erfc': 'math.erfc',
    'gamma': 'math.gamma',
    'loggamma': 'math.lgamma',  # map SymPy loggamma -> math.lgamma
    'hypot': 'math.hypot',

    # Rounding / absolute
    'Abs': 'math.fabs',  # prefer math.fabs for CUDA
    'floor': 'math.floor',
    'ceiling': 'math.ceil',  # SymPy uses ceiling()

    # Power / roots
    'sqrt': 'math.sqrt',
    'pow': 'math.pow',

    # Min/Max
    'Min': 'min',
    'Max': 'max',

    # Misc math
    'copysign': 'math.copysign',
    'fmod': 'math.fmod',
    'modf': 'math.modf',
    'frexp': 'math.frexp',
    'ldexp': 'math.ldexp',
    'remainder': 'math.remainder',

    # Classification
    'isnan': 'math.isnan',
    'isinf': 'math.isinf',
    'isfinite': 'math.isfinite',
}

class CUDAPrinter(PythonCodePrinter):
    """SymPy printer that maps expressions to CUDA-compatible source strings."""

    def __init__(
        self,
        symbol_map: Optional[Dict[sp.Symbol, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise a CUDA-aware code printer.

        Parameters
        ----------
        symbol_map
            Optional mapping from scalar symbols to indexed array references or
            other replacement nodes.
        args
            Positional arguments forwarded to :class:`PythonCodePrinter`.
        kwargs
            Keyword arguments forwarded to :class:`PythonCodePrinter`.
        """
        super().__init__(*args, **kwargs)
        self.symbol_map: Dict[sp.Symbol, Any] = symbol_map or {}
        self.cuda_functions: Dict[str, str] = CUDA_FUNCTIONS
        # User function alias mapping: underscored symbolic name -> original printable name
        self.func_aliases: Dict[str, str] = {}
        if isinstance(self.symbol_map, dict) and "__function_aliases__" in self.symbol_map:
            aliases = self.symbol_map.get("__function_aliases__")
            if isinstance(aliases, dict):
                self.func_aliases = aliases

        #Printer context flags for conditional wrapping, for example
        self._in_index = False
        self._in_pow = False

    def doprint(self, expr: sp.Expr, **kwargs: Any) -> str:
        """Return the CUDA-oriented source string for ``expr``.

        Parameters
        ----------
        expr
            SymPy expression to print.
        **kwargs
            Additional keyword arguments passed to :meth:`PythonCodePrinter.doprint`.

        Returns
        -------
        str
            CUDA-compatible code representation of ``expr``.
        """
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

    def _print_Symbol(self, expr: sp.Symbol) -> str:
        """Print a symbol, applying array substitutions when configured.

        Parameters
        ----------
        expr
            Symbol to render.

        Returns
        -------
        str
            Printed representation that respects the configured substitutions.
        """
        if expr in self.symbol_map:
            return self._print(self.symbol_map[expr])
        return super()._print_Symbol(expr)

    def _print_Indexed(self, expr: sp.Indexed) -> str:
        """Print indexed expression without wrapping indices.

        Parameters
        ----------
        expr
            Indexed expression to render.

        Returns
        -------
        str
            Printed representation with base[indices] where index expressions
            are NOT wrapped with precision().

        Notes
        -----
        Array indices must be integers and should not be wrapped with
        precision() to avoid type errors. This method ensures that integers
        appearing in index expressions (e.g., state[i + 1]) are not wrapped.
        """
        base = self._print(expr.args[0])
        indices = expr.args[1:]

        # let other print functions know we're in an index -
        # for example, leave integers uncast
        self._in_index = True
        index = [self._print(ind) for ind in indices]
        self._in_index = False
        return f"{base}[{', '.join(index)}]"

    def _print_Integer(self, expr: sp.Integer) -> str:
        """Print an integer literal, wrapping with precision() unless in index or power contexts.

        Parameters
        ----------
        expr
            Integer expression to render.

        Returns
        -------
        str
            Precision-wrapped representation: ``precision(value)``.
            Unwrapped representation when in index or power contexts.

        Notes
        -----
        Integer literals in expressions are wrapped to avoid float64 promotion
        during mixed-type arithmetic operations. Array indices and power
        exponents are not wrapped to preserve their integer types.
        """
        if self._in_index or self._in_pow:
            return super()._print_Integer(expr)
        return f"precision({str(expr)})"

    def _print_Pow(self, expr: sp.Pow) -> str:
        """Print power expression, avoiding precision wrap for numeric exponents.

        Parameters
        ----------
        expr
            Power expression to render.

        Returns
        -------
        str
            Printed representation that preserves numeric exponents unwrapped
            for power-to-multiplication optimization.

        Notes
        -----
        Exponents are not wrapped with precision() so that the
        _replace_powers_with_multiplication regex can detect patterns like
        x**2 and x**3 for optimization.
        
        Parentheses are added around compound base expressions to ensure
        correct precedence (e.g., (a + b)**2 not a + b**2).
        """
        PREC = sp.printing.precedence.precedence
        base_str = self.parenthesize(expr.base, PREC(expr))
        exp_expr = expr.exp

        # Let other print functions know we're in a power expression
        self._in_pow = True
        exponent = self._print(exp_expr)
        self._in_pow = False
        return f"{base_str}**{exponent}"

    def _print_Piecewise(self, expr: sp.Piecewise) -> str:
        """Render a ``Piecewise`` expression as nested ternaries.

        Parameters
        ----------
        expr
            Piecewise expression to render.

        Returns
        -------
        str
            Nested ternary representation of ``expr``.

        Notes
        -----
        This avoids generating assignments inside conditional expressions when
        ``assign_to`` is supplied to :meth:`doprint`. The outer assignment is
        handled by :meth:`doprint` itself (``"lhs = <expr>"``).
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

    def _replace_powers_with_multiplication(self, expr_str: str) -> str:
        """Replace square and cube powers with explicit multiplications.

        Parameters
        ----------
        expr_str
            Source string to rewrite.

        Returns
        -------
        str
            Source string with ``x**2`` and ``x**3`` rewritten.
        """
        expr_str = self._replace_square_powers(expr_str)
        expr_str = self._replace_cube_powers(expr_str)
        return expr_str

    def _replace_square_powers(self, expr_str: str) -> str:
        """Replace ``x**2`` or ``x**2.0`` with ``x*x`` while preserving spacing.

        Parameters
        ----------
        expr_str
            Source string to rewrite.

        Returns
        -------
        str
            Source string with ``x**2`` or ``x**2.0`` rewritten.
            
        Notes
        -----
        Matches both simple identifiers (x**2, arr[i]**2) and parenthesized
        expressions ((a + b)**2).
        """
        # Match identifier or array access
        simple_pattern = r"(\w+(?:\[[^]]+])*)\s*\*\*\s*2(?:\.\d+)?\b"
        expr_str = re.sub(simple_pattern, r"\1*\1", expr_str)
        
        # Match parenthesized expressions (but not function calls)
        # Negative lookbehind (?<!\w) ensures no word char before (
        paren_pattern = r"(?<!\w)(\([^()]+\))\s*\*\*\s*2(?:\.\d+)?\b"
        expr_str = re.sub(paren_pattern, r"\1*\1", expr_str)
        
        return expr_str

    def _replace_cube_powers(self, expr_str: str) -> str:
        """Replace ``x**3`` or ``x**3.0`` with ``x*x*x`` while preserving spacing.

        Parameters
        ----------
        expr_str
            Source string to rewrite.

        Returns
        -------
        str
            Source string with ``x**3`` or ``x**3.0`` rewritten.
            
        Notes
        -----
        Matches both simple identifiers (x**3, arr[i]**3) and parenthesized
        expressions ((a + b)**3).
        """
        # Match identifier or array access
        simple_pattern = r'(\w+(?:\[[^]]+])*)\s*\*\*\s*3(?:\.\d+)?\b'
        expr_str = re.sub(simple_pattern, r'\1*\1*\1', expr_str)
        
        # Match parenthesized expressions (but not function calls)
        # Negative lookbehind (?<!\w) ensures no word char before (
        paren_pattern = r'(?<!\w)(\([^()]+\))\s*\*\*\s*3(?:\.\d+)?\b'
        expr_str = re.sub(paren_pattern, r'\1*\1*\1', expr_str)
        
        return expr_str

    def _ifelse_to_selp(self, expr_str: str) -> str:
        """Replace conditional expressions with ``selp`` calls.

        Parameters
        ----------
        expr_str
            Source string to rewrite.

        Returns
        -------
        str
            Source string with ternaries replaced by ``selp`` calls.
        """
        return re.sub(
            r"\s+(.+?)\sif\s+(.+?)\s+else\s+(.+)",
            r"selp(\2, \1, \3)",
            expr_str,
        )

    def _print_Function(self, expr: sp.Function) -> str:
        """Print a function call with CUDA-specific substitutions.

        Parameters
        ----------
        expr
            Function expression to render.

        Returns
        -------
        str
            Printed representation that accounts for CUDA-specific mappings.

        Notes
        -----
        Lookup precedence is:

        1. Known CUDA-mapped SymPy functions supplied in ``CUDA_FUNCTIONS``.
        2. User-defined functions recorded in ``__function_aliases__``.
        3. Derivative helper functions prefixed with ``"d_"``.
        4. Fallback to the SymPy function name.
        """
        func_name = expr.func.__name__

        # CUDA-known functions first
        if func_name in self.cuda_functions:
            cuda_func = self.cuda_functions[func_name]
            args = [self._print(arg) for arg in expr.args]
            return f"{cuda_func}({', '.join(args)})"

        # User-defined functions that were underscored during parsing
        if func_name in self.func_aliases:
            real_name = self.func_aliases[func_name]
            args = [self._print(arg) for arg in expr.args]
            return f"{real_name}({', '.join(args)})"

        # Derivative user functions d_<name>(...): print as-is
        if func_name.startswith('d_'):
            args = [self._print(arg) for arg in expr.args]
            return f"{func_name}({', '.join(args)})"

        # Fallback: print a plain function call to avoid PrintMethodNotImplementedError
        args = [self._print(arg) for arg in expr.args]
        return f"{func_name}({', '.join(args)})"

    def _print_Float(self, expr: sp.Float) -> str:
        """Print a floating-point literal wrapped with precision().

        Parameters
        ----------
        expr
            Float expression to render.

        Returns
        -------
        str
            Precision-wrapped representation: ``precision(value)``.
            When in index context, returns unwrapped value.
        """

        # Leave small round floats intact for replace_powers optimisation
        if self._in_pow:
            if (expr == sp.Float(2.0) or expr == sp.Float(3.0)):
                return super()._print_Float(expr)
        return f"precision({str(expr)})"

    def _print_Rational(self, expr: sp.Rational) -> str:
        """Print a rational number literal wrapped with precision().

        Parameters
        ----------
        expr
            Rational expression to render.

        Returns
        -------
        str
            Precision-wrapped representation: ``precision(p/q)``.
            When in index context, returns unwrapped value.

        Notes
        -----
        The rational is printed as ``p/q`` where ``p`` and ``q`` are the
        numerator and denominator. Python evaluates this division at
        runtime, then ``precision()`` casts the result to the configured
        dtype.
        """
        return f"precision({str(expr)})"



def print_cuda(
    expr: sp.Expr,
    symbol_map: Optional[Dict[sp.Symbol, Any]] = None,
    **kwargs: Any,
) -> str:
    """Return a CUDA-oriented source string for a SymPy expression.

    Parameters
    ----------
    expr
        SymPy expression to print.
    symbol_map
        Optional symbol replacement mapping used when constructing the printer.
    **kwargs
        Additional keyword arguments forwarded to :class:`CUDAPrinter`.

    Returns
    -------
    str
        CUDA-compatible code representation of ``expr``.
    """
    printer = CUDAPrinter(symbol_map=symbol_map, **kwargs)
    return printer.doprint(expr)

def print_cuda_multiple(
    exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    symbol_map: Optional[Dict[sp.Symbol, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    """Return CUDA-friendly source strings for assignment-style expressions.

    Parameters
    ----------
    exprs
        Iterable of ``(assign_to, expression)`` pairs to print.
    symbol_map
        Optional symbol replacement mapping used when constructing the printer.
    **kwargs
        Additional keyword arguments forwarded to :class:`CUDAPrinter`.

    Returns
    -------
    List[str]
        CUDA-compatible code representation for each assignment.
    """
    printer = CUDAPrinter(symbol_map=symbol_map, **kwargs)
    lines: List[str] = []
    for assign_to, expr in exprs:
        line = printer.doprint(expr, assign_to=assign_to)
        lines.append(line)

    return lines
