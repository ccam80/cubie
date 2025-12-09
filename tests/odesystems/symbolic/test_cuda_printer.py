import sympy as sp
import re

from cubie.odesystems.symbolic.codegen import (
    CUDAPrinter,
    print_cuda,
    print_cuda_multiple,
)


class TestCUDAPrinter:
    """Test cases for CUDAPrinter class."""

    def test_init_default(self):
        """Test CUDAPrinter initialization with default parameters."""
        printer = CUDAPrinter()
        assert printer.symbol_map == {}
        assert isinstance(printer, CUDAPrinter)

    def test_init_with_symbol_map(self):
        """Test CUDAPrinter initialization with symbol map."""
        x = sp.Symbol("x")
        arr = sp.IndexedBase("arr")
        symbol_map = {x: arr[0]}
        printer = CUDAPrinter(symbol_map=symbol_map)
        assert printer.symbol_map == symbol_map

    def test_print_symbol_without_mapping(self):
        """Test printing symbol without symbol map."""
        printer = CUDAPrinter()
        x = sp.Symbol("x")
        result = printer._print_Symbol(x)
        assert result == "x"

    def test_print_symbol_with_mapping(self):
        """Test printing symbol with symbol map."""
        x = sp.Symbol("x")
        arr = sp.IndexedBase("arr")
        symbol_map = {x: arr[0]}
        printer = CUDAPrinter(symbol_map=symbol_map)
        result = printer._print_Symbol(x)
        assert result == "arr[0]"

    def test_replace_square_powers_simple(self):
        """Test replacement of x**2 with x*x."""
        printer = CUDAPrinter()
        expr_str = "x**2"
        result = printer._replace_square_powers(expr_str)
        assert result == "x*x"

    def test_replace_square_powers_indexed(self):
        """Test replacement of indexed variable powers."""
        printer = CUDAPrinter()
        expr_str = "arr[0]**2"
        result = printer._replace_square_powers(expr_str)
        assert result == "arr[0]*arr[0]"

    def test_replace_square_powers_multiple(self):
        """Test replacement of multiple square powers."""
        printer = CUDAPrinter()
        expr_str = "x**2 + y**2"
        result = printer._replace_square_powers(expr_str)
        assert result == "x*x + y*y"

    def test_replace_cube_powers_simple(self):
        """Test replacement of x**3 with x*x*x."""
        printer = CUDAPrinter()
        expr_str = "x**3"
        result = printer._replace_cube_powers(expr_str)
        assert result == "x*x*x"

    def test_replace_cube_powers_indexed(self):
        """Test replacement of indexed variable cube powers."""
        printer = CUDAPrinter()
        expr_str = "arr[0]**3"
        result = printer._replace_cube_powers(expr_str)
        assert result == "arr[0]*arr[0]*arr[0]"

    def test_replace_powers_with_multiplication(self):
        """Test combined power replacements."""
        printer = CUDAPrinter()
        expr_str = "x**2 + y**3 + z**2"
        result = printer._replace_powers_with_multiplication(expr_str)
        assert result == "x*x + y*y*y + z*z"

    def test_replace_powers_ignores_higher_powers(self):
        """Test that higher powers are not replaced."""
        printer = CUDAPrinter()
        expr_str = "x**4 + y**5"
        result = printer._replace_powers_with_multiplication(expr_str)
        assert result == "x**4 + y**5"

    def test_doprint_simple_expression(self):
        """Test doprint with simple expression."""
        printer = CUDAPrinter()
        x = sp.Symbol("x")
        expr = x**2 + x
        result = printer.doprint(expr)
        assert "x*x" in result

    def test_doprint_with_symbol_mapping(self):
        """Test doprint with symbol mapping."""
        x = sp.Symbol("x")
        arr = sp.IndexedBase("arr")
        symbol_map = {x: arr[0]}
        printer = CUDAPrinter(symbol_map=symbol_map)
        expr = x**2
        result = printer.doprint(expr)
        assert "arr[0]*arr[0]" in result

    def test_ifelse_to_selp(self):
        """Test if-else to select conversion."""
        printer = CUDAPrinter()
        expr_str = "  a if x > 0 else b"
        result = printer._ifelse_to_selp(expr_str)
        assert result == "selp(x > 0, a, b)"

    def test_ifelse_to_selp_complex(self):
        """Test complex if-else to select conversion."""
        printer = CUDAPrinter()
        expr_str = "  x + 1 if y < z else x - 1"
        result = printer._ifelse_to_selp(expr_str)
        assert result == "selp(y < z, x + 1, x - 1)"


class TestPrintCudaFunction:
    """Test cases for print_cuda convenience function."""

    def test_print_cuda_simple(self):
        """Test print_cuda with simple expression."""
        x = sp.Symbol("x")
        expr = x**2
        result = print_cuda(expr)
        assert "x*x" in result

    def test_print_cuda_with_symbol_map(self):
        """Test print_cuda with symbol mapping."""
        x = sp.Symbol("x")
        arr = sp.IndexedBase("arr")
        symbol_map = {x: arr[0]}
        expr = x**2
        result = print_cuda(expr, symbol_map=symbol_map)
        assert "arr[0]*arr[0]" in result

    def test_print_cuda_complex_expression(self):
        """Test print_cuda with complex expression."""
        x, y = sp.symbols("x y")
        expr = x**2 + y**3 + sp.sin(x)
        result = print_cuda(expr)
        assert "x*x" in result
        assert "y*y*y" in result
        assert "sin" in result


class TestPrintCudaMultiple:
    """Test cases for print_cuda_multiple function."""

    def test_print_cuda_multiple_simple(self):
        """Test print_cuda_multiple with simple expressions."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")
        exprs = [(a, x**2), (b, y**3)]
        result = print_cuda_multiple(exprs)
        assert len(result) == 2
        assert all(isinstance(line, str) for line in result)

    def test_print_cuda_multiple_with_symbol_map(self):
        """Test print_cuda_multiple with symbol mapping."""
        x, y = sp.symbols("x y")
        a, b = sp.symbols("a b")
        arr = sp.IndexedBase("arr")
        symbol_map = {x: arr[0], y: arr[1]}
        exprs = [(a, x**2), (b, y**3)]
        result = print_cuda_multiple(exprs, symbol_map=symbol_map)
        assert len(result) == 2
        # Check that symbol mapping was applied
        combined = " ".join(result)
        assert "arr[0]" in combined
        assert "arr[1]" in combined

    def test_print_cuda_multiple_empty(self):
        """Test print_cuda_multiple with empty list."""
        result = print_cuda_multiple([])
        assert result == []

    def test_print_cuda_multiple_single(self):
        """Test print_cuda_multiple with single expression."""
        x = sp.Symbol("x")
        a = sp.Symbol("a")
        exprs = [(a, x**2)]
        result = print_cuda_multiple(exprs)
        assert len(result) == 1
        assert "x*x" in result[0]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_power_replacement_with_whitespace(self):
        """Test power replacement handles whitespace correctly."""
        printer = CUDAPrinter()
        expr_str = "x **2 + y** 3"
        # Should now replace despite whitespace
        result = printer._replace_powers_with_multiplication(expr_str)
        assert result == "x*x + y*y*y"

    def test_power_replacement_with_tabs(self):
        """Test power replacement handles tabs correctly."""
        printer = CUDAPrinter()
        expr_str = "x\t**\t2 + y **\t3"
        # Should replace despite tabs
        result = printer._replace_powers_with_multiplication(expr_str)
        assert result == "x*x + y*y*y"

    def test_power_replacement_mixed_whitespace(self):
        """Test power replacement with mixed spaces and tabs."""
        printer = CUDAPrinter()
        expr_str = "arr[0] \t** \t2 + var **  3"
        result = printer._replace_powers_with_multiplication(expr_str)
        assert result == "arr[0]*arr[0] + var*var*var"

    def test_nested_indexed_variables(self):
        """Test handling of nested indexed variables."""
        printer = CUDAPrinter()
        expr_str = "matrix[i][j]**2"
        result = printer._replace_square_powers(expr_str)
        # Should handle nested indexing properly
        assert "matrix[i][j]*matrix[i][j]" in result

    def test_symbol_map_with_complex_indexing(self):
        """Test symbol mapping with complex indexed expressions."""
        x = sp.Symbol("x")
        i, j = sp.symbols("i j")
        matrix = sp.IndexedBase('matrix')
        symbol_map = {x: matrix[i, j]}
        printer = CUDAPrinter(symbol_map=symbol_map)
        result = printer._print_Symbol(x)
        assert "matrix[i, j]" in result

def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_piecewise_assignment_is_wrapped_outside():
    aux_4, aux_2 = sp.symbols('aux_4 aux_2')
    _cse1, _cse2, _cse3 = sp.symbols('_cse1 _cse2 _cse3')
    expr = sp.Piecewise((_cse1 * (_cse2 + aux_2), _cse3), (0.0, True))

    p = CUDAPrinter()
    out = p.doprint(expr, assign_to=aux_4)

    # Expect: aux_4 = (_cse1*(_cse2 + aux_2) if _cse3 else (precision(0)))
    # Note: 0 is wrapped with precision() for type safety
    assert _compact(out) == _compact("aux_4 = (_cse1*(_cse2 + aux_2) if "
                                     "_cse3 else (precision(0.0)))")


def test_piecewise_inside_expression_assignment():
    _cse10, _cse1, _cse3, E_v = sp.symbols('_cse10 _cse1 _cse3 E_v')
    expr = E_v * sp.Piecewise((_cse1, _cse3), (0.0, True))

    p = CUDAPrinter()
    out = p.doprint(expr, assign_to=_cse10)

    # Expect: _cse10 = E_v*(_cse1 if _cse3 else (precision(0)))
    # Note: 0 is wrapped with precision() for type safety
    assert _compact(out) == _compact("_cse10 = E_v*(_cse1 if _cse3 else ("
                                     "precision(0.0)))")

def test_functions():
    """Test that expressions containing SymPy functions are successfully
    converted to CUDA-compatible functions as given by CUDA_FUNCTIONS"""
    pass


class TestNumericPrecisionWrapping:
    """Test cases for numeric literal precision wrapping."""

    def test_print_float_wrapped(self):
        """Test that Float literals are wrapped with precision()."""
        printer = CUDAPrinter()
        expr = sp.Float(0.5)
        result = printer.doprint(expr)
        # SymPy uses full precision representation
        assert result.startswith("precision(")
        assert result.endswith(")")
        assert "0.5" in result or "0.500000" in result

    def test_print_rational_wrapped(self):
        """Test that Rational literals are wrapped with precision()."""
        printer = CUDAPrinter()
        expr = sp.Rational(1, 2)
        result = printer.doprint(expr)
        assert result == "precision(1/2)"

    def test_expression_with_literals(self):
        """Test that all literals in expressions are wrapped."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        expr = x + sp.Float(1.0)
        result = printer.doprint(expr)
        # Verify literal is wrapped (SymPy may use full precision)
        assert "precision(" in result
        assert "x" in result
        # Check structure: x + precision(...)
        assert "+" in result

    def test_negative_float(self):
        """Test negative float literals are wrapped correctly."""
        printer = CUDAPrinter()
        expr = sp.Float(-0.5)
        result = printer.doprint(expr)
        assert result.startswith("precision(-")
        assert "-0.5" in result or "-0.500000" in result

    def test_scientific_notation(self):
        """Test scientific notation floats are wrapped correctly."""
        printer = CUDAPrinter()
        expr = sp.Float(1.5e-10)
        result = printer.doprint(expr)
        assert result.startswith("precision(")
        assert "e-" in result.lower() or "e-0" in result.lower()

    def test_piecewise_with_literals(self):
        """Test that literals in Piecewise expressions are wrapped."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        # Piecewise((0.5, x > 0), (0, True))
        expr = sp.Piecewise((sp.Float(0.5), x > sp.Float(0.0)),
                            (sp.Float(0.0), True))
        result = printer.doprint(expr)
        # Count precision() calls - should have at least 2
        # (one for 0.5 and one for 0 in the condition)
        assert result.count("precision(") >= 2
        assert "if" in result  # Piecewise uses ternary

    def test_zero_literal(self):
        """Test zero literal is wrapped."""
        printer = CUDAPrinter()
        expr = sp.Float(0.0)
        result = printer.doprint(expr)
        assert result == "precision(0.0)"

    def test_rational_negative(self):
        """Test negative rational literals are wrapped."""
        printer = CUDAPrinter()
        expr = sp.Rational(-1, 3)
        result = printer.doprint(expr)
        assert result == "precision(-1/3)"

    def test_mixed_expression_with_power_replacement(self):
        """Test literals are wrapped and power replacement works."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        # x**2 + 0.5 should become x*x + precision(0.5...)
        expr = x**2 + sp.Float(0.5)
        result = printer.doprint(expr)
        assert "x*x" in result  # Power replacement happens
        assert "precision(" in result  # Literal wrapping happens
        # Count precision calls - should be 1 (for the 0.5)
        # The exponent 2 is NOT wrapped (handled by _print_Pow)
        assert result.count("precision(") == 1

    def test_indexed_indices_not_wrapped(self):
        """Test that array indices are NOT wrapped with precision()."""
        printer = CUDAPrinter()
        arr = sp.IndexedBase('state')
        # Test simple index
        expr1 = arr[0]
        result1 = printer.doprint(expr1)
        assert result1 == "state[0]"
        assert "precision" not in result1
        
        # Test multi-dimensional index
        expr2 = arr[sp.Integer(0), sp.Integer(1)]
        result2 = printer.doprint(expr2)
        assert result2 == "state[0, 1]"
        assert "precision" not in result2

    def test_power_exponents_not_wrapped(self):
        """Test that power exponents are NOT wrapped to preserve optimization."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        
        # Test integer exponent (optimizable to x*x)
        expr1 = x ** sp.Float(2.0)
        result1 = printer.doprint(expr1)
        assert result1 == "x*x"  # Power replacement works
        assert "precision" not in result1
        
        # Test cube exponent (optimizable to x*x*x)
        expr2 = x ** sp.Integer(3)
        result2 = printer.doprint(expr2)
        assert result2 == "x*x*x"  # Power replacement works
        assert "precision" not in result2
        
    def test_indexed_with_literal_expression(self):
        """Test indexed expressions with literals in non-index positions."""
        printer = CUDAPrinter()
        arr = sp.IndexedBase('state')
        
        # arr[0] * 2.5 - literal should be wrapped, index should not
        expr = arr[0] * sp.Float(2.5)
        result = printer.doprint(expr)
        assert "state[0]" in result  # Index not wrapped
        assert "precision(" in result  # Literal is wrapped
        # Only one precision call for the 2.5
        assert result.count("precision(") == 1

    def test_index_arithmetic_not_wrapped(self):
        """Test that arithmetic in array indices is not wrapped with precision()."""
        printer = CUDAPrinter()
        arr = sp.IndexedBase('state')
        i = sp.Symbol('i')
        
        # Test i + 1 as index
        expr1 = arr[i + 1]
        result1 = printer.doprint(expr1)
        assert result1 == "state[i + 1]"
        assert "precision" not in result1
        
        # Test 2*i + 1 as index
        expr2 = arr[2*i + 1]
        result2 = printer.doprint(expr2)
        assert result2 == "state[2*i + 1]"
        assert "precision" not in result2
        
        # Test that the same expression outside an index IS wrapped
        expr3 = sp.Float(2)*i + sp.Integer(1)
        result3 = printer.doprint(expr3)
        assert "precision(" in result3

    def test_integer_literals_wrapped(self):
        """Test that integer literals in expressions are wrapped with precision()."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        
        # Test integer addition
        expr1 = x + sp.Integer(5)
        result1 = printer.doprint(expr1)
        assert result1 == "x + precision(5)"
        
        # Test integer multiplication
        expr2 = sp.Integer(2) * x
        result2 = printer.doprint(expr2)
        assert result2 == "precision(2)*x"
        
        # Test integer subtraction (SymPy treats x - 3 as x + (-3))
        expr3 = x - sp.Integer(3)
        result3 = printer.doprint(expr3)
        assert result3 == "x + precision(-3)"
        
    def test_integer_constants_not_upcasted(self):
        """Test that integer constants prevent float64 upcasting."""
        printer = CUDAPrinter()
        x = sp.Symbol('x')
        
        # Ensure integer constants are wrapped with precision()
        # This prevents mixed-precision arithmetic from upcasting to float64
        expr = sp.Integer(2) * x + sp.Integer(1)
        result = printer.doprint(expr)
        assert result == "precision(2)*x + precision(1)"
        # Verify no int32() usage
        assert "int32" not in result
