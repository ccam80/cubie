import sympy as sp

from cubie.systemmodels.symbolic.numba_cuda_printer import (
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
        assert result == "cuda.selp(x > 0, a, b)"

    def test_ifelse_to_selp_complex(self):
        """Test complex if-else to select conversion."""
        printer = CUDAPrinter()
        expr_str = "  x + 1 if y < z else x - 1"
        result = printer._ifelse_to_selp(expr_str)
        assert result == "cuda.selp(y < z, x + 1, x - 1)"


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
