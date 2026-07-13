"""Exact linear algebra and SymPy-primitive tests."""

import sympy as sp

from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss_update_virtual_colswap_clil,
    exactdiv,
    nullspace_rank,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    as_small_int,
    fixpoint_sub,
    linear_expansion,
    lower_varname,
    solve_linear,
    total_derivative,
)


def dense_from_clil(mm):
    rows, cols = mm.size()
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for c, v in zip(mm.row_cols[i], mm.row_vals[i]):
            out[i][c] = v
    return out


class TestClil:
    def test_exactdiv_raises_on_remainder(self):
        assert exactdiv(6, 3) == 2
        try:
            exactdiv(7, 3)
        except AssertionError:
            pass
        else:
            raise AssertionError("inexact division did not raise")

    def test_elimination_step_matches_dense_bareiss(self):
        # M = [[2, 1, 0], [4, 3, 1], [6, 1, 2]]; eliminate col 0
        # with pivot M[0][0]=2, last_pivot=1: row_i <- (2*row_i -
        # coeff*row_0) / 1.
        mm = SparseMatrixCLIL(
            3,
            3,
            [0, 1, 2],
            [[0, 1], [0, 1, 2], [0, 1, 2]],
            [[2, 1], [4, 3, 1], [6, 1, 2]],
        )
        bareiss_update_virtual_colswap_clil(mm, 0, 0, 2, 1)
        dense = dense_from_clil(mm)
        assert dense[1] == [0, 2, 2]
        assert dense[2] == [0, -4, 4]

    def test_pivot_equal_optimization_skips_disjoint_rows(self):
        mm = SparseMatrixCLIL(
            2, 3, [0, 1], [[0], [1, 2]], [[1], [5, 7]]
        )
        bareiss_update_virtual_colswap_clil(mm, 0, 0, 1, 1)
        assert mm.row_vals[1] == [5, 7]

    def test_nullspace_rank(self):
        col_order = []
        rank = nullspace_rank(
            [[1, 2, 3], [2, 4, 6], [0, 1, 1]], col_order
        )
        assert rank == 2
        assert len(col_order) == 3

    def test_nullspace_rank_full(self):
        assert nullspace_rank([[1, 0], [0, 1]]) == 2
        assert nullspace_rank([[0, 0], [0, 0]]) == 0


class TestLinearExpansion:
    x, y, k = sp.symbols("x y k", real=True)

    def test_simple_linear(self):
        a, b, lin = linear_expansion(2 * self.x + self.y, self.x)
        assert lin and a == 2 and b == self.y

    def test_symbolic_coefficient(self):
        a, b, lin = linear_expansion(
            self.k * self.x + 1, self.x
        )
        assert lin and a == self.k and b == 1

    def test_nonlinear_power(self):
        _, _, lin = linear_expansion(self.x**2, self.x)
        assert not lin

    def test_nonlinear_function(self):
        _, _, lin = linear_expansion(
            sp.sin(self.x) + self.y, self.x
        )
        assert not lin

    def test_product_of_var_terms_nonlinear(self):
        _, _, lin = linear_expansion(
            self.x * (self.x + 1), self.x
        )
        assert not lin

    def test_absent_variable(self):
        a, b, lin = linear_expansion(self.y + 1, self.x)
        assert lin and a == 0 and b == self.y + 1

    def test_solve_linear(self):
        sol = solve_linear(
            sp.S.Zero, 2 * self.x - self.y, self.x
        )
        assert sp.simplify(sol - self.y / 2) == 0

    def test_solve_linear_singular(self):
        assert solve_linear(self.y, self.y, self.x) is None


class TestSymbolics:
    t = sp.Symbol("t", real=True)

    def test_fixpoint_sub_chains(self):
        a, b, c = sp.symbols("a b c", real=True)
        result = fixpoint_sub(a, {a: b + 1, b: c})
        assert result == c + 1

    def test_as_small_int(self):
        assert as_small_int(sp.Integer(-5)) == -5
        assert as_small_int(sp.Float(3.0)) == 3
        assert as_small_int(sp.Integer(1000)) is None
        assert as_small_int(sp.Rational(1, 2)) is None
        assert as_small_int(sp.Symbol("q")) is None

    def test_total_derivative(self):
        x, dx, w = sp.symbols("x dx_sym w", real=True)
        expr = x**2 + self.t * w
        result = total_derivative(expr, {x: dx}, self.t)
        assert sp.simplify(result - (2 * x * dx + w)) == 0

    def test_total_derivative_known_map(self):
        x, dx, drv = sp.symbols("x dx_sym drv", real=True)
        result = total_derivative(
            x * drv, {x: dx}, self.t, {drv: sp.S.One}
        )
        assert sp.simplify(result - (dx * drv + x)) == 0

    def test_registry_chain_and_rename(self):
        x = sp.Symbol("x", real=True)
        reg = DerivativeRegistry({"x", "t"})
        d1 = reg.derivative(x)
        d2 = reg.derivative(d1)
        assert reg.base_and_order(d2) == (x, 2)
        assert reg.lower_order(d2) == d1
        x_t = sp.Symbol("x_t", real=True)
        reg.rename(d1, x_t)
        assert reg.lower_order(d2) == x_t
        # x_t becomes an ordinary chain root (diff2term semantics).
        assert reg.base_and_order(d2) == (x_t, 1)
        assert reg.base_and_order(x_t) == (x_t, 0)

    def test_lower_varname_collision(self):
        reserved = {"x_t"}
        assert lower_varname("x", 1, reserved) == "x_t_"
        assert lower_varname("x", 2, reserved) == "x_tt"
