"""Tests for cubie.odesystems.symbolic.codegen._stage_utils."""

from __future__ import annotations

import sympy as sp

from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)


# ── prepare_stage_data ──────────────────────────────── #

def test_prepare_stage_data_sympifies_coefficient_matrix():
    """Coefficient matrix entries are converted to SymPy expressions."""
    coeffs = [[sp.Rational(1, 4), 0], [sp.Rational(1, 2), sp.Rational(1, 4)]]
    nodes = [sp.Rational(1, 4), sp.Rational(3, 4)]
    matrix, _, _ = prepare_stage_data(coeffs, nodes)
    assert matrix[0, 0] == sp.Rational(1, 4)
    assert matrix[1, 0] == sp.Rational(1, 2)
    assert matrix[1, 1] == sp.Rational(1, 4)
    assert matrix[0, 1] == sp.S(0)


def test_prepare_stage_data_sympifies_nodes():
    """Node expressions are sympified to exact SymPy objects."""
    coeffs = [[sp.Rational(1, 2)]]
    nodes = [sp.Rational(1, 2)]
    _, node_exprs, _ = prepare_stage_data(coeffs, nodes)
    assert node_exprs == (sp.Rational(1, 2),)


def test_prepare_stage_data_returns_stage_count():
    """Stage count equals the number of rows in the coefficient matrix."""
    coeffs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    nodes = [0.0, 0.5, 1.0]
    _, _, n_stages = prepare_stage_data(coeffs, nodes)
    assert n_stages == 3


# ── build_stage_metadata ────────────────────────────── #

def test_build_stage_metadata_node_symbols():
    """Node symbols are named c_0, c_1, ... with correct values."""
    coeffs = sp.Matrix([[sp.Rational(1, 4), 0], [0, sp.Rational(3, 4)]])
    nodes = (sp.Rational(1, 4), sp.Rational(3, 4))
    metadata, _, node_syms = build_stage_metadata(coeffs, nodes)
    assert len(node_syms) == 2
    assert node_syms[0] == sp.Symbol("c_0")
    assert node_syms[1] == sp.Symbol("c_1")
    # Check that metadata contains the node assignments
    node_assignments = {sym: val for sym, val in metadata if str(sym).startswith("c_")}
    assert node_assignments[sp.Symbol("c_0")] == sp.Rational(1, 4)
    assert node_assignments[sp.Symbol("c_1")] == sp.Rational(3, 4)


def test_build_stage_metadata_coeff_symbols():
    """Coefficient symbols are named a_i_j with correct values."""
    coeffs = sp.Matrix([[sp.Rational(1, 3), sp.Rational(2, 3)],
                         [sp.Rational(1, 2), sp.Rational(1, 2)]])
    nodes = (sp.S(0), sp.S(1))
    metadata, coeff_syms, _ = build_stage_metadata(coeffs, nodes)
    assert len(coeff_syms) == 2
    assert len(coeff_syms[0]) == 2
    assert coeff_syms[0][0] == sp.Symbol("a_0_0")
    assert coeff_syms[1][1] == sp.Symbol("a_1_1")
    # Check coefficient assignments in metadata
    coeff_assignments = {sym: val for sym, val in metadata if str(sym).startswith("a_")}
    assert coeff_assignments[sp.Symbol("a_0_0")] == sp.Rational(1, 3)
    assert coeff_assignments[sp.Symbol("a_0_1")] == sp.Rational(2, 3)
    assert coeff_assignments[sp.Symbol("a_1_0")] == sp.Rational(1, 2)
    assert coeff_assignments[sp.Symbol("a_1_1")] == sp.Rational(1, 2)


def test_build_stage_metadata_returns_triple():
    """Return value is (metadata_exprs, coeff_symbols, node_symbols)."""
    coeffs = sp.Matrix([[sp.S(1)]])
    nodes = (sp.S(0),)
    result = build_stage_metadata(coeffs, nodes)
    metadata, coeff_syms, node_syms = result
    # metadata is a list of (symbol, expr) tuples
    assert len(metadata) == 2  # 1 node + 1 coeff for 1x1
    assert len(coeff_syms) == 1
    assert len(coeff_syms[0]) == 1
    assert len(node_syms) == 1
