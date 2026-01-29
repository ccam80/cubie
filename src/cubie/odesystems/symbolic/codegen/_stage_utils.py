"""Shared helpers for FIRK stage metadata preparation.

Published Functions
-------------------
:func:`prepare_stage_data`
    Sympify Butcher tableau coefficients and nodes into SymPy objects.

:func:`build_stage_metadata`
    Create symbol assignments for FIRK coefficients and nodes for use
    in generated CUDA code.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Uses these helpers when generating multi-stage operators.
:mod:`cubie.odesystems.symbolic.codegen.nonlinear_residuals`
    Uses these helpers when generating multi-stage residuals.
:mod:`cubie.odesystems.symbolic.codegen.preconditioners`
    Uses these helpers when generating multi-stage preconditioners.
"""

from typing import List, Sequence, Tuple, Union

import sympy as sp


def prepare_stage_data(
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
) -> Tuple[sp.Matrix, Tuple[sp.Expr, ...], int]:
    """Normalise FIRK tableau metadata for code generation.

    Parameters
    ----------
    stage_coefficients
        Butcher tableau A matrix as nested sequences.
    stage_nodes
        Butcher tableau c vector (quadrature nodes).

    Returns
    -------
    tuple[sp.Matrix, tuple[sp.Expr, ...], int]
        Sympified coefficient matrix, node expressions, and stage count.
    """

    coeff_matrix = sp.Matrix(stage_coefficients).applyfunc(sp.S)
    node_exprs = tuple(sp.S(node) for node in stage_nodes)
    return coeff_matrix, node_exprs, coeff_matrix.rows


def build_stage_metadata(
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    List[List[sp.Symbol]],
    List[sp.Symbol],
]:
    """Create symbol assignments for FIRK coefficients and nodes.

    Parameters
    ----------
    stage_coefficients
        Sympified Butcher tableau A matrix.
    stage_nodes
        Sympified Butcher tableau c vector.

    Returns
    -------
    tuple[list, list[list], list]
        Tuple of (metadata assignments, coefficient symbols by stage,
        node symbols) for use in generated CUDA code.
    """

    stage_count = stage_coefficients.rows
    coeff_symbols: List[List[sp.Symbol]] = []
    node_symbols: List[sp.Symbol] = []
    metadata_exprs: List[Tuple[sp.Symbol, sp.Expr]] = []
    for stage_idx in range(stage_count):
        node_symbol = sp.Symbol(f"c_{stage_idx}")
        node_symbols.append(node_symbol)
        metadata_exprs.append((node_symbol, stage_nodes[stage_idx]))
        stage_row: List[sp.Symbol] = []
        for col_idx in range(stage_count):
            coeff_symbol = sp.Symbol(f"a_{stage_idx}_{col_idx}")
            stage_row.append(coeff_symbol)
            metadata_exprs.append(
                (coeff_symbol, stage_coefficients[stage_idx, col_idx])
            )
        coeff_symbols.append(stage_row)
    return metadata_exprs, coeff_symbols, node_symbols


__all__ = ["prepare_stage_data", "build_stage_metadata"]
