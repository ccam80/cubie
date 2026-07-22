"""Shared helpers for FIRK stage metadata preparation.

Published Functions
-------------------
:func:`prepare_stage_data`
    Normalise Butcher tableau coefficients and nodes into IR values.

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

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import from_sympy


def _to_ir_value(value) -> ir.Expr:
    """Convert a tableau entry (number or SymPy scalar) to IR."""
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, (int, float)):
        return ir.num(value)
    return from_sympy(value)


def prepare_stage_data(
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
) -> Tuple[List[List[ir.Expr]], Tuple[ir.Expr, ...], int]:
    """Normalise FIRK tableau metadata for code generation.

    Parameters
    ----------
    stage_coefficients
        Butcher tableau A matrix as nested sequences of numbers or
        SymPy scalars.
    stage_nodes
        Butcher tableau c vector (quadrature nodes).

    Returns
    -------
    tuple[list[list[ir.Expr]], tuple[ir.Expr, ...], int]
        IR coefficient matrix rows, node expressions, and stage count.
    """
    coeff_matrix = [
        [_to_ir_value(entry) for entry in row]
        for row in stage_coefficients
    ]
    node_exprs = tuple(_to_ir_value(node) for node in stage_nodes)
    return coeff_matrix, node_exprs, len(coeff_matrix)


def build_stage_metadata(
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
) -> Tuple[
    List[Tuple[ir.Sym, ir.Expr]],
    List[List[ir.Sym]],
    List[ir.Sym],
]:
    """Create symbol assignments for FIRK coefficients and nodes.

    Parameters
    ----------
    stage_coefficients
        IR Butcher tableau A matrix rows.
    stage_nodes
        IR Butcher tableau c vector.

    Returns
    -------
    tuple[list, list[list], list]
        Tuple of (metadata assignments, coefficient symbols by stage,
        node symbols) for use in generated CUDA code.
    """
    stage_count = len(stage_coefficients)
    coeff_symbols: List[List[ir.Sym]] = []
    node_symbols: List[ir.Sym] = []
    metadata_exprs: List[Tuple[ir.Sym, ir.Expr]] = []
    for stage_idx in range(stage_count):
        node_symbol = ir.sym(f"_cubie_codegen_c_{stage_idx}")
        node_symbols.append(node_symbol)
        metadata_exprs.append((node_symbol, stage_nodes[stage_idx]))
        stage_row: List[ir.Sym] = []
        for col_idx in range(stage_count):
            coeff_symbol = ir.sym(
                f"_cubie_codegen_a_{stage_idx}_{col_idx}"
            )
            stage_row.append(coeff_symbol)
            metadata_exprs.append(
                (
                    coeff_symbol,
                    stage_coefficients[stage_idx][col_idx],
                )
            )
        coeff_symbols.append(stage_row)
    return metadata_exprs, coeff_symbols, node_symbols


__all__ = ["prepare_stage_data", "build_stage_metadata"]
