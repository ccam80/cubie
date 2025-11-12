"""Shared helpers for FIRK stage metadata preparation."""

from typing import List, Sequence, Tuple, Union

import sympy as sp


def prepare_stage_data(
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
) -> Tuple[sp.Matrix, Tuple[sp.Expr, ...], int]:
    """Normalise FIRK tableau metadata for code generation."""

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
    """Create symbol assignments for FIRK coefficients and nodes."""

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
