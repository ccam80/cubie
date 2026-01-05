"""Deprecated module: Renamed to BatchInputHandler.

This module exists for backward compatibility. Import from
``cubie.batchsolving.BatchInputHandler`` instead.
"""
from cubie.batchsolving.BatchInputHandler import (  # noqa: F401
    BatchInputHandler,
    BatchInputHandler as BatchGridBuilder,
    combine_grids,
    combinatorial_grid,
    extend_grid_to_array,
    generate_array,
    generate_grid,
    unique_cartesian_product,
    verbatim_grid,
)
