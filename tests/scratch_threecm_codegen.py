"""Generate FIRK helpers for the three-chamber cardiovascular model."""

import numpy as np

from cubie.integrators.algorithms.generic_dirk_tableaus import (
    LOBATTO_IIIC_3_TABLEAU,
)
from tests.system_fixtures import build_three_chamber_system


def main() -> None:
    """Emit FIRK solver helpers for manual inspection."""

    system = build_three_chamber_system(np.float64)
    system.build()
    tableau = LOBATTO_IIIC_3_TABLEAU
    stage_coefficients = [list(row) for row in tableau.a]
    stage_nodes = list(tableau.c)
    system.get_solver_helper(
        "n_stage_residual",
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    system.get_solver_helper(
        "n_stage_linear_operator",
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    output_path = system.gen_file.file_path
    print(f"Generated helpers at {output_path}")


if __name__ == "__main__":
    main()
