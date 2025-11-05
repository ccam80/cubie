"""Test DIRK dynamic controller defaults selection.

This test module validates the automatic selection of step controller
defaults in :class:`~cubie.integrators.algorithms.generic_dirk.DIRKStep`
based on whether the tableau has an embedded error estimate.

The test ensures that the default DIRK tableau selects appropriate
controller defaults:

- If the tableau has an error estimate: defaults to PI controller (adaptive)
- If the tableau lacks an error estimate: defaults to fixed-step controller

This automatic selection prevents users from accidentally pairing an
errorless tableau with an adaptive controller, which would fail at runtime
since adaptive controllers require error estimates.

Notes
-----
This test uses the actual default DIRK tableau and adapts assertions based
on its ``has_error_estimate`` property. This ensures the test remains valid
even if the default tableau changes in the future.
"""

import numpy as np

from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
)


def test_dirk_default_tableau_has_appropriate_defaults(
    system
):
    """DIRK default tableau selects appropriate controller defaults."""

    has_error = DEFAULT_DIRK_TABLEAU.has_error_estimate

    step = DIRKStep(
        precision=np.float32,
        n=3,
        dt=None,
        get_solver_helper_fn=system.solver_helper,
    )

    defaults = step.controller_defaults.step_controller

    if has_error:
        assert defaults["step_controller"] == "pi"
    else:
        assert defaults["step_controller"] == "fixed"
