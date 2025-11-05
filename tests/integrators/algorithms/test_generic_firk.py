"""Test FIRK dynamic controller defaults selection.

This test module validates the automatic selection of step controller
defaults in :class:`~cubie.integrators.algorithms.generic_firk.FIRKStep`
based on whether the tableau has an embedded error estimate.

The test ensures that the default FIRK tableau selects appropriate
controller defaults:

- If the tableau has an error estimate: defaults to PI controller (adaptive)
- If the tableau lacks an error estimate: defaults to fixed-step controller

This automatic selection prevents users from accidentally pairing an
errorless tableau with an adaptive controller, which would fail at runtime
since adaptive controllers require error estimates.

Notes
-----
This test uses the actual default FIRK tableau and adapts assertions based
on its ``has_error_estimate`` property. This ensures the test remains valid
even if the default tableau changes in the future.
"""

import numpy as np

from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
)


def test_firk_default_tableau_has_appropriate_defaults(
    system
):
    """FIRK default tableau selects appropriate controller defaults."""

    has_error = DEFAULT_FIRK_TABLEAU.has_error_estimate

    step = FIRKStep(
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
