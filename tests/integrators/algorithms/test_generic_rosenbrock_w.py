"""Tests for cubie.integrators.algorithms.generic_rosenbrock_w."""

import attrs
import numpy as np

from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    ROS3P_TABLEAU,
)


def test_errorless_tableau_selects_fixed_controller_defaults():
    """A Rosenbrock tableau without an error estimate selects the fixed

    step-controller defaults instead of the adaptive PI defaults.
    """
    errorless_tableau = attrs.evolve(ROS3P_TABLEAU, b_hat=None)
    assert errorless_tableau.has_error_estimate is False

    step = GenericRosenbrockWStep(
        precision=np.float32, n=3, tableau=errorless_tableau,
    )
    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "fixed"


def test_cached_auxiliary_count_lazily_builds_helpers(precision, system):
    """cached_auxiliary_count builds the implicit helper chain on first

    access when it has not been built yet, then returns the cached
    value.
    """
    step = GenericRosenbrockWStep(
        precision=precision,
        n=system.sizes.states,
        evaluate_f=system.evaluate_f,
        evaluate_observables=system.evaluate_observables,
        get_solver_helper_fn=system.get_solver_helper,
        tableau=DEFAULT_ROSENBROCK_TABLEAU,
    )
    assert step._cached_auxiliary_count is None

    count = step.cached_auxiliary_count

    assert isinstance(count, int)
    assert step._cached_auxiliary_count == count
    # Second access reuses the cached value without rebuilding.
    assert step.cached_auxiliary_count == count
