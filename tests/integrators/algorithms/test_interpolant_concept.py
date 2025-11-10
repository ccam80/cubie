"""Proof-of-concept test demonstrating interpolant evaluation.

This test shows how the trapezoidal DIRK tableau's interpolant
coefficients could be used to compute intermediate values within a step.
"""
import numpy as np
import pytest

from cubie.integrators.algorithms.generic_dirk_tableaus import (
    TRAPEZOIDAL_DIRK_TABLEAU,
)


def test_trapezoidal_has_interpolant():
    """Verify trapezoidal tableau has interpolant support."""
    assert TRAPEZOIDAL_DIRK_TABLEAU.has_interpolant
    assert TRAPEZOIDAL_DIRK_TABLEAU.b_interp is not None
    
    b_interp = TRAPEZOIDAL_DIRK_TABLEAU.b_interp
    assert len(b_interp) >= 2
    assert len(b_interp[0]) == 2
    assert len(b_interp[1]) == 2


def test_interpolant_evaluation_concept():
    """Demonstrate how interpolant coefficients would be evaluated.
    
    This is a proof-of-concept showing the evaluation pattern that would
    be used in the device code, not a validation of correctness.
    """
    tableau = TRAPEZOIDAL_DIRK_TABLEAU
    b_interp = tableau.b_interp
    
    t = 0.0
    dt = 0.1
    theta = 0.3
    
    y_start = np.array([1.0])
    k_0 = np.array([0.5])
    k_1 = np.array([0.4])
    
    stage_derivatives = [k_0, k_1]
    stage_count = 2
    
    y_interp = y_start.copy()
    
    for stage_idx in range(stage_count):
        theta_power = 1.0
        weight = 0.0
        for coeff_row in b_interp:
            weight += coeff_row[stage_idx] * theta_power
            theta_power *= theta
        y_interp += dt * weight * stage_derivatives[stage_idx]
    
    assert y_interp.shape == y_start.shape
    
    # For linear interpolation (current placeholder)
    expected = y_start + dt * (
        (b_interp[0][0] + theta * b_interp[1][0]) * k_0 +
        (b_interp[0][1] + theta * b_interp[1][1]) * k_1
    )
    np.testing.assert_allclose(y_interp, expected, rtol=1e-15)


def test_interpolant_at_boundaries():
    """Verify interpolant boundary conditions for trapezoidal method.
    
    At theta=0, interpolant should give y(t).
    At theta=1, interpolant should give y(t+dt) = y(t) + dt*(b[0]*k_0 + 
    b[1]*k_1).
    
    NOTE: Currently using linear interpolation placeholder, so this test
    validates that the placeholder coefficients are consistent.
    """
    tableau = TRAPEZOIDAL_DIRK_TABLEAU
    b_interp = tableau.b_interp
    b = tableau.b
    
    dt = 0.1
    y_start = np.array([1.0])
    k_0 = np.array([0.5])
    k_1 = np.array([0.4])
    stage_derivatives = [k_0, k_1]
    
    theta = 0.0
    y_theta_0 = y_start.copy()
    for stage_idx in range(2):
        theta_power = 1.0
        weight = 0.0
        for coeff_row in b_interp:
            weight += coeff_row[stage_idx] * theta_power
            theta_power *= theta
        y_theta_0 += dt * weight * stage_derivatives[stage_idx]
    
    np.testing.assert_allclose(y_theta_0, y_start, rtol=1e-15)
    
    theta = 1.0
    y_theta_1 = y_start.copy()
    for stage_idx in range(2):
        theta_power = 1.0
        weight = 0.0
        for coeff_row in b_interp:
            weight += coeff_row[stage_idx] * theta_power
            theta_power *= theta
        y_theta_1 += dt * weight * stage_derivatives[stage_idx]
    
    # With linear interpolation (current placeholder), at theta=1 we get b
    y_end_expected = y_start + dt * (b[0] * k_0 + b[1] * k_1)
    np.testing.assert_allclose(y_theta_1, y_end_expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
