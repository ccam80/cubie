"""Tests for save count calculation logic."""
import pytest
import numpy as np
from cubie.integrators.loops.ode_loop_config import ODELoopConfig


class TestSaveCountCalculation:
    """Test ODELoopConfig.calculate_n_saves static method."""
    
    def test_exact_multiple(self):
        """Save count for exact multiple of dt_save."""
        n_saves = ODELoopConfig.calculate_n_saves(1.0, 0.1)
        assert n_saves == 11  # t=0.0, 0.1, 0.2, ..., 0.9, 1.0
    
    def test_non_divisible_duration(self):
        """Save count for non-divisible duration."""
        n_saves = ODELoopConfig.calculate_n_saves(1.23, 0.1)
        assert n_saves == 14  # t=0.0, 0.1, ..., 1.2, 1.23
    
    def test_very_small_dt_save(self):
        """Save count with very small dt_save."""
        n_saves = ODELoopConfig.calculate_n_saves(1.0, 0.001)
        assert n_saves == 1001
    
    def test_near_integer_multiple_below(self):
        """Save count for duration just below integer multiple."""
        n_saves = ODELoopConfig.calculate_n_saves(1.0 - 1e-10, 0.1)
        assert n_saves == 11
    
    def test_near_integer_multiple_above(self):
        """Save count for duration just above integer multiple."""
        n_saves = ODELoopConfig.calculate_n_saves(1.0 + 1e-10, 0.1)
        # May be 11 or 12 depending on float precision - both acceptable
        assert n_saves in [11, 12]
    
    def test_large_duration(self):
        """Save count for large duration."""
        n_saves = ODELoopConfig.calculate_n_saves(100.0, 0.1)
        assert n_saves == 1001
    
    def test_fractional_duration(self):
        """Save count for fractional duration."""
        n_saves = ODELoopConfig.calculate_n_saves(0.5, 0.1)
        assert n_saves == 6  # t=0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    
    def test_duration_less_than_dt_save(self):
        """Save count when duration < dt_save."""
        n_saves = ODELoopConfig.calculate_n_saves(0.05, 0.1)
        assert n_saves == 2  # t=0.0 and t=0.05 (still need both endpoints)
    
    @pytest.mark.parametrize("duration,dt_save,expected_min", [
        (1.0, 0.1, 11),
        (2.0, 0.2, 11),
        (0.5, 0.05, 11),
        (10.0, 1.0, 11),
    ])
    def test_multiple_exact_divisions(self, duration, dt_save, expected_min):
        """Test various exact divisions all have at least expected saves."""
        n_saves = ODELoopConfig.calculate_n_saves(duration, dt_save)
        assert n_saves >= expected_min
        # Check it's the correct formula
        from math import ceil
        expected = int(ceil(np.float64(duration) / dt_save)) + 1
        assert n_saves == expected
