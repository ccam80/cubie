"""Tests for LinearSolverBufferSettings class."""
import pytest
from cubie.integrators.matrix_free_solvers.buffer_settings import (
    LinearSolverBufferSettings,
)


class TestLinearSolverBufferSettings:
    """Tests for LinearSolverBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should be local."""
        settings = LinearSolverBufferSettings(n=5)

        assert settings.preconditioned_vec_location == 'local'
        assert settings.temp_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = LinearSolverBufferSettings(
            n=5,
            preconditioned_vec_location='shared',
        )

        assert settings.use_shared_preconditioned_vec is True
        assert settings.use_shared_temp is False

    def test_shared_memory_elements_both_shared(self):
        """Both shared should give 2*n shared memory."""
        settings = LinearSolverBufferSettings(
            n=5,
            preconditioned_vec_location='shared',
            temp_location='shared',
        )
        assert settings.shared_memory_elements == 10

    def test_shared_memory_elements_both_local(self):
        """Both local should give 0 shared memory."""
        settings = LinearSolverBufferSettings(n=5)
        assert settings.shared_memory_elements == 0

    def test_local_memory_elements_both_local(self):
        """Both local should give 2*n local memory."""
        settings = LinearSolverBufferSettings(n=5)
        assert settings.local_memory_elements == 10

    def test_invalid_n_raises(self):
        """n < 1 should raise error."""
        with pytest.raises((ValueError, TypeError)):
            LinearSolverBufferSettings(n=0)

    def test_invalid_location_raises(self):
        """Invalid location string should raise ValueError."""
        with pytest.raises(ValueError):
            LinearSolverBufferSettings(
                n=5,
                preconditioned_vec_location='invalid'
            )
