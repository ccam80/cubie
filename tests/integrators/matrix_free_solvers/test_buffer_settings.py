"""Tests for LinearSolverBufferSettings class."""
import pytest
from cubie.integrators.matrix_free_solvers.buffer_settings import (
    LinearSolverBufferSettings,
    LinearSolverLocalSizes,
    LinearSolverSliceIndices,
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

    def test_local_sizes_property(self):
        """local_sizes property should return LinearSolverLocalSizes."""
        settings = LinearSolverBufferSettings(n=5)
        sizes = settings.local_sizes

        assert isinstance(sizes, LinearSolverLocalSizes)
        assert sizes.preconditioned_vec == 5
        assert sizes.temp == 5

    def test_local_sizes_nonzero(self):
        """nonzero method should return 1 for zero-size attributes."""
        sizes = LinearSolverLocalSizes(preconditioned_vec=0, temp=0)
        assert sizes.nonzero('preconditioned_vec') == 1
        assert sizes.nonzero('temp') == 1

    def test_shared_indices_property(self):
        """shared_indices property should return LinearSolverSliceIndices."""
        settings = LinearSolverBufferSettings(
            n=5,
            preconditioned_vec_location='shared',
            temp_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, LinearSolverSliceIndices)
        assert indices.preconditioned_vec == slice(0, 5)
        assert indices.temp == slice(5, 10)
        assert indices.local_end == 10

    def test_shared_indices_local_gives_empty_slices(self):
        """Local buffers should get empty slices."""
        settings = LinearSolverBufferSettings(n=5)
        indices = settings.shared_indices

        assert indices.preconditioned_vec == slice(0, 0)
        assert indices.temp == slice(0, 0)
        assert indices.local_end == 0
