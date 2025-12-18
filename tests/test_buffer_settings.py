"""Tests for BufferSettings infrastructure and buffer_registry migration."""
import pytest

from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings,
    LinearSolverLocalSizes,
    LinearSolverSliceIndices,
    LocalSizes,
    SliceIndices,
)


class TestLocalSizes:
    """Tests for LocalSizes base class."""

    def test_nonzero_returns_value_when_positive(self):
        """nonzero() should return actual value when positive."""
        sizes = LinearSolverLocalSizes(preconditioned_vec=10, temp=5)
        assert sizes.nonzero('preconditioned_vec') == 10
        assert sizes.nonzero('temp') == 5

    def test_nonzero_returns_one_when_zero(self):
        """nonzero() should return 1 for zero-sized attributes.

        This ensures cuda.local.array always gets valid size >= 1.
        """
        sizes = LinearSolverLocalSizes(preconditioned_vec=0, temp=0)
        assert sizes.nonzero('preconditioned_vec') == 1
        assert sizes.nonzero('temp') == 1


class TestSliceIndices:
    """Tests for SliceIndices base class.

    SliceIndices is abstract, tested via concrete subclasses.
    """

    def test_slice_indices_subclass_instantiation(self):
        """Concrete subclasses should instantiate properly."""
        indices = LinearSolverSliceIndices(
            preconditioned_vec=slice(0, 10),
            temp=slice(10, 20),
            local_end=20,
        )
        assert indices.preconditioned_vec == slice(0, 10)
        assert indices.temp == slice(10, 20)
        assert indices.local_end == 20


class TestBufferSettingsAbstract:
    """Tests for BufferSettings abstract base class.

    BufferSettings is abstract and tested via concrete subclasses.
    """

    def test_buffer_settings_subclass_has_required_properties(self):
        """Concrete subclasses should implement abstract properties."""
        settings = LinearSolverBufferSettings(n=10)
        # All these should be implemented
        assert isinstance(settings.shared_memory_elements, int)
        assert isinstance(settings.local_memory_elements, int)
        assert settings.local_sizes is not None
        assert settings.shared_indices is not None
