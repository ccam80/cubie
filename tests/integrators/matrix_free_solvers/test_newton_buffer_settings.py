"""Tests for Newton solver BufferSettings."""
import pytest
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonBufferSettings,
    NewtonLocalSizes,
    NewtonSliceIndices,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings,
)


class TestNewtonBufferSettings:
    """Tests for NewtonBufferSettings initialization and properties."""

    def test_shared_memory_elements_default(self):
        """Default: delta and residual shared gives 2*n."""
        settings = NewtonBufferSettings(n=10)
        assert settings.shared_memory_elements == 20  # 2 * n

    def test_shared_memory_elements_with_linear_solver(self):
        """Including linear solver adds its shared memory."""
        lin_settings = LinearSolverBufferSettings(
            n=10,
            preconditioned_vec_location='shared',
            temp_location='shared',
        )
        settings = NewtonBufferSettings(
            n=10,
            linear_solver_buffer_settings=lin_settings,
        )
        # 2*n (newton) + lin_solver shared (2*n for all-shared)
        expected = 20 + lin_settings.shared_memory_elements
        assert settings.shared_memory_elements == expected

    def test_local_memory_elements_default(self):
        """Default shared buffers give residual_temp + krylov_iters."""
        settings = NewtonBufferSettings(n=10)
        # residual_temp (n) + krylov_iters (1) = 11
        assert settings.local_memory_elements == 11

    def test_local_memory_elements_all_local(self):
        """All local gives delta + residual + residual_temp + krylov_iters."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='local',
        )
        # delta (10) + residual (10) + residual_temp (10) + krylov_iters (1)
        assert settings.local_memory_elements == 31

    def test_shared_indices_contiguous(self):
        """Shared memory slices should be contiguous."""
        settings = NewtonBufferSettings(n=10)
        indices = settings.shared_indices
        assert indices.delta.stop == indices.residual.start
        assert indices.local_end == indices.residual.stop

    def test_shared_indices_all_local_gives_empty(self):
        """Local buffers get empty slices."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='local',
        )
        indices = settings.shared_indices
        assert indices.delta == slice(0, 0)
        assert indices.residual == slice(0, 0)
        assert indices.local_end == 0

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='shared',
        )
        assert settings.use_shared_delta is False
        assert settings.use_shared_residual is True

    def test_lin_solver_start_matches_local_end(self):
        """lin_solver_start should equal local_end."""
        settings = NewtonBufferSettings(n=10)
        indices = settings.shared_indices
        assert indices.lin_solver_start == indices.local_end

    def test_invalid_n_raises(self):
        """n < 1 should raise error."""
        with pytest.raises((ValueError, TypeError)):
            NewtonBufferSettings(n=0)

    def test_invalid_location_raises(self):
        """Invalid location string should raise ValueError."""
        with pytest.raises(ValueError):
            NewtonBufferSettings(n=10, delta_location='invalid')


class TestNewtonLocalSizes:
    """Tests for NewtonLocalSizes class."""

    def test_nonzero_returns_value_when_positive(self):
        """nonzero should return actual value when positive."""
        sizes = NewtonLocalSizes(
            delta=10, residual=10, residual_temp=10, krylov_iters=1
        )
        assert sizes.nonzero('delta') == 10
        assert sizes.nonzero('residual') == 10

    def test_nonzero_returns_one_when_zero(self):
        """nonzero should return 1 for zero-size attributes."""
        sizes = NewtonLocalSizes(
            delta=0, residual=0, residual_temp=0, krylov_iters=0
        )
        assert sizes.nonzero('delta') == 1
        assert sizes.nonzero('residual') == 1
        assert sizes.nonzero('residual_temp') == 1
        assert sizes.nonzero('krylov_iters') == 1


class TestNewtonSliceIndices:
    """Tests for NewtonSliceIndices class."""

    def test_slice_attributes(self):
        """SliceIndices should store slice objects."""
        indices = NewtonSliceIndices(
            delta=slice(0, 10),
            residual=slice(10, 20),
            local_end=20,
            lin_solver_start=20,
        )
        assert indices.delta == slice(0, 10)
        assert indices.residual == slice(10, 20)
        assert indices.local_end == 20
        assert indices.lin_solver_start == 20
