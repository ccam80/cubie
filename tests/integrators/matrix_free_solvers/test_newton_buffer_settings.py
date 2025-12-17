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

    def test_local_memory_elements_all_local(self):
        """All local gives delta + residual + residual_temp + stage_base_bt
        + krylov_iters."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='local',
            residual_temp_location='local',
            stage_base_bt_location='local',
        )
        # delta(10) + residual(10) + residual_temp(10) + stage_base_bt(10)
        # + krylov_iters(1)
        assert settings.local_memory_elements == 41

    def test_shared_memory_elements_all_local(self):
        """All local buffers should give zero shared memory elements."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='local',
            residual_temp_location='local',
            stage_base_bt_location='local',
        )
        assert settings.shared_memory_elements == 0

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = NewtonBufferSettings(
            n=10,
            delta_location='local',
            residual_location='shared',
        )
        assert settings.use_shared_delta is False
        assert settings.use_shared_residual is True

    def test_residual_temp_toggleability(self):
        """residual_temp_location should affect memory calculations."""
        # Default local
        settings_local = NewtonBufferSettings(n=10)
        assert settings_local.use_shared_residual_temp is False

        # Explicit shared
        settings_shared = NewtonBufferSettings(
            n=10,
            residual_temp_location='shared',
        )
        assert settings_shared.use_shared_residual_temp is True
        # Shared should have n more shared elements
        diff = (settings_shared.shared_memory_elements -
                settings_local.shared_memory_elements)
        assert diff == 10  # n elements for residual_temp

    def test_stage_base_bt_toggleability(self):
        """stage_base_bt_location should affect memory calculations."""
        # Default local
        settings_local = NewtonBufferSettings(n=10)
        assert settings_local.use_shared_stage_base_bt is False

        # Explicit shared
        settings_shared = NewtonBufferSettings(
            n=10,
            stage_base_bt_location='shared',
        )
        assert settings_shared.use_shared_stage_base_bt is True
        # Shared should have n more shared elements
        diff = (settings_shared.shared_memory_elements -
                settings_local.shared_memory_elements)
        assert diff == 10  # n elements for stage_base_bt

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
            delta=10, residual=10, residual_temp=10, stage_base_bt=10,
            krylov_iters=1
        )
        assert sizes.nonzero('delta') == 10
        assert sizes.nonzero('residual') == 10
        assert sizes.nonzero('stage_base_bt') == 10

    def test_nonzero_returns_one_when_zero(self):
        """nonzero should return 1 for zero-size attributes."""
        sizes = NewtonLocalSizes(
            delta=0, residual=0, residual_temp=0, stage_base_bt=0,
            krylov_iters=0
        )
        assert sizes.nonzero('delta') == 1
        assert sizes.nonzero('residual') == 1
        assert sizes.nonzero('residual_temp') == 1
        assert sizes.nonzero('stage_base_bt') == 1
        assert sizes.nonzero('krylov_iters') == 1


class TestNewtonSliceIndices:
    """Tests for NewtonSliceIndices class."""

    def test_slice_attributes(self):
        """SliceIndices should store slice objects."""
        indices = NewtonSliceIndices(
            delta=slice(0, 10),
            residual=slice(10, 20),
            residual_temp=slice(20, 30),
            stage_base_bt=slice(30, 40),
            local_end=40,
            lin_solver_start=40,
        )
        assert indices.delta == slice(0, 10)
        assert indices.residual == slice(10, 20)
        assert indices.residual_temp == slice(20, 30)
        assert indices.stage_base_bt == slice(30, 40)
        assert indices.local_end == 40
        assert indices.lin_solver_start == 40
