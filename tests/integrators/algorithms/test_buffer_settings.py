"""Tests for ERKBufferSettings, DIRKBufferSettings, FIRKBufferSettings,
and RosenbrockBufferSettings classes."""
import pytest
from cubie.integrators.algorithms.generic_erk import (
    ERKBufferSettings,
    ERKLocalSizes,
    ERKSliceIndices,
)
from cubie.integrators.algorithms.generic_dirk import (
    DIRKBufferSettings,
    DIRKLocalSizes,
    DIRKSliceIndices,
)
from cubie.integrators.algorithms.generic_firk import (
    FIRKBufferSettings,
    FIRKLocalSizes,
    FIRKSliceIndices,
)
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    RosenbrockBufferSettings,
    RosenbrockLocalSizes,
    RosenbrockSliceIndices,
)


class TestERKBufferSettings:
    """Tests for ERKBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should be local for ERK buffers."""
        settings = ERKBufferSettings(n=3, stage_count=4)

        assert settings.stage_rhs_location == 'local'
        assert settings.stage_accumulator_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='shared',
        )

        assert settings.use_shared_stage_rhs is True
        assert settings.use_shared_stage_accumulator is False

    def test_stage_cache_aliases_rhs_when_rhs_shared(self):
        """stage_cache should alias rhs when rhs is shared."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='shared',
            stage_accumulator_location='local',
        )

        assert settings.use_shared_stage_cache is True
        assert settings.stage_cache_aliases_rhs is True
        assert settings.stage_cache_aliases_accumulator is False

    def test_stage_cache_aliases_accumulator_when_only_acc_shared(self):
        """stage_cache should alias accumulator when only acc is shared."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='local',
            stage_accumulator_location='shared',
        )

        assert settings.use_shared_stage_cache is True
        assert settings.stage_cache_aliases_rhs is False
        assert settings.stage_cache_aliases_accumulator is True

    def test_stage_cache_needs_persistent_local_when_both_local(self):
        """stage_cache should need persistent local when both are local."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='local',
            stage_accumulator_location='local',
        )

        assert settings.use_shared_stage_cache is False
        assert settings.persistent_local_elements == 3

    def test_accumulator_length(self):
        """Accumulator length should be (stage_count - 1) * n."""
        settings = ERKBufferSettings(n=3, stage_count=5)
        assert settings.accumulator_length == 12  # (5-1) * 3

    def test_single_stage_accumulator_length(self):
        """Single-stage method should have zero accumulator length."""
        settings = ERKBufferSettings(n=3, stage_count=1)
        assert settings.accumulator_length == 0

    def test_shared_memory_elements(self):
        """Shared memory should sum sizes of shared buffers."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='shared',
            stage_accumulator_location='shared',
        )
        # rhs (3) + accumulator ((4-1)*3=9) = 12
        assert settings.shared_memory_elements == 12

    def test_local_sizes_property(self):
        """local_sizes property should return ERKLocalSizes instance."""
        settings = ERKBufferSettings(n=3, stage_count=4)
        sizes = settings.local_sizes

        assert isinstance(sizes, ERKLocalSizes)
        assert sizes.stage_rhs == 3
        assert sizes.stage_accumulator == 9  # (4-1)*3

    def test_local_sizes_nonzero(self):
        """nonzero method should return 1 for zero-size attributes."""
        sizes = ERKLocalSizes(stage_rhs=0, stage_accumulator=0, stage_cache=0)
        assert sizes.nonzero('stage_rhs') == 1
        assert sizes.nonzero('stage_accumulator') == 1

    def test_shared_indices_property(self):
        """shared_indices property should return ERKSliceIndices instance."""
        settings = ERKBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='shared',
            stage_accumulator_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, ERKSliceIndices)
        assert indices.stage_rhs == slice(0, 3)
        assert indices.stage_accumulator == slice(3, 12)
        assert indices.local_end == 12


class TestDIRKBufferSettings:
    """Tests for DIRKBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should match all_in_one.py."""
        settings = DIRKBufferSettings(n=3, stage_count=4)

        assert settings.stage_increment_location == 'local'
        assert settings.stage_base_location == 'local'
        assert settings.accumulator_location == 'local'
        assert settings.solver_scratch_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=4,
            stage_increment_location='local',
        )

        assert settings.use_shared_stage_increment is False
        assert settings.use_shared_accumulator is False

    def test_stage_base_aliases_accumulator_multistage(self):
        """stage_base should alias accumulator when multistage and shared."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=4,
            accumulator_location='shared',
        )

        assert settings.multistage is True
        assert settings.stage_base_aliases_accumulator is True

    def test_stage_base_no_alias_single_stage(self):
        """stage_base cannot alias accumulator for single-stage."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=1,
            accumulator_location='shared',
        )

        assert settings.multistage is False
        assert settings.stage_base_aliases_accumulator is False

    def test_solver_scratch_elements(self):
        """Solver scratch should be 2 * n without newton_buffer_settings."""
        settings = DIRKBufferSettings(n=5, stage_count=3)
        assert settings.solver_scratch_elements == 10

    def test_solver_scratch_with_newton_buffer_settings(self):
        """Solver scratch should use newton_buffer_settings when provided."""
        from cubie.integrators.matrix_free_solvers.newton_krylov import (
            NewtonBufferSettings
        )
        from cubie.integrators.matrix_free_solvers.linear_solver import (
            LinearSolverBufferSettings
        )
        linear_settings = LinearSolverBufferSettings(n=5)
        newton_settings = NewtonBufferSettings(
            n=5,
            linear_solver_buffer_settings=linear_settings,
        )
        settings = DIRKBufferSettings(
            n=5,
            stage_count=3,
            newton_buffer_settings=newton_settings,
        )
        # Should use newton_buffer_settings.shared_memory_elements
        assert settings.solver_scratch_elements == (
            newton_settings.shared_memory_elements
        )

    def test_shared_memory_elements_multistage(self):
        """Shared memory should sum sizes of shared buffers."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=4,
            accumulator_location='shared',
            solver_scratch_location='shared',
            stage_increment_location='shared',
        )
        # accumulator ((4-1)*3=9) + solver (2*3=6) + increment (3) = 18
        assert settings.shared_memory_elements == 18

    def test_local_sizes_property(self):
        """local_sizes property should return DIRKLocalSizes instance."""
        settings = DIRKBufferSettings(n=3, stage_count=4)
        sizes = settings.local_sizes

        assert isinstance(sizes, DIRKLocalSizes)
        assert sizes.stage_increment == 3
        assert sizes.accumulator == 9  # (4-1)*3
        assert sizes.solver_scratch == 6  # 2*3

    def test_shared_indices_property(self):
        """shared_indices property should return DIRKSliceIndices instance."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=4,
            accumulator_location='shared',
            solver_scratch_location='shared',
            stage_increment_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, DIRKSliceIndices)
        assert indices.accumulator == slice(0, 9)
        assert indices.solver_scratch == slice(9, 15)
        assert indices.stage_increment == slice(15, 18)
        assert indices.local_end == 18


class TestFIRKBufferSettings:
    """Tests for FIRKBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should match expected values."""
        settings = FIRKBufferSettings(n=3, stage_count=4)

        assert settings.solver_scratch_location == 'local'
        assert settings.stage_increment_location == 'local'
        assert settings.stage_driver_stack_location == 'local'
        assert settings.stage_state_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            solver_scratch_location='local',
        )

        assert settings.use_shared_solver_scratch is False
        assert settings.use_shared_stage_increment is False

    def test_all_stages_n(self):
        """all_stages_n should be stage_count * n."""
        settings = FIRKBufferSettings(n=3, stage_count=4)
        assert settings.all_stages_n == 12

    def test_solver_scratch_elements(self):
        """Solver scratch should be 2 * all_stages_n without buffer settings."""
        settings = FIRKBufferSettings(n=3, stage_count=4)
        # 2 * (4 * 3) = 24
        assert settings.solver_scratch_elements == 24

    def test_solver_scratch_with_newton_buffer_settings(self):
        """Solver scratch should use newton_buffer_settings when provided."""
        from cubie.integrators.matrix_free_solvers.newton_krylov import (
            NewtonBufferSettings
        )
        from cubie.integrators.matrix_free_solvers.linear_solver import (
            LinearSolverBufferSettings
        )
        all_stages_n = 4 * 3  # stage_count * n
        linear_settings = LinearSolverBufferSettings(n=all_stages_n)
        newton_settings = NewtonBufferSettings(
            n=all_stages_n,
            linear_solver_buffer_settings=linear_settings,
        )
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            newton_buffer_settings=newton_settings,
        )
        # Should use newton_buffer_settings.shared_memory_elements
        assert settings.solver_scratch_elements == (
            newton_settings.shared_memory_elements
        )

    def test_stage_driver_stack_elements(self):
        """Stage driver stack should be stage_count * n_drivers."""
        settings = FIRKBufferSettings(n=3, stage_count=4, n_drivers=2)
        # 4 * 2 = 8
        assert settings.stage_driver_stack_elements == 8

    def test_shared_memory_elements(self):
        """Shared memory should sum sizes of shared buffers."""
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            n_drivers=2,
            solver_scratch_location='shared',
            stage_increment_location='shared',
            stage_driver_stack_location='shared',
            stage_state_location='shared',
        )
        # solver (2*12=24) + increment (12) + drivers (8) + state (3) = 47
        assert settings.shared_memory_elements == 47

    def test_local_memory_elements(self):
        """Local memory should sum sizes of local buffers."""
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            n_drivers=2,
            solver_scratch_location='local',
            stage_increment_location='local',
            stage_driver_stack_location='local',
            stage_state_location='local',
        )
        # solver (24) + increment (12) + drivers (8) + state (3) = 47
        assert settings.local_memory_elements == 47

    def test_local_sizes_property(self):
        """local_sizes property should return FIRKLocalSizes instance."""
        settings = FIRKBufferSettings(n=3, stage_count=4, n_drivers=2)
        sizes = settings.local_sizes

        assert isinstance(sizes, FIRKLocalSizes)
        assert sizes.solver_scratch == 24  # 2*4*3
        assert sizes.stage_increment == 12  # 4*3
        assert sizes.stage_driver_stack == 8  # 4*2
        assert sizes.stage_state == 3

    def test_shared_indices_property(self):
        """shared_indices property should return FIRKSliceIndices instance."""
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            n_drivers=2,
            solver_scratch_location='shared',
            stage_increment_location='shared',
            stage_driver_stack_location='shared',
            stage_state_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, FIRKSliceIndices)
        assert indices.solver_scratch == slice(0, 24)
        assert indices.stage_increment == slice(24, 36)
        assert indices.stage_driver_stack == slice(36, 44)
        assert indices.stage_state == slice(44, 47)
        assert indices.local_end == 47


class TestRosenbrockBufferSettings:
    """Tests for RosenbrockBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should be shared for all buffers."""
        settings = RosenbrockBufferSettings(n=3, stage_count=4)

        assert settings.stage_rhs_location == 'local'
        assert settings.stage_store_location == 'local'
        assert settings.cached_auxiliaries_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='local',
        )

        assert settings.use_shared_stage_rhs is False
        assert settings.use_shared_stage_store is False

    def test_stage_store_elements(self):
        """stage_store_elements should be stage_count * n."""
        settings = RosenbrockBufferSettings(n=3, stage_count=4)
        assert settings.stage_store_elements == 12

    def test_shared_memory_elements(self):
        """Shared memory should sum sizes of shared buffers."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            cached_auxiliary_count=10,
            stage_rhs_location='shared',
            stage_store_location='shared',
            cached_auxiliaries_location='shared',
        )
        # rhs (3) + store (12) + aux (10) = 25
        assert settings.shared_memory_elements == 25

    def test_local_memory_elements(self):
        """Local memory should sum sizes of local buffers."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            cached_auxiliary_count=10,
            stage_rhs_location='local',
            stage_store_location='local',
            cached_auxiliaries_location='local',
        )
        # rhs (3) + store (12) + aux (10) = 25
        assert settings.local_memory_elements == 25

    def test_local_sizes_property(self):
        """local_sizes property should return RosenbrockLocalSizes instance."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            cached_auxiliary_count=10,
        )
        sizes = settings.local_sizes

        assert isinstance(sizes, RosenbrockLocalSizes)
        assert sizes.stage_rhs == 3
        assert sizes.stage_store == 12  # 4*3
        assert sizes.cached_auxiliaries == 10

    def test_shared_indices_property(self):
        """shared_indices property should return RosenbrockSliceIndices."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            cached_auxiliary_count=10,
            stage_rhs_location='shared',
            stage_store_location='shared',
            cached_auxiliaries_location='shared',
        )
        indices = settings.shared_indices

        assert isinstance(indices, RosenbrockSliceIndices)
        assert indices.stage_rhs == slice(0, 3)
        assert indices.stage_store == slice(3, 15)
        assert indices.cached_auxiliaries == slice(15, 25)
        assert indices.local_end == 25

    def test_invalid_location_raises(self):
        """Invalid location string should raise ValueError."""
        with pytest.raises(ValueError):
            RosenbrockBufferSettings(
                n=3,
                stage_count=4,
                stage_rhs_location='invalid'
            )
