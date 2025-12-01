"""Tests for ERKBufferSettings, DIRKBufferSettings, FIRKBufferSettings,
and RosenbrockBufferSettings classes."""
import pytest
from cubie.integrators.algorithms.generic_erk import ERKBufferSettings
from cubie.integrators.algorithms.generic_dirk import DIRKBufferSettings
from cubie.integrators.algorithms.generic_firk import FIRKBufferSettings
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    RosenbrockBufferSettings,
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


class TestDIRKBufferSettings:
    """Tests for DIRKBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should match all_in_one.py."""
        settings = DIRKBufferSettings(n=3, stage_count=4)

        assert settings.stage_increment_location == 'local'
        assert settings.stage_base_location == 'shared'
        assert settings.accumulator_location == 'shared'
        assert settings.solver_scratch_location == 'shared'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = DIRKBufferSettings(
            n=3,
            stage_count=4,
            stage_increment_location='shared',
        )

        assert settings.use_shared_stage_increment is True
        assert settings.use_shared_accumulator is True

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
        """Solver scratch should be 2 * n."""
        settings = DIRKBufferSettings(n=5, stage_count=3)
        assert settings.solver_scratch_elements == 10

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


class TestFIRKBufferSettings:
    """Tests for FIRKBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should match expected values."""
        settings = FIRKBufferSettings(n=3, stage_count=4)

        assert settings.solver_scratch_location == 'shared'
        assert settings.stage_increment_location == 'shared'
        assert settings.stage_driver_stack_location == 'shared'
        assert settings.stage_state_location == 'local'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = FIRKBufferSettings(
            n=3,
            stage_count=4,
            solver_scratch_location='local',
        )

        assert settings.use_shared_solver_scratch is False
        assert settings.use_shared_stage_increment is True

    def test_all_stages_n(self):
        """all_stages_n should be stage_count * n."""
        settings = FIRKBufferSettings(n=3, stage_count=4)
        assert settings.all_stages_n == 12

    def test_solver_scratch_elements(self):
        """Solver scratch should be 2 * all_stages_n."""
        settings = FIRKBufferSettings(n=3, stage_count=4)
        # 2 * (4 * 3) = 24
        assert settings.solver_scratch_elements == 24

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


class TestRosenbrockBufferSettings:
    """Tests for RosenbrockBufferSettings initialization and properties."""

    def test_default_locations(self):
        """Default locations should be shared for all buffers."""
        settings = RosenbrockBufferSettings(n=3, stage_count=4)

        assert settings.stage_rhs_location == 'shared'
        assert settings.stage_store_location == 'shared'
        assert settings.cached_auxiliaries_location == 'shared'

    def test_boolean_flags(self):
        """Boolean properties should reflect location settings."""
        settings = RosenbrockBufferSettings(
            n=3,
            stage_count=4,
            stage_rhs_location='local',
        )

        assert settings.use_shared_stage_rhs is False
        assert settings.use_shared_stage_store is True

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

    def test_invalid_location_raises(self):
        """Invalid location string should raise ValueError."""
        with pytest.raises(ValueError):
            RosenbrockBufferSettings(
                n=3,
                stage_count=4,
                stage_rhs_location='invalid'
            )
