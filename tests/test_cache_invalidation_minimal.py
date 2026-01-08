"""Test cache invalidation behavior after compile_settings cleanup.

This test validates that deleted parameters (controller_local_len,
algorithm_local_len) no longer affect the caching system, while build-used
parameters continue to properly invalidate the cache when changed.
"""
import pytest
from numpy import float32, float64
from attrs import evolve

from cubie.integrators.loops.ode_loop_config import ODELoopConfig
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling.output_config import OutputCompileFlags


class TestCacheInvalidationMinimal:
    """Test cache invalidation after removing redundant compile_settings."""

    def test_build_used_parameter_invalidates_cache(self):
        """Verify that changing a build-used parameter invalidates the cache.
        
        Precision is used throughout build() to select types for device arrays
        and function signatures. Changing it must invalidate the cached
        loop_function.
        """
        config1 = ODELoopConfig(
            n_states=3,
            precision=float32,
        )
        config2 = ODELoopConfig(
            n_states=3,
            precision=float64,
        )
        
        assert config1 != config2, (
            "Configs with different precision should not be equal"
        )

    def test_deleted_fields_not_in_config_equality(self):
        """Verify deleted fields are truly removed from config.
        
        Since controller_local_len and algorithm_local_len were removed from
        ODELoopConfig, attempting to set them via evolve should fail with
        TypeError (attrs won't recognize unknown fields).
        """
        config = ODELoopConfig(n_states=3)
        
        with pytest.raises(TypeError, match="controller_local_len|got an unexpected keyword"):
            evolve(config, controller_local_len=10)
        
        with pytest.raises(TypeError, match="algorithm_local_len|got an unexpected keyword"):
            evolve(config, algorithm_local_len=20)

    def test_essential_parameters_affect_equality(self):
        """Verify that essential loop parameters affect config equality.
        
        Parameters like n_states and buffer locations are used in build()
        and should affect config equality (and thus cache invalidation).
        """
        base_config = ODELoopConfig(
            n_states=3,
            precision=float32,
        )
        
        different_n_states = ODELoopConfig(
            n_states=5,
            precision=float32,
        )
        assert base_config != different_n_states, (
            "Configs with different n_states should not be equal"
        )
        
        different_location = ODELoopConfig(
            n_states=3,
            precision=float32,
            state_location='shared',
        )
        assert base_config != different_location, (
            "Configs with different buffer locations should not be equal"
        )

    def test_config_equality_basics(self):
        """Verify that identical configs are equal.
        
        This is the baseline expectation for the caching system - identical
        compile_settings should produce cache hits.
        """
        config1 = ODELoopConfig(
            n_states=3,
            n_parameters=2,
            precision=float32,
            state_location='local',
        )
        config2 = ODELoopConfig(
            n_states=3,
            n_parameters=2,
            precision=float32,
            state_location='local',
        )
        
        assert config1 == config2, (
            "Identical configs should be equal (cache hit)"
        )

    def test_minimal_config_fields_suffice(self):
        """Verify that minimal field set is sufficient for config creation.
        
        After cleanup, users should be able to create configs with only the
        essential fields. This test validates the cleanup didn't break basic
        instantiation patterns.
        """
        minimal_config = ODELoopConfig(n_states=3)
        
        assert minimal_config.n_states == 3
        assert minimal_config.n_parameters == 0
        assert minimal_config.precision == float32
        assert minimal_config.state_location == 'local'
        
        assert not hasattr(minimal_config, 'controller_local_len')
        assert not hasattr(minimal_config, 'algorithm_local_len')
