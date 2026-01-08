"""Test that ODELoopConfig contains only build-used fields.

This test validates the compile_settings cleanup by verifying that
ODELoopConfig only contains fields actually used in build() chains or
buffer registration.
"""
import pytest
from attrs import fields

from cubie.integrators.loops.ode_loop_config import ODELoopConfig


class TestODELoopConfigMinimal:
    """Test ODELoopConfig contains only necessary fields."""

    def test_no_controller_local_len_field(self):
        """Verify controller_local_len was removed from ODELoopConfig.
        
        Controller manages its own buffer allocation through the buffer
        registry, so the loop config doesn't need to track this metadata.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        assert 'controller_local_len' not in config_fields, (
            "controller_local_len should be removed from ODELoopConfig; "
            "controller manages its own buffers"
        )

    def test_no_algorithm_local_len_field(self):
        """Verify algorithm_local_len was removed from ODELoopConfig.
        
        Algorithm manages its own buffer allocation through the buffer
        registry, so the loop config doesn't need to track this metadata.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        assert 'algorithm_local_len' not in config_fields, (
            "algorithm_local_len should be removed from ODELoopConfig; "
            "algorithm manages its own buffers"
        )

    def test_no_dt_min_field(self):
        """Verify _dt_min was removed from ODELoopConfig.
        
        dt_min belongs to the step controller config (AdaptiveStepControlConfig),
        not the loop config. The loop receives a compiled step_controller_fn
        with dt_min already baked in.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        assert '_dt_min' not in config_fields, (
            "_dt_min should be removed from ODELoopConfig; "
            "it belongs to step controller config"
        )

    def test_no_dt_max_field(self):
        """Verify _dt_max was removed from ODELoopConfig.
        
        dt_max belongs to the step controller config (AdaptiveStepControlConfig),
        not the loop config. The loop receives a compiled step_controller_fn
        with dt_max already baked in.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        assert '_dt_max' not in config_fields, (
            "_dt_max should be removed from ODELoopConfig; "
            "it belongs to step controller config"
        )

    def test_essential_size_fields_present(self):
        """Verify essential size fields are retained in ODELoopConfig.
        
        These fields are used directly in build() for loop iteration bounds
        and in register_buffers() for buffer allocation.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        
        essential_fields = {
            'n_states',
            'n_parameters',
            'n_drivers',
            'n_observables',
            'n_error',
            'n_counters',
            'state_summaries_buffer_height',
            'observable_summaries_buffer_height',
        }
        
        for field_name in essential_fields:
            assert field_name in config_fields, (
                f"{field_name} should be retained in ODELoopConfig"
            )

    def test_location_parameters_present(self):
        """Verify all buffer location parameters are retained.
        
        Location parameters are used in buffer_registry.register() calls
        during register_buffers().
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        
        location_fields = {
            'state_location',
            'proposed_state_location',
            'parameters_location',
            'drivers_location',
            'proposed_drivers_location',
            'observables_location',
            'proposed_observables_location',
            'error_location',
            'counters_location',
            'state_summary_location',
            'observable_summary_location',
            'dt_location',
            'accept_step_location',
            'proposed_counters_location',
        }
        
        for field_name in location_fields:
            assert field_name in config_fields, (
                f"{field_name} should be retained in ODELoopConfig"
            )

    def test_device_function_callbacks_present(self):
        """Verify device function callbacks are retained.
        
        These callbacks are captured in closures during build() and are
        essential for loop compilation.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        
        callback_fields = {
            'save_state_fn',
            'update_summaries_fn',
            'save_summaries_fn',
            'step_controller_fn',
            'step_function',
            'evaluate_driver_at_t',
            'evaluate_observables',
        }
        
        for field_name in callback_fields:
            assert field_name in config_fields, (
                f"{field_name} should be retained in ODELoopConfig"
            )

    def test_timing_parameters_present(self):
        """Verify timing parameters are retained.
        
        These parameters are captured in closures during build() and control
        loop behavior.
        """
        config_fields = {f.name for f in fields(ODELoopConfig)}
        
        timing_fields = {
            '_save_every',
            '_summarise_every',
            '_sample_summaries_every',
            '_dt0',
            'save_last',
            'save_regularly',
            'summarise_regularly',
            'is_adaptive',
        }
        
        for field_name in timing_fields:
            assert field_name in config_fields, (
                f"{field_name} should be retained in ODELoopConfig"
            )

    def test_config_instantiation_without_deleted_fields(self):
        """Verify ODELoopConfig can be instantiated without deleted fields.
        
        This test ensures that code not passing controller_local_len or
        algorithm_local_len continues to work correctly.
        """
        config = ODELoopConfig(
            n_states=3,
            n_parameters=2,
            n_drivers=1,
            n_observables=2,
            n_error=3,
            n_counters=4,
        )
        
        assert config.n_states == 3
        assert config.n_parameters == 2
        assert config.n_drivers == 1
        assert config.n_observables == 2
        assert config.n_error == 3
        assert config.n_counters == 4
