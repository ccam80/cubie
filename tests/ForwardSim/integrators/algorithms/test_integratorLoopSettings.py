import pytest
import numpy as np
from numba import float32, float64
from warnings import warn

from CuMC.ForwardSim.integrators.algorithms.IntegratorLoopSettings import (
    IntegratorLoopSettings
)
from CuMC.ForwardSim.integrators.algorithms.LoopStepConfig import LoopStepConfig
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes


class TestLoopStepConfig:
    """Test class for LoopStepConfig with various parameter combinations."""

    def test_default_initialization(self):
        """Test that LoopStepConfig initializes with sensible defaults."""
        config = LoopStepConfig()
        assert config.dt_min == 1e-6
        assert config.dt_max == 1.0
        assert config.dt_save == 0.1
        assert config.dt_summarise == 0.1
        assert config.atol == 1e-6
        assert config.rtol == 1e-6

    @pytest.mark.parametrize("dt_min, dt_max, dt_save, dt_summarise, atol, rtol", [
        (1e-5, 0.5, 0.01, 0.1, 1e-5, 1e-4),
        (1e-4, 1.0, 0.1, 0.2, 1e-7, 1e-6),
        (1e-3, 2.0, 0.05, 0.5, 1e-8, 1e-7),
    ], ids=['standard_params', 'coarse_params', 'fine_params'])
    def test_custom_initialization(self, dt_min, dt_max, dt_save, dt_summarise, atol, rtol):
        """Test LoopStepConfig with various custom parameter sets."""
        config = LoopStepConfig(
            dt_min=dt_min,
            dt_max=dt_max,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
            atol=atol,
            rtol=rtol
        )
        assert config.dt_min == dt_min
        assert config.dt_max == dt_max
        assert config.dt_save == dt_save
        assert config.dt_summarise == dt_summarise
        assert config.atol == atol
        assert config.rtol == rtol

    @pytest.mark.parametrize("invalid_param, invalid_value", [
        ('dt_min', 'invalid'),
        ('dt_max', None),
        ('dt_save', [1.0]),
        ('atol', 'string'),
    ], ids=['dt_min_string', 'dt_max_none', 'dt_save_list', 'atol_string'])
    def test_type_validation(self, invalid_param, invalid_value):
        """Test that LoopStepConfig validates parameter types."""
        kwargs = {invalid_param: invalid_value}
        with pytest.raises(TypeError):
            LoopStepConfig(**kwargs)


class TestIntegratorLoopSettings:
    """Test class for IntegratorLoopSettings with various configurations."""

    @pytest.fixture
    def mock_buffer_sizes(self):
        """Create a mock LoopBufferSizes for testing."""
        return LoopBufferSizes(
            state_summaries=5,
            observable_summaries=3,
            state=10,
            observables=8,
            dxdt=10,
            parameters=6,
            drivers=4
        )

    @pytest.fixture
    def mock_step_config(self):
        """Create a mock LoopStepConfig for testing."""
        return LoopStepConfig(
            dt_min=1e-4,
            dt_max=0.1,
            dt_save=0.01,
            dt_summarise=0.1,
            atol=1e-6,
            rtol=1e-5
        )

    @pytest.fixture
    def mock_functions(self):
        """Create mock functions for testing."""
        def mock_dxdt(state, params, drivers, obs, dxdt):
            pass
        def mock_save_state(state, obs, state_out, obs_out, i):
            pass
        def mock_update_summary(state, obs, state_sum, obs_sum, i):
            pass
        def mock_save_summary(state_sum, obs_sum, state_out, obs_out, n_steps):
            pass

        return {
            'dxdt_func': mock_dxdt,
            'save_state_func': mock_save_state,
            'update_summary_func': mock_update_summary,
            'save_summary_func': mock_save_summary
        }

    def test_default_initialization(self, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test IntegratorLoopSettings initializes with defaults."""
        settings = IntegratorLoopSettings(
            buffer_sizes=mock_buffer_sizes,
            loop_step_config=mock_step_config,
            **mock_functions
        )
        assert settings.precision == float32
        assert settings.dt_min == 1e-4
        assert settings.dt_max == 0.1
        assert settings.dt_save == 0.01
        assert settings.buffer_sizes == mock_buffer_sizes

    @pytest.mark.parametrize("precision", [float32, float64], ids=['float32', 'float64'])
    def test_precision_validation(self, precision, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test that precision validation works correctly."""
        settings = IntegratorLoopSettings(
            precision=precision,
            loop_step_config=mock_step_config,
            buffer_sizes=mock_buffer_sizes,
            **mock_functions
        )
        assert settings.precision == precision

    def test_invalid_precision_raises_error(self, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError):
            IntegratorLoopSettings(
                precision=np.float16,  # Invalid precision
                loop_step_config=mock_step_config,
                buffer_sizes=mock_buffer_sizes,
                **mock_functions
            )

    def test_step_config_properties(self, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test that step config properties are accessible."""
        settings = IntegratorLoopSettings(
            loop_step_config=mock_step_config,
            buffer_sizes=mock_buffer_sizes,
            **mock_functions
        )
        assert settings.dt_min == mock_step_config.dt_min
        assert settings.dt_max == mock_step_config.dt_max
        assert settings.dt_save == mock_step_config.dt_save
        assert settings.dt_summarise == mock_step_config.dt_summarise
        assert settings.atol == mock_step_config.atol
        assert settings.rtol == mock_step_config.rtol

    def test_buffer_sizes_property(self, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test that buffer_sizes property returns correct values."""
        settings = IntegratorLoopSettings(
            loop_step_config=mock_step_config,
            buffer_sizes=mock_buffer_sizes,
            **mock_functions
        )
        assert settings.buffer_sizes == mock_buffer_sizes
        assert settings.buffer_sizes.state == 10
        assert settings.buffer_sizes.observables == 8
        assert settings.buffer_sizes.parameters == 6

    def test_function_assignment(self, mock_buffer_sizes, mock_step_config, mock_functions):
        """Test that function references are properly assigned."""
        settings = IntegratorLoopSettings(
            loop_step_config=mock_step_config,
            buffer_sizes=mock_buffer_sizes,
            **mock_functions
        )
        assert settings.dxdt_func == mock_functions['dxdt_func']
        assert settings.save_state_func == mock_functions['save_state_func']
        assert settings.update_summary_func == mock_functions['update_summary_func']
        assert settings.save_summary_func == mock_functions['save_summary_func']

    def test_none_functions_allowed(self, mock_buffer_sizes, mock_step_config):
        """Test that None functions are allowed during initialization."""
        settings = IntegratorLoopSettings(
            loop_step_config=mock_step_config,
            buffer_sizes=mock_buffer_sizes
        )
        assert settings.dxdt_func is None
        assert settings.save_state_func is None
        assert settings.update_summary_func is None
        assert settings.save_summary_func is None
