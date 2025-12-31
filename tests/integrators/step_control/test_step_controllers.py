"""Tests for step controller _generate_dummy_args implementations."""

import numpy as np
import pytest

from cubie.integrators.step_control.fixed_step_controller import (
    FixedStepController,
)
from cubie.integrators.step_control.adaptive_I_controller import (
    AdaptiveIController,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    AdaptivePIController,
)
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)


class TestFixedControllerGenerateDummyArgs:
    """Tests for FixedStepController._generate_dummy_args."""

    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('n', [1, 3, 10])
    def test_returns_correct_keys(self, precision, n):
        """Verify dict contains expected function names as keys."""
        controller = FixedStepController(
            precision=precision,
            dt=0.001,
            n=n,
        )
        dummy_args = controller._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'device_function' in dummy_args

    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('n', [1, 3, 10])
    def test_shapes_match_config(self, precision, n):
        """Verify array shapes match compile_settings dimensions."""
        controller = FixedStepController(
            precision=precision,
            dt=0.001,
            n=n,
        )
        dummy_args = controller._generate_dummy_args()
        args = dummy_args['device_function']

        # Controller signature: (dt, proposed_state, current_state,
        #                        error, niters, accept_step,
        #                        shared_scratch, persistent_local)
        assert len(args) == 8

        # dt buffer
        assert args[0].shape == (1,)
        assert args[0].dtype == precision

        # proposed_state
        assert args[1].shape == (n,)
        assert args[1].dtype == precision

        # current_state
        assert args[2].shape == (n,)
        assert args[2].dtype == precision

        # error
        assert args[3].shape == (n,)
        assert args[3].dtype == precision

        # niters (scalar)
        assert args[4] == np.int32(1)

        # accept_step
        assert args[5].shape == (1,)
        assert args[5].dtype == np.int32

        # shared_scratch
        assert args[6].shape == (8,)
        assert args[6].dtype == precision

        # persistent_local
        assert args[7].shape == (8,)
        assert args[7].dtype == precision


class TestAdaptiveControllerGenerateDummyArgs:
    """Tests for adaptive controller _generate_dummy_args implementations."""

    @pytest.mark.parametrize('controller_class', [
        AdaptiveIController,
        AdaptivePIController,
        AdaptivePIDController,
    ])
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('n', [1, 3, 10])
    def test_returns_correct_keys(self, controller_class, precision, n):
        """Verify dict contains expected function names as keys."""
        controller = controller_class(
            precision=precision,
            n=n,
        )
        dummy_args = controller._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'device_function' in dummy_args

    @pytest.mark.parametrize('controller_class', [
        AdaptiveIController,
        AdaptivePIController,
        AdaptivePIDController,
    ])
    @pytest.mark.parametrize('precision', [np.float32, np.float64])
    @pytest.mark.parametrize('n', [1, 3, 10])
    def test_shapes_match_config(self, controller_class, precision, n):
        """Verify array shapes match compile_settings dimensions."""
        controller = controller_class(
            precision=precision,
            n=n,
        )
        dummy_args = controller._generate_dummy_args()
        args = dummy_args['device_function']

        # Controller signature: (dt, proposed_state, current_state,
        #                        error, niters, accept_step,
        #                        shared_scratch, persistent_local)
        assert len(args) == 8

        # dt buffer
        assert args[0].shape == (1,)
        assert args[0].dtype == precision

        # proposed_state
        assert args[1].shape == (n,)
        assert args[1].dtype == precision

        # current_state
        assert args[2].shape == (n,)
        assert args[2].dtype == precision

        # error
        assert args[3].shape == (n,)
        assert args[3].dtype == precision

        # niters (scalar)
        assert args[4] == np.int32(1)

        # accept_step
        assert args[5].shape == (1,)
        assert args[5].dtype == np.int32

        # shared_scratch
        assert args[6].shape == (8,)
        assert args[6].dtype == precision

        # persistent_local
        assert args[7].shape == (8,)
        assert args[7].dtype == precision
