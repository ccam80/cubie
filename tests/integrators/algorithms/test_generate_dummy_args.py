"""Tests for _generate_dummy_args implementations in algorithm step classes."""

import numpy as np
import pytest

from cubie.integrators.algorithms.explicit_euler import ExplicitEulerStep
from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.backwards_euler_predict_correct import (
    BackwardsEulerPCStep,
)
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_rosenbrock_w import GenericRosenbrockWStep


class TestExplicitEulerGenerateDummyArgs:
    """Tests for ExplicitEulerStep._generate_dummy_args."""

    def test_returns_dict_with_step_key(self):
        """Verify _generate_dummy_args returns dict with 'step' key."""
        step = ExplicitEulerStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'step' in dummy_args

    def test_step_args_is_tuple(self):
        """Verify step args is a tuple."""
        step = ExplicitEulerStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args['step'], tuple)

    def test_step_args_has_correct_length(self):
        """Verify step args tuple has 16 elements matching signature."""
        step = ExplicitEulerStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16

    def test_array_shapes_match_config(self):
        """Verify array shapes match compile_settings dimensions."""
        n = 5
        n_drivers = 2
        precision = np.float32
        step = ExplicitEulerStep(
            precision=precision, n=n, n_drivers=n_drivers
        )
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,)
        assert args[0].dtype == precision
        # proposed_state
        assert args[1].shape == (n,)
        assert args[1].dtype == precision
        # parameters
        assert args[2].shape == (n,)
        assert args[2].dtype == precision
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6)
        assert args[3].dtype == precision
        # drivers_buffer
        assert args[4].shape == (n_drivers,)
        assert args[4].dtype == precision
        # proposed_drivers
        assert args[5].shape == (n_drivers,)
        assert args[5].dtype == precision
        # observables
        assert args[6].shape == (n,)
        assert args[6].dtype == precision
        # proposed_observables
        assert args[7].shape == (n,)
        assert args[7].dtype == precision
        # error
        assert args[8].shape == (n,)
        assert args[8].dtype == precision
        # counters
        assert args[15].shape == (2,)
        assert args[15].dtype == np.int32


class TestImplicitAlgorithmGenerateDummyArgs:
    """Tests for implicit algorithm _generate_dummy_args implementations."""

    @pytest.mark.parametrize("step_class,step_name", [
        (BackwardsEulerStep, "BackwardsEulerStep"),
        (CrankNicolsonStep, "CrankNicolsonStep"),
        (DIRKStep, "DIRKStep"),
    ])
    def test_returns_dict_with_step_key(self, step_class, step_name):
        """Verify _generate_dummy_args returns dict with 'step' key."""
        step = step_class(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict), f"{step_name} should return dict"
        assert 'step' in dummy_args, f"{step_name} should have 'step' key"

    @pytest.mark.parametrize("step_class,step_name", [
        (BackwardsEulerStep, "BackwardsEulerStep"),
        (CrankNicolsonStep, "CrankNicolsonStep"),
        (DIRKStep, "DIRKStep"),
    ])
    def test_step_args_has_correct_length(self, step_class, step_name):
        """Verify step args tuple has 16 elements matching signature."""
        step = step_class(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16, \
            f"{step_name} should return 16 args"

    @pytest.mark.parametrize("step_class,step_name", [
        (BackwardsEulerStep, "BackwardsEulerStep"),
        (CrankNicolsonStep, "CrankNicolsonStep"),
        (DIRKStep, "DIRKStep"),
    ])
    def test_array_shapes_match_config(self, step_class, step_name):
        """Verify array shapes match compile_settings dimensions."""
        n = 4
        n_drivers = 3
        precision = np.float64
        step = step_class(precision=precision, n=n, n_drivers=n_drivers)
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,), f"{step_name} state shape"
        assert args[0].dtype == precision, f"{step_name} state dtype"
        # proposed_state
        assert args[1].shape == (n,), f"{step_name} proposed_state shape"
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6), \
            f"{step_name} driver_coeffs shape"
        # drivers_buffer
        assert args[4].shape == (n_drivers,), \
            f"{step_name} drivers_buffer shape"
        # counters
        assert args[15].shape == (2,), f"{step_name} counters shape"
        assert args[15].dtype == np.int32, f"{step_name} counters dtype"


class TestERKStepGenerateDummyArgs:
    """Tests for ERKStep._generate_dummy_args."""

    def test_returns_dict_with_step_key(self):
        """Verify _generate_dummy_args returns dict with 'step' key."""
        step = ERKStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'step' in dummy_args

    def test_step_args_has_correct_length(self):
        """Verify step args tuple has 16 elements matching signature."""
        step = ERKStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16

    def test_array_shapes_match_config(self):
        """Verify array shapes match compile_settings dimensions."""
        n = 6
        n_drivers = 4
        precision = np.float32
        step = ERKStep(precision=precision, n=n, n_drivers=n_drivers)
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,)
        assert args[0].dtype == precision
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6)
        # counters
        assert args[15].shape == (2,)
        assert args[15].dtype == np.int32


class TestFIRKStepGenerateDummyArgs:
    """Tests for FIRKStep._generate_dummy_args."""

    def test_returns_dict_with_step_key(self):
        """Verify _generate_dummy_args returns dict with 'step' key."""
        step = FIRKStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'step' in dummy_args

    def test_step_args_is_tuple(self):
        """Verify step args is a tuple."""
        step = FIRKStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args['step'], tuple)

    def test_step_args_has_correct_length(self):
        """Verify step args tuple has 16 elements matching signature."""
        step = FIRKStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16

    def test_array_shapes_match_config(self):
        """Verify array shapes match compile_settings dimensions."""
        n = 4
        n_drivers = 2
        precision = np.float64
        step = FIRKStep(precision=precision, n=n, n_drivers=n_drivers)
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,)
        assert args[0].dtype == precision
        # proposed_state
        assert args[1].shape == (n,)
        assert args[1].dtype == precision
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6)
        assert args[3].dtype == precision
        # drivers_buffer
        assert args[4].shape == (n_drivers,)
        assert args[4].dtype == precision
        # counters
        assert args[15].shape == (2,)
        assert args[15].dtype == np.int32


class TestRosenbrockWStepGenerateDummyArgs:
    """Tests for GenericRosenbrockWStep._generate_dummy_args."""

    def test_returns_dict_with_step_key(self):
        """Verify _generate_dummy_args returns dict with 'step' key."""
        step = GenericRosenbrockWStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'step' in dummy_args

    def test_step_args_is_tuple(self):
        """Verify step args is a tuple."""
        step = GenericRosenbrockWStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args['step'], tuple)

    def test_step_args_has_correct_length(self):
        """Verify step args tuple has 16 elements matching signature."""
        step = GenericRosenbrockWStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16

    def test_array_shapes_match_config(self):
        """Verify array shapes match compile_settings dimensions."""
        n = 5
        n_drivers = 3
        precision = np.float32
        step = GenericRosenbrockWStep(
            precision=precision, n=n, n_drivers=n_drivers
        )
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,)
        assert args[0].dtype == precision
        # proposed_state
        assert args[1].shape == (n,)
        assert args[1].dtype == precision
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6)
        assert args[3].dtype == precision
        # drivers_buffer
        assert args[4].shape == (n_drivers,)
        assert args[4].dtype == precision
        # counters
        assert args[15].shape == (2,)
        assert args[15].dtype == np.int32


class TestBackwardsEulerPCStepGenerateDummyArgs:
    """Tests for BackwardsEulerPCStep._generate_dummy_args inheritance."""

    def test_returns_dict_with_step_key(self):
        """Verify inherited _generate_dummy_args returns dict with 'step'."""
        step = BackwardsEulerPCStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert isinstance(dummy_args, dict)
        assert 'step' in dummy_args

    def test_step_args_has_correct_length(self):
        """Verify step args tuple has 16 elements matching signature."""
        step = BackwardsEulerPCStep(precision=np.float64, n=3)
        dummy_args = step._generate_dummy_args()

        assert len(dummy_args['step']) == 16

    def test_array_shapes_match_config(self):
        """Verify array shapes match compile_settings dimensions."""
        n = 4
        n_drivers = 2
        precision = np.float64
        step = BackwardsEulerPCStep(
            precision=precision, n=n, n_drivers=n_drivers
        )
        dummy_args = step._generate_dummy_args()
        args = dummy_args['step']

        # state
        assert args[0].shape == (n,)
        assert args[0].dtype == precision
        # driver_coefficients
        assert args[3].shape == (100, n_drivers, 6)
        # counters
        assert args[15].shape == (2,)
        assert args[15].dtype == np.int32
