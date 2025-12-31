"""Tests for BaseODE and SymbolicODE _generate_dummy_args implementation."""

import numpy as np
import pytest

from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


@pytest.fixture(scope="function")
def simple_symbolic_ode(precision):
    """Create a simple SymbolicODE for testing."""
    return SymbolicODE.create(
        dxdt=["dx = -k * x", "dy = k * x - m * y"],
        states={"x": 1.0, "y": 0.0},
        parameters={"k": 0.1, "m": 0.05},
        observables=["z"],
        drivers=["d1"],
        precision=precision,
        name="test_simple_ode",
        strict=False,
    )


def test_symbolic_ode_generate_dummy_args(simple_symbolic_ode, precision):
    """Verify SymbolicODE returns properly shaped args for dxdt."""
    dummy_args = simple_symbolic_ode._generate_dummy_args()

    # Should have dxdt and observables keys
    assert 'dxdt' in dummy_args
    assert 'observables' in dummy_args

    # Check dxdt args structure
    dxdt_args = dummy_args['dxdt']
    assert len(dxdt_args) == 5

    n_states = simple_symbolic_ode.num_states
    n_params = simple_symbolic_ode.num_parameters
    n_drivers = simple_symbolic_ode.num_drivers

    # dxdt signature: (state, dxdt_out, parameters, drivers, t)
    state_arr, dxdt_out_arr, params_arr, drivers_arr, t_val = dxdt_args

    assert state_arr.shape == (n_states,)
    assert state_arr.dtype == precision
    assert dxdt_out_arr.shape == (n_states,)
    assert dxdt_out_arr.dtype == precision
    assert params_arr.shape == (n_params,)
    assert params_arr.dtype == precision
    assert drivers_arr.shape == (n_drivers,)
    assert drivers_arr.dtype == precision
    assert isinstance(t_val, (np.floating, float))


def test_symbolic_ode_generate_dummy_args_observables(
    simple_symbolic_ode, precision
):
    """Verify SymbolicODE returns properly shaped args for observables."""
    dummy_args = simple_symbolic_ode._generate_dummy_args()

    obs_args = dummy_args['observables']
    assert len(obs_args) == 5

    n_states = simple_symbolic_ode.num_states
    n_params = simple_symbolic_ode.num_parameters
    n_drivers = simple_symbolic_ode.num_drivers
    n_obs = simple_symbolic_ode.num_observables

    # observables signature: (state, params, drivers, obs_out, t)
    state_arr, params_arr, drivers_arr, obs_out_arr, t_val = obs_args

    assert state_arr.shape == (n_states,)
    assert state_arr.dtype == precision
    assert params_arr.shape == (n_params,)
    assert params_arr.dtype == precision
    assert drivers_arr.shape == (n_drivers,)
    assert drivers_arr.dtype == precision
    assert obs_out_arr.shape == (n_obs,)
    assert obs_out_arr.dtype == precision
    assert isinstance(t_val, (np.floating, float))


def test_symbolic_ode_generate_dummy_args_no_drivers(precision):
    """Verify SymbolicODE handles zero drivers correctly."""
    ode = SymbolicODE.create(
        dxdt=["dx = -k * x"],
        states={"x": 1.0},
        parameters={"k": 0.1},
        precision=precision,
        name="test_no_drivers_ode",
        strict=False,
    )

    dummy_args = ode._generate_dummy_args()

    dxdt_args = dummy_args['dxdt']
    drivers_arr = dxdt_args[3]

    # Zero drivers should result in shape (0,)
    assert drivers_arr.shape == (0,)
    assert drivers_arr.dtype == precision
