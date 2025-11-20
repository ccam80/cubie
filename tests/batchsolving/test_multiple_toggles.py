"""Tests for multiple_inits and multiple_params compile-time toggles."""

import numpy as np
import pytest

from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder


@pytest.fixture(scope="function")
def grid_builder(system):
    """Create a grid builder from a test system."""
    return BatchGridBuilder.from_system(system)


def test_single_init_multiple_params(grid_builder, system):
    """Test that single init is detected when only params are swept."""
    # Single initial state
    states = np.array([[1.0, 2.0]])
    # Multiple parameter sets
    params = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="combinatorial"
    )

    # Should have 3 rows (one for each param set)
    assert inits.shape[0] == 3
    assert params_out.shape[0] == 3

    # Check metadata flags
    assert metadata["multiple_inits"] is False
    assert metadata["multiple_params"] is True

    # All init rows should be identical
    assert np.all(inits == inits[0, :])


def test_multiple_inits_single_param(grid_builder, system):
    """Test that single param is detected when only inits are swept."""
    # Multiple initial states
    states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Single parameter set
    params = np.array([[0.1, 0.2]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="combinatorial"
    )

    # Should have 3 rows (one for each init set)
    assert inits.shape[0] == 3
    assert params_out.shape[0] == 3

    # Check metadata flags
    assert metadata["multiple_inits"] is True
    assert metadata["multiple_params"] is False

    # All param rows should be identical
    assert np.all(params_out == params_out[0, :])


def test_both_multiple(grid_builder, system):
    """Test that both flags are True when both are swept."""
    # Multiple initial states
    states = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Multiple parameter sets
    params = np.array([[0.1, 0.2], [0.3, 0.4]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="combinatorial"
    )

    # Should have 4 rows (2x2 combinations)
    assert inits.shape[0] == 4
    assert params_out.shape[0] == 4

    # Check metadata flags
    assert metadata["multiple_inits"] is True
    assert metadata["multiple_params"] is True


def test_both_single(grid_builder, system):
    """Test that both flags are False when neither is swept."""
    # Single initial state
    states = np.array([[1.0, 2.0]])
    # Single parameter set
    params = np.array([[0.1, 0.2]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="verbatim"
    )

    # Should have 1 row
    assert inits.shape[0] == 1
    assert params_out.shape[0] == 1

    # Check metadata flags
    assert metadata["multiple_inits"] is False
    assert metadata["multiple_params"] is False


def test_dict_sweep_single_param(grid_builder, system):
    """Test single param detection with dictionary input."""
    state_names = list(system.initial_values.names)

    # Sweep first state, keep params at default
    request = {state_names[0]: [1.0, 2.0, 3.0]}

    inits, params, metadata = grid_builder(
        request=request, kind="combinatorial"
    )

    # Should have 3 rows
    assert inits.shape[0] == 3
    assert params.shape[0] == 3

    # multiple_inits should be True (sweeping), multiple_params should be False
    assert metadata["multiple_inits"] is True
    assert metadata["multiple_params"] is False

    # All param rows should be identical (default values)
    assert np.all(params == params[0, :])


def test_dict_sweep_single_init(grid_builder, system):
    """Test single init detection with dictionary input."""
    param_names = list(system.parameters.names)

    # Sweep first param, keep inits at default
    request = {param_names[0]: [0.1, 0.2, 0.3]}

    inits, params, metadata = grid_builder(
        request=request, kind="combinatorial"
    )

    # Should have 3 rows
    assert inits.shape[0] == 3
    assert params.shape[0] == 3

    # multiple_inits should be False, multiple_params should be True
    assert metadata["multiple_inits"] is False
    assert metadata["multiple_params"] is True

    # All init rows should be identical (default values)
    assert np.all(inits == inits[0, :])


def test_has_single_unique_row_helper():
    """Test the _has_single_unique_row helper function."""
    # Single row
    arr1 = np.array([[1.0, 2.0]])
    assert BatchGridBuilder._has_single_unique_row(arr1) is True

    # Multiple identical rows
    arr2 = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert BatchGridBuilder._has_single_unique_row(arr2) is True

    # Multiple different rows
    arr3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert BatchGridBuilder._has_single_unique_row(arr3) is False

    # Empty array
    arr4 = np.array([[]]).reshape(0, 2)
    assert BatchGridBuilder._has_single_unique_row(arr4) is True


def test_verbatim_with_single_row_expansion(grid_builder, system):
    """Test verbatim mode with single row expansion."""
    # Single initial state
    states = np.array([[1.0, 2.0]])
    # Multiple parameter sets
    params = np.array([[0.1, 0.2], [0.3, 0.4]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="verbatim"
    )

    # Should have 2 rows (params expanded states to match)
    assert inits.shape[0] == 2
    assert params_out.shape[0] == 2

    # Check metadata - after expansion, inits are duplicated
    assert metadata["multiple_inits"] is False
    assert metadata["multiple_params"] is True


def test_combinatorial_identical_rows(grid_builder, system):
    """Test that identical rows are detected in combinatorial mode."""
    # Duplicate initial states (should still be detected as single)
    states = np.array([[1.0, 2.0], [1.0, 2.0]])
    # Single parameter set
    params = np.array([[0.1, 0.2]])

    inits, params_out, metadata = grid_builder(
        states=states, params=params, kind="combinatorial"
    )

    # Should have 2 rows (combinatorial of 2 states × 1 param)
    assert inits.shape[0] == 2
    assert params_out.shape[0] == 2

    # Even though we provided 2 state rows, they're identical
    assert metadata["multiple_inits"] is False
    assert metadata["multiple_params"] is False
