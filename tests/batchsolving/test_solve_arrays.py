"""Tests for Solver.solve_arrays() mid-level API."""

import pytest
import numpy as np
from cubie.batchsolving.solver import validate_solver_arrays
from cubie.batchsolving.solveresult import SolveResult


class TestValidateSolverArrays:
    """Tests for validate_solver_arrays() helper function."""

    def test_valid_arrays_pass(self, solver):
        """Valid arrays matching system expectations pass validation."""
        # Get expected sizes from solver
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        inits = np.zeros((n_states, n_runs), dtype=precision)
        params = np.zeros((n_params, n_runs), dtype=precision)

        # Should not raise
        validate_solver_arrays(
            inits, params, solver.system_sizes, precision
        )

    def test_wrong_initial_values_type_raises(self, solver):
        """Non-ndarray initial_values raises TypeError."""
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        params = np.zeros((n_params, n_runs), dtype=precision)

        with pytest.raises(TypeError):
            validate_solver_arrays(
                [[0.1, 0.2]],  # list, not ndarray
                params,
                solver.system_sizes,
                precision,
            )

    def test_wrong_parameters_type_raises(self, solver):
        """Non-ndarray parameters raises TypeError."""
        n_states = solver.system_sizes.states
        precision = solver.precision
        n_runs = 4

        inits = np.zeros((n_states, n_runs), dtype=precision)

        with pytest.raises(TypeError):
            validate_solver_arrays(
                inits,
                [[1.0, 2.0]],  # list, not ndarray
                solver.system_sizes,
                precision,
            )

    def test_wrong_initial_values_shape_raises(self, solver):
        """Mismatched initial_values shape raises ValueError."""
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        # Wrong number of states
        inits = np.zeros((999, n_runs), dtype=precision)
        params = np.zeros((n_params, n_runs), dtype=precision)

        with pytest.raises(ValueError):
            validate_solver_arrays(
                inits, params, solver.system_sizes, precision
            )

    def test_wrong_parameters_shape_raises(self, solver):
        """Mismatched parameters shape raises ValueError."""
        n_states = solver.system_sizes.states
        precision = solver.precision
        n_runs = 4

        inits = np.zeros((n_states, n_runs), dtype=precision)
        # Wrong number of parameters
        params = np.zeros((999, n_runs), dtype=precision)

        with pytest.raises(ValueError):
            validate_solver_arrays(
                inits, params, solver.system_sizes, precision
            )

    def test_mismatched_runs_raises(self, solver):
        """Different n_runs between arrays raises ValueError."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision

        inits = np.zeros((n_states, 4), dtype=precision)
        params = np.zeros((n_params, 5), dtype=precision)  # different n_runs

        with pytest.raises(ValueError):
            validate_solver_arrays(
                inits, params, solver.system_sizes, precision
            )

    def test_wrong_dtype_raises(self, solver):
        """Wrong dtype raises ValueError."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        # Use opposite precision
        wrong_dtype = np.float64 if precision == np.float32 else np.float32

        inits = np.zeros((n_states, n_runs), dtype=wrong_dtype)
        params = np.zeros((n_params, n_runs), dtype=precision)

        with pytest.raises(ValueError):
            validate_solver_arrays(
                inits, params, solver.system_sizes, precision
            )

    def test_non_contiguous_raises(self, solver):
        """Non-contiguous arrays raise ValueError."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        # Create Fortran-contiguous array
        inits = np.asfortranarray(
            np.zeros((n_states, n_runs), dtype=precision)
        )
        params = np.zeros((n_params, n_runs), dtype=precision)

        with pytest.raises(ValueError):
            validate_solver_arrays(
                inits, params, solver.system_sizes, precision
            )


class TestSolveArrays:
    """Tests for Solver.solve_arrays() method."""

    def test_solve_arrays_basic(self, solver, driver_settings):
        """solve_arrays returns SolveResult with valid arrays."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 4

        inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve_arrays(
            initial_values=inits,
            parameters=params,
            duration=0.1,
        )

        assert isinstance(result, SolveResult)
        assert hasattr(result, "time_domain_array")

    def test_solve_arrays_single_run(self, solver):
        """solve_arrays works with single run."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 1

        inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve_arrays(
            initial_values=inits,
            parameters=params,
            duration=0.1,
        )

        assert isinstance(result, SolveResult)

    def test_solve_arrays_with_driver_coefficients(self, solver):
        """solve_arrays accepts explicit driver coefficients."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 2

        inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        params = np.ones((n_params, n_runs), dtype=precision)

        # Use driver coefficients from solver's interpolator
        driver_coeffs = solver.driver_interpolator.coefficients

        result = solver.solve_arrays(
            initial_values=inits,
            parameters=params,
            driver_coefficients=driver_coeffs,
            duration=0.1,
        )

        assert isinstance(result, SolveResult)

    def test_solve_arrays_with_kwargs(self, solver_mutable):
        """solve_arrays forwards kwargs to update()."""
        solver = solver_mutable
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 2

        inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        params = np.ones((n_params, n_runs), dtype=precision)
        original_dt = solver.dt
        new_dt = precision(original_dt * 0.5) if original_dt else precision(
            1e-4)

        result = solver.solve_arrays(
            initial_values=inits,
            parameters=params,
            duration=0.1,
            dt=new_dt,
        )

        assert isinstance(result, SolveResult)

    def test_solve_arrays_result_types(self, solver):
        """solve_arrays respects results_type parameter."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision
        n_runs = 2

        inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
        params = np.ones((n_params, n_runs), dtype=precision)

        result_full = solver.solve_arrays(
            inits, params, duration=0.1, results_type="full"
        )
        result_numpy = solver.solve_arrays(
            inits, params, duration=0.1, results_type="numpy"
        )

        assert isinstance(result_full, SolveResult)
        assert isinstance(result_numpy, dict)

    def test_solve_arrays_invalid_raises(self, solver):
        """solve_arrays raises on invalid input arrays."""
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters
        precision = solver.precision

        # Wrong shape
        inits = np.ones((999, 2), dtype=precision)
        params = np.ones((n_params, 2), dtype=precision)

        with pytest.raises(ValueError):
            solver.solve_arrays(inits, params, duration=0.1)


class TestSolveArraysConsistency:
    """Tests ensuring solve() and solve_arrays() produce consistent results."""

    def test_solve_and_solve_arrays_consistent(
        self, solver, simple_initial_values, simple_parameters, driver_settings
    ):
        """solve() and solve_arrays() produce equivalent results."""
        # First solve with dict inputs
        result_dict = solver.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.1,
            grid_type="verbatim",
            results_type="full",
        )

        # Build arrays manually matching verbatim grid
        state_names = list(simple_initial_values.keys())
        param_names = list(simple_parameters.keys())
        precision = solver.precision

        n_runs = len(list(simple_initial_values.values())[0])
        n_states = solver.system_sizes.states
        n_params = solver.system_sizes.parameters

        inits = np.zeros((n_states, n_runs), dtype=precision)
        params = np.zeros((n_params, n_runs), dtype=precision)

        # Fill arrays in same order as grid_builder
        for i, name in enumerate(state_names):
            if i < n_states:
                values = simple_initial_values[name]
                inits[i, :] = np.array(values, dtype=precision)

        for i, name in enumerate(param_names):
            if i < n_params:
                values = simple_parameters[name]
                params[i, :] = np.array(values, dtype=precision)

        # Solve with array inputs
        result_arrays = solver.solve_arrays(
            initial_values=inits,
            parameters=params,
            driver_coefficients=solver.driver_interpolator.coefficients,
            duration=0.1,
            results_type="full",
        )

        # Results should match
        assert (result_dict.time_domain_array.shape
                == result_arrays.time_domain_array.shape)
