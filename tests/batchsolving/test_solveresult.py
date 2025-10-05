import numpy as np
import pandas as pd
import pytest

from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving.solveresult import SolveResult
from tests._utils import extract_state_and_time


# --------------------------------------------------------------------------- #
#Create a one-off set of module-scope fixtures for testing SolveResult
# instances

@pytest.fixture(scope="function")
def solver_with_arrays(
    solver,
    batch_input_arrays,
    solver_settings,
    system,
    precision,
    driver_array,
):
    """Solver with actual arrays computed - ready for SolveResult instantiation"""
    inits, params = batch_input_arrays


    solver.kernel.run(
        duration=solver_settings["duration"],
        params=params,
        inits=inits,
        driver_coefficients=driver_array.coefficients,
        blocksize=solver_settings["blocksize"],
        stream=solver_settings["stream"],
        warmup=solver_settings["warmup"],
    )
    return solver


# --------------------------------------------------------------------------- #


class TestSolveResultStaticMethods:
    """Test static methods with minimal unit tests using known arrays."""

    @pytest.mark.parametrize(
        "state_flag, obs_flag, expected",
        [
            (True, True, lambda s, o: np.concatenate((s, o), axis=-1)),
            (True, False, lambda s, o: s),
            (False, True, lambda s, o: o),
            (False, False, lambda s, o: np.array([])),
        ],
    )
    def test_combine_time_domain_arrays(self, state_flag, obs_flag, expected):
        """Test time domain array combination with different active flags."""
        s = np.array([[[1, 2]], [[3, 4]]])
        o = np.array([[[5, 6]], [[7, 8]]])
        result = SolveResult.combine_time_domain_arrays(
            s, o, state_flag, obs_flag
        )
        expected_result = expected(s, o)
        assert np.array_equal(result, expected_result)

    @pytest.mark.parametrize(
        "state_flag, obs_flag, expected",
        [
            (True, True, lambda s, o: np.concatenate((s, o), axis=-1)),
            (True, False, lambda s, o: s),
            (False, True, lambda s, o: o),
            (False, False, lambda s, o: np.array([])),
        ],
    )
    def test_combine_summaries_array(self, state_flag, obs_flag, expected):
        """Test summaries array combination with different active flags."""
        s = np.array([[[10, 20]], [[30, 40]]])
        o = np.array([[[50, 60]], [[70, 80]]])
        result = SolveResult.combine_summaries_array(
            s, o, state_flag, obs_flag
        )
        expected_result = expected(s, o)
        assert np.array_equal(result, expected_result)

    def test_cleave_time_with_time_saved(self):
        """Test time cleaving when time is saved."""
        # Create test array with time as last variable
        state = np.array([[[1, 2, 0.1]], [[3, 4, 0.2]]])  # time, run, variable
        time_result, state_result = SolveResult.cleave_time(
            state, time_saved=True
        )

        expected_time = np.array([[0.1], [0.2]])
        expected_state = np.array([[[1, 2]], [[3, 4]]])

        assert np.array_equal(time_result, expected_time)
        assert np.array_equal(state_result, expected_state)

    def test_cleave_time_without_time_saved(self):
        """Test time cleaving when time is not saved."""
        state = np.array([[[1, 2]], [[3, 4]]])
        time_result, state_result = SolveResult.cleave_time(
            state, time_saved=False
        )

        assert time_result is None
        assert np.array_equal(state_result, state)


class TestSolveResultInstantiation:
    """Test the four instantiation types return equivalent results."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_instantiation_type_equivalence(self, solver_with_arrays):
        """Test that the four instantiation types return equivalent results to full + methods."""

        # Create full SolveResult instance
        full_result = SolveResult.from_solver(
            solver_with_arrays, results_type="full"
        )

        # Create other types directly
        numpy_result = SolveResult.from_solver(
            solver_with_arrays, results_type="numpy"
        )
        numpy_per_summary_result = SolveResult.from_solver(
            solver_with_arrays, results_type="numpy_per_summary"
        )
        pandas_result = SolveResult.from_solver(
            solver_with_arrays, results_type="pandas"
        )

        # Test equivalence by calling methods on full instance
        assert set(numpy_result.keys()) == set(full_result.as_numpy.keys())
        for key in numpy_result.keys():
            if isinstance(numpy_result[key], np.ndarray):
                assert np.array_equal(
                    numpy_result[key], full_result.as_numpy[key]
                )
            else:
                assert numpy_result[key] == full_result.as_numpy[key]

        # Compare numpy per summary result dictionaries
        assert set(numpy_per_summary_result.keys()) == set(
            full_result.as_numpy_per_summary.keys()
        )
        for key in numpy_per_summary_result.keys():
            if isinstance(numpy_per_summary_result[key], np.ndarray):
                assert np.array_equal(
                    numpy_per_summary_result[key],
                    full_result.as_numpy_per_summary[key],
                )
            else:
                assert (
                    numpy_per_summary_result[key]
                    == full_result.as_numpy_per_summary[key]
                )
        assert pandas_result["time_domain"].equals(
            full_result.as_pandas["time_domain"]
        )
        assert pandas_result["summaries"].equals(
            full_result.as_pandas["summaries"]
        )

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_from_solver_full_instantiation(self, solver_with_arrays):
        """Test full SolveResult instantiation from solver."""
        result = SolveResult.from_solver(
            solver_with_arrays, results_type="full"
        )

        assert isinstance(result, SolveResult)
        assert result.time_domain_array.size > 0
        assert result.summaries_array.size > 0
        assert len(result.time_domain_legend) > 0
        assert len(result.summaries_legend) > 0
        assert (
            result._stride_order
            == solver_with_arrays.kernel.output_arrays.host._stride_order
        )

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_from_solver_numpy_instantiation(self, solver_with_arrays):
        """Test numpy dict instantiation from solver."""
        result = SolveResult.from_solver(
            solver_with_arrays, results_type="numpy"
        )

        assert isinstance(result, dict)
        assert "time_domain_array" in result
        assert "summaries_array" in result
        assert "time_domain_legend" in result
        assert "summaries_legend" in result
        assert isinstance(result["time_domain_array"], np.ndarray)

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_from_solver_numpy_per_summary_instantiation(
        self, solver_with_arrays
    ):
        """Test numpy per summary dict instantiation from solver."""
        result = SolveResult.from_solver(
            solver_with_arrays, results_type="numpy_per_summary"
        )

        assert isinstance(result, dict)
        assert "mean" in result
        assert "time_domain_array" in result
        # Check that summary types are separated
        for (
            summary_type
        ) in solver_with_arrays.summary_legend_per_variable.values():
            assert summary_type in result

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          'duration': 0.05}],
        indirect=True,
    )
    def test_from_solver_pandas_instantiation(self, solver_with_arrays):
        """Test pandas DataFrame instantiation from solver."""
        result = SolveResult.from_solver(
            solver_with_arrays, results_type="pandas"
        )

        assert isinstance(result, dict)
        assert "time_domain" in result
        assert "summaries" in result
        assert isinstance(result["time_domain"], pd.DataFrame)
        assert isinstance(result["summaries"], pd.DataFrame)


class TestSolveResultFromSolver:
    """Test SolveResult creation and methods using real solver instances."""

    @pytest.mark.parametrize(
        "solver_settings_override",
            [{
            "output_types": ["state", "observables", "time", "mean", "rms"],
            "duration": 0.05,
        }],
        indirect=True,
    )
    @pytest.mark.parametrize("system_override", ["linear"], indirect=True)
    def test_time_domain_legend_from_solver(self, solver_with_arrays):
        """Test time domain legend creation from real solver."""
        legend = SolveResult.time_domain_legend_from_solver(solver_with_arrays)

        assert isinstance(legend, dict)
        assert len(legend) > 0
        # Check that state variables are included
        for i, state_name in enumerate(solver_with_arrays.saved_states):
            assert legend[i] == state_name
        # Check that observables are included
        assert any(
            v.startswith("o") for v in legend.values() if isinstance(v, str)
        )

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_summary_legend_from_solver(self, solver_with_arrays):
        """Test summary legend creation from real solver."""
        legend = SolveResult.summary_legend_from_solver(solver_with_arrays)

        assert isinstance(legend, dict)
        # Check that summary metrics are in the legend values
        legend_values = list(legend.values())
        summary_metrics = [
            s
            for s in solver_with_arrays.output_types
            if any(s.startswith(metric) for metric in ["mean", "rms"])
        ]

        for metric in summary_metrics:
            assert any(metric in val for val in legend_values)

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_stride_order_from_solver(self, solver_with_arrays):
        """Test that stride order is correctly captured from solver."""
        result = SolveResult.from_solver(solver_with_arrays)

        assert (
            result._stride_order
            == solver_with_arrays.kernel.output_arrays.host._stride_order
        )

        # Test that run dimension is found correctly
        run_dim = result._stride_order['state'].index("run")
        assert isinstance(run_dim, int)
        assert 0 <= run_dim < len(result._stride_order)

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms", "peaks[2]"],
          "duration": 0.6}],
        indirect=True,
    )
    def test_full_result_matches_cpu_outputs(
        self,
        solver_with_arrays,
        cpu_batch_results,
        output_functions,
        precision,
        tolerance,
    ) -> None:
        """Ensure ``SolveResult`` values match CPU reference integrations."""

        result = SolveResult.from_solver(solver_with_arrays, results_type="full")
        axes = tuple(result._stride_order['state'].index(dim) for dim in (
            "time",
                                                                  "run", "variable"))
        time_domain = np.transpose(result.time_domain_array, axes)

        legend = result.time_domain_legend
        state_labels = set(solver_with_arrays.saved_states)
        observable_labels = set(solver_with_arrays.saved_observables)
        state_columns = [idx for idx, label in legend.items() if label in state_labels]
        observable_columns = [
            idx for idx, label in legend.items() if label in observable_labels
        ]
        state_ref, time_ref = extract_state_and_time(cpu_batch_results.state, output_functions)

        if state_columns:
            np.testing.assert_allclose(
                time_domain[:, :, state_columns],
                state_ref,
                atol=tolerance.abs_loose,
                rtol=tolerance.rel_loose,
                err_msg="state mismatch.\n"
                f"device: {time_domain[:, :, state_columns]}\n"
                f"reference: {cpu_batch_results.state}\n",
            )
        if observable_columns:
            np.testing.assert_allclose(
                time_domain[:, :, observable_columns],
                cpu_batch_results.observables,
                atol=tolerance.abs_loose,
                rtol=tolerance.rel_loose,
                err_msg="observables mismatch.\n"
                        f"device: {time_domain[:, :, observable_columns]}\n"
                        f"reference: {cpu_batch_results.observables}\n"
            )

        if result.summaries_array.size:
            summary_axes = tuple(
                result._stride_order['state'].index(dim) for dim in ("time",
                                                                 "run",
                                                             "variable")
            )
            summaries = np.transpose(result.summaries_array, summary_axes)
            summary_legend = result.summaries_legend
            state_prefixes = set(solver_with_arrays.summarised_states)
            observable_prefixes = set(solver_with_arrays.summarised_observables)
            state_summary_columns = [
                idx
                for idx, label in summary_legend.items()
                if label.split()[0] in state_prefixes
            ]
            observable_summary_columns = [
                idx
                for idx, label in summary_legend.items()
                if label.split()[0] in observable_prefixes
            ]

            if state_summary_columns:
                np.testing.assert_allclose(
                    summaries[:, :, state_summary_columns],
                    cpu_batch_results.state_summaries,
                    atol=tolerance.abs_loose,
                    rtol=tolerance.rel_loose,
                    err_msg="state summaries mismatch.\n"
                    f"device: {summaries[:, :, state_summary_columns]}\n"
                    f"reference: {cpu_batch_results.observables}\n",
                )
            if observable_summary_columns:
                np.testing.assert_allclose(
                    summaries[:,:, observable_summary_columns],
                    cpu_batch_results.observable_summaries,
                    atol=tolerance.abs_loose,
                    rtol=tolerance.rel_loose,
                    err_msg="observables summaries mismatch.\n"
                        f"device: {summaries[:,:, observable_summary_columns]}\n"
                        f"reference: {cpu_batch_results.observable_summaries}\n"
                )


class TestSolveResultProperties:
    """Test SolveResult property methods using real solver data."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_as_numpy_property(self, solver_with_arrays):
        """Test as_numpy property returns correct structure."""
        result = SolveResult.from_solver(solver_with_arrays)
        numpy_dict = result.as_numpy

        assert isinstance(numpy_dict, dict)
        assert "time_domain_array" in numpy_dict
        assert "summaries_array" in numpy_dict
        assert "time_domain_legend" in numpy_dict
        assert "summaries_legend" in numpy_dict

        # Verify arrays are copies
        assert np.array_equal(
            numpy_dict["time_domain_array"], result.time_domain_array
        )
        assert numpy_dict["time_domain_array"] is not result.time_domain_array

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_per_summary_arrays_property(self, solver_with_arrays):
        """Test per_summary_arrays property splits summaries correctly."""
        result = SolveResult.from_solver(solver_with_arrays)
        per_summary = result.per_summary_arrays

        assert isinstance(per_summary, dict)
        singlevar_legend = result._singlevar_summary_legend

        # Check that each summary type has its own array
        for summary_name in singlevar_legend.values():
            assert summary_name in per_summary
            assert isinstance(per_summary[summary_name], np.ndarray)

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
         'duration': 0.05}],
        indirect=True,
    )
    def test_as_pandas_property(self, solver_with_arrays):
        """Test as_pandas property creates proper DataFrames."""
        result = SolveResult.from_solver(solver_with_arrays)
        pandas_dict = result.as_pandas

        assert isinstance(pandas_dict, dict)
        assert "time_domain" in pandas_dict
        assert "summaries" in pandas_dict

        time_df = pandas_dict["time_domain"]
        summaries_df = pandas_dict["summaries"]

        assert isinstance(time_df, pd.DataFrame)
        assert isinstance(summaries_df, pd.DataFrame)

        # Check MultiIndex columns for multiple runs
        if result.time_domain_array.ndim == 3:
            assert isinstance(time_df.columns, pd.MultiIndex)
            # Check run naming
            run_levels = time_df.columns.get_level_values(0).unique()
            expected_runs = result.time_domain_array.shape[
                result._stride_order['state'].index("run")
            ]
            assert len(run_levels) == expected_runs
            for i, run_name in enumerate(run_levels):
                assert run_name == f"run_{i}"

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_active_outputs_property(self, solver_with_arrays):
        """Test active_outputs property returns correct ActiveOutputs."""
        result = SolveResult.from_solver(solver_with_arrays)
        active = result.active_outputs

        assert isinstance(active, ActiveOutputs)
        assert active == solver_with_arrays.active_output_arrays


class TestSolveResultDefaultBehavior:
    """Test SolveResult default instantiation and edge cases."""

    def test_default_instantiation(self):
        """Test SolveResult instantiation with defaults."""
        result = SolveResult()

        assert result.time_domain_array.size == 0
        assert result.summaries_array.size == 0
        assert result.time is not None and result.time.size == 0
        assert len(result.time_domain_legend) == 0
        assert len(result.summaries_legend) == 0
        assert result._stride_order == ("time", "run", "variable")

    def test_custom_stride_order(self):
        """Test SolveResult with custom stride order."""
        custom_order = ("variable", "run", "time")
        result = SolveResult(stride_order=custom_order)

        assert result._stride_order == custom_order

        # Test with mock data
        result.time_domain_array = np.random.rand(
            5, 4, 3
        )  # matches custom order
        result.summaries_array = np.random.rand(2, 4, 6)
        result._singlevar_summary_legend = {0: "mean", 1: "rms"}
        result.time_domain_legend = {0: "x0", 1: "x1", 2: "time"}

        run_dim = result._stride_order.index("run")
        assert run_dim == 1

        # Test per_summary_arrays uses correct dimension
        per_summary = result.per_summary_arrays
        assert isinstance(per_summary, dict)


class TestSolveResultPandasIntegration:
    """Test pandas-specific functionality."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_pandas_shape_consistency(self, solver_with_arrays):
        """Test pandas DataFrame shapes are consistent with array dimensions."""
        result = SolveResult.from_solver(solver_with_arrays)
        td_df, sum_df = (
            result.as_pandas["time_domain"],
            result.as_pandas["summaries"],
        )

        array_shape = result.time_domain_array.shape
        if result.time_domain_array.ndim == 3:
            # For 3D arrays: (time, run, variable) -> DataFrame (time, run*variable)
            expected_cols = array_shape[1] * array_shape[2]  # run * variable
            assert td_df.shape == (array_shape[0], expected_cols)

        # Check summaries DataFrame columns match legend
        if result.summaries_array.size > 0:
            expected_sum_cols = (
                len(result.summaries_legend) * array_shape[1]
                if result.summaries_array.ndim == 3
                else len(result.summaries_legend)
            )
            assert sum_df.shape[1] == expected_sum_cols

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"output_types": ["state", "observables", "time", "mean", "rms"],
          "duration": 0.05}],
        indirect=True,
    )
    def test_pandas_time_indexing(self, solver_with_arrays):
        """Test that time array is used as DataFrame index when available."""
        result = SolveResult.from_solver(solver_with_arrays)
        td_df = result.as_pandas["time_domain"]

        if result.time is not None and result.time.size > 0:
            # Check that time values are used as index
            if result.time.ndim == 1:
                assert np.array_equal(td_df.index.values, result.time)
            else:
                # For multi-run, should use time from first run or appropriate run
                expected_time = (
                    result.time[:, 0]
                    if result.time.shape[1] > 0
                    else result.time.flatten()
                )
                assert len(td_df.index) == len(expected_time)


class TestSolveResultErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_summaries_per_summary_arrays(self):
        """Test per_summary_arrays with no summaries."""
        result = SolveResult()
        result._active_outputs = ActiveOutputs(
            state_summaries=False, observable_summaries=False
        )

        per_summary = result.per_summary_arrays
        assert per_summary == {}
