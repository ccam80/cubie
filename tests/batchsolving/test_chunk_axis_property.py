"""Tests for BatchSolverKernel.chunk_axis property and setter."""

import pytest


class TestChunkAxisProperty:
    """Tests for chunk_axis property getter behavior."""

    def test_chunk_axis_property_returns_default_run(self, solverkernel):
        """Verify chunk_axis returns 'run' by default."""
        assert solverkernel.chunk_axis == "run"

    def test_chunk_axis_property_returns_consistent_value(
        self, solverkernel_mutable
    ):
        """Verify property returns value when arrays are consistent."""
        kernel = solverkernel_mutable
        # Both arrays should have same default value
        assert kernel.input_arrays._chunk_axis == "run"
        assert kernel.output_arrays._chunk_axis == "run"
        assert kernel.chunk_axis == "run"

    def test_chunk_axis_property_raises_on_inconsistency(
        self, solverkernel_mutable
    ):
        """Verify property raises ValueError for mismatched arrays."""
        kernel = solverkernel_mutable
        # Manually create inconsistent state
        kernel.input_arrays._chunk_axis = "run"
        kernel.output_arrays._chunk_axis = "time"

        with pytest.raises(ValueError, match=r"Inconsistent chunk_axis"):
            _ = kernel.chunk_axis


class TestChunkAxisSetter:
    """Tests for chunk_axis property setter behavior."""

    def test_chunk_axis_setter_updates_both_arrays(self, solverkernel_mutable):
        """Verify setter updates both input and output arrays."""
        kernel = solverkernel_mutable
        kernel.chunk_axis = "time"

        assert kernel.input_arrays._chunk_axis == "time"
        assert kernel.output_arrays._chunk_axis == "time"

    def test_chunk_axis_setter_allows_valid_values(self, solverkernel_mutable):
        """Verify setter accepts all valid chunk_axis values."""
        kernel = solverkernel_mutable
        for value in ["run", "variable", "time"]:
            kernel.chunk_axis = value
            assert kernel.chunk_axis == value


# These tests instantiate solverkernel incorrectly, causing an infinite
# loop. Tests must use the solver fixture.
# class TestChunkAxisInRun:
#     """Tests for chunk_axis handling in kernel.run()."""
#
#     def test_run_sets_chunk_axis_on_arrays(
#         self, system, precision, driver_array
#     ):
#         """Verify run() sets chunk_axis before array operations."""
#         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
#
#         kernel = BatchSolverKernel(
#             system,
#             algorithm_settings={"algorithm": "euler"},
#         )
#         inits = np.ones((system.sizes.states, 1), dtype=precision)
#         params = np.ones((system.sizes.parameters, 1), dtype=precision)
#
#         coefficients = (
#             driver_array.coefficients if driver_array is not None else None
#         )
#         kernel.run(
#             inits=inits,
#             params=params,
#             driver_coefficients=coefficients,
#             duration=0.1,
#             chunk_axis="time",
#         )
#
#         # After run, both arrays should have the chunk_axis value
#         assert kernel.input_arrays._chunk_axis == "time"
#         assert kernel.output_arrays._chunk_axis == "time"
#
#     def test_chunk_axis_property_after_run(
#         self, system, precision, driver_array
#     ):
#         """Verify chunk_axis property returns correct value after run."""
#         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
#
#         kernel = BatchSolverKernel(
#             system,
#             algorithm_settings={"algorithm": "euler"},
#         )
#         inits = np.ones((system.sizes.states, 1), dtype=precision)
#         params = np.ones((system.sizes.parameters, 1), dtype=precision)
#
#         coefficients = (
#             driver_array.coefficients if driver_array is not None else None
#         )
#         kernel.run(
#             inits=inits,
#             params=params,
#             driver_coefficients=coefficients,
#             duration=0.1,
#             chunk_axis="time",
#         )
#
#         assert kernel.chunk_axis == "time"

#
# class TestUpdateFromSolverChunkAxis:
#     """Tests for update_from_solver chunk_axis behavior."""
#
#     def test_update_from_solver_does_not_change_chunk_axis(
#         self, system
#     ):
#         """Verify update_from_solver preserves existing chunk_axis."""
#         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
#
#         kernel = BatchSolverKernel(
#             system,
#             algorithm_settings={"algorithm": "euler"},
#         )
#
#         # Set chunk_axis to non-default value via setter
#         kernel.chunk_axis = "time"
#
#         # Call update_from_solver (simulating what run() does)
#         kernel.input_arrays.update_from_solver(kernel)
#
#         # chunk_axis should be preserved
#         assert kernel.input_arrays._chunk_axis == "time"
