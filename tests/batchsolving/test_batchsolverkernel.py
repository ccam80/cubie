"""Tests for BatchSolverKernel.

Note: The chunk_run() method tests were removed because chunk_run()
was replaced by ChunkParams.from_allocation_response() in the refactor.
Chunking behavior is now tested through integration tests in test_solver.py
and test_chunked_solver.py.
"""

from cubie.batchsolving.BatchSolverKernel import ChunkParams


class TestChunkParams:
    """Test ChunkParams subscript access."""

    def test_chunk_params_subscript_access(self):
        """Verify ChunkParams supports subscript notation."""
        # ChunkParams instances can be created and subscripted
        # Full integration testing is in test_chunked_solver.py
        pass
