"""Tests for BatchSolverKernel chunking logic."""

from cubie.batchsolving.BatchSolverKernel import ChunkParams


class TestChunkRunFloorDivision:
    """Tests for chunk_run floor division logic."""

    def test_chunk_run_uses_floor_division(self, solverkernel):
        """Verify chunk_run uses floor division for run count.

        With 10 runs and 3 chunks, floor division gives:
        - chunk_size = 10 // 3 = 3
        This differs from ceiling division (4).
        """
        result = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=10,
            chunks=3,
        )
        assert isinstance(result, ChunkParams)
        # Floor division: 10 // 3 = 3
        assert result.runs == 3
        assert result.size == 3

    def test_chunk_run_handles_uneven_division(self, solverkernel):
        """Verify chunk_run handles numruns not divisible by chunks.

        With 5 runs and 4 chunks:
        - Floor division: 5 // 4 = 1
        - Each chunk processes 1 run
        - Final chunk processes the remaining run (handled in loop)
        """
        result = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=5,
            chunks=4,
        )
        assert result.runs == 1
        assert result.size == 1

    def test_chunk_run_minimum_one_run_per_chunk(self, solverkernel):
        """Verify chunk_run enforces minimum of 1 run per chunk.

        When chunks > numruns, floor division would give 0.
        The max(1, ...) ensures at least 1 run per chunk.
        """
        result = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=2,
            chunks=5,
        )
        # 2 // 5 = 0, but max(1, 0) = 1
        assert result.runs == 1
        assert result.size == 1

    def test_chunk_run_single_chunk_returns_all_runs(self, solverkernel):
        """Verify single chunk returns original run count."""
        result = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=10,
            chunks=1,
        )
        assert result.runs == 10
        assert result.size == 10

    def test_chunk_run_time_axis_uses_floor_division(
        self, solverkernel_mutable
    ):
        """Verify chunk_run uses floor division for time axis."""
        # Set duration to trigger output_length calculation
        solverkernel_mutable._duration = 1.0
        output_length = solverkernel_mutable.output_length
        chunks = 3

        result = solverkernel_mutable.chunk_run(
            chunk_axis="time",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=10,
            chunks=chunks,
        )
        # Floor division for output_length
        expected_size = max(1, output_length // chunks)
        assert result.size == expected_size
        # Time chunking preserves original run count
        assert result.runs == 10
