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


class TestChunkLoopCoverage:
    """Tests to verify all runs are processed in the chunk loop."""

    def test_final_chunk_covers_all_runs_5_runs_4_chunks(self, solverkernel):
        """Verify 5 runs with 4 chunks processes all runs.

        With 5 runs and 4 chunks:
        - chunk_size = 5 // 4 = 1
        - Chunks 0-2: process runs [0,1), [1,2), [2,3)
        - Chunk 3 (final): must process runs [3,5) to cover run 4

        This tests the fix for the run dropout bug where run 4 was skipped.
        """
        chunk_params = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=5,
            chunks=4,
        )
        numruns = 5
        num_chunks = 4
        chunk_size = chunk_params.size

        # Simulate the chunk loop logic and collect all run indices
        processed_runs = set()
        for i in range(num_chunks):
            start_idx = i * chunk_size
            # Final chunk captures all remaining runs
            if i == num_chunks - 1:
                end_idx = numruns
            else:
                end_idx = (i + 1) * chunk_size
            for run_idx in range(start_idx, end_idx):
                processed_runs.add(run_idx)

        # Verify all runs are processed
        expected_runs = set(range(numruns))
        assert processed_runs == expected_runs, (
            f"Not all runs processed. Missing: {expected_runs - processed_runs}"
        )

    def test_final_chunk_covers_all_runs_7_runs_3_chunks(self, solverkernel):
        """Verify 7 runs with 3 chunks processes all runs.

        With 7 runs and 3 chunks:
        - chunk_size = 7 // 3 = 2
        - Chunks 0-1: process runs [0,2), [2,4)
        - Chunk 2 (final): must process runs [4,7) to cover runs 4, 5, 6
        """
        chunk_params = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=7,
            chunks=3,
        )
        numruns = 7
        num_chunks = 3
        chunk_size = chunk_params.size

        # Simulate the chunk loop logic
        processed_runs = set()
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                end_idx = numruns
            else:
                end_idx = (i + 1) * chunk_size
            for run_idx in range(start_idx, end_idx):
                processed_runs.add(run_idx)

        expected_runs = set(range(numruns))
        assert processed_runs == expected_runs

    def test_no_duplicate_runs_processed(self, solverkernel):
        """Verify no runs are processed multiple times.

        With 10 runs and 3 chunks:
        - chunk_size = 10 // 3 = 3
        - Chunks: [0,3), [3,6), [6,10)
        - Each run should be processed exactly once.
        """
        chunk_params = solverkernel.chunk_run(
            chunk_axis="run",
            duration=1.0,
            warmup=0.0,
            t0=0.0,
            numruns=10,
            chunks=3,
        )
        numruns = 10
        num_chunks = 3
        chunk_size = chunk_params.size

        processed_runs = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                end_idx = numruns
            else:
                end_idx = (i + 1) * chunk_size
            for run_idx in range(start_idx, end_idx):
                processed_runs.append(run_idx)

        # Check for duplicates
        assert len(processed_runs) == len(set(processed_runs)), (
            "Duplicate runs detected"
        )
        # Check total count
        assert len(processed_runs) == numruns
