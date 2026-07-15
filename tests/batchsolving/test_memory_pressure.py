"""Live-solver memory pressure and host-array disk spill.

These tests assert that an idle persistent solver's device buffers are
evicted when a competing solver's allocation does not fit in free
memory (and that the evicted solver self-heals on its next solve), and
that host output arrays above the spill threshold are transparently
backed by disk without changing results.
"""

import numpy as np

from cubie.batchsolving.solver import Solver
from cubie.memory import MemoryManager


def _make_inputs(system, precision, n_runs=32):
    """Build verbatim ``(variable, run)`` inits and parameters."""
    y0 = np.repeat(
        system.initial_values.values_array[:, None].astype(precision),
        n_runs,
        axis=1,
    )
    params = np.repeat(
        system.parameters.values_array[:, None].astype(precision),
        n_runs,
        axis=1,
    )
    return y0, params


def test_idle_solver_evicted_under_pressure_and_self_heals(
    system, precision, low_memory
):
    """A competing solve evicts the idle solver, which then recovers.

    Both solvers share a manager whose reported free memory is smaller
    than either request, so every allocation runs under pressure. The
    second solver's allocation evicts the idle first solver's device
    buffers; the first solver's next solve reallocates and completes.
    """
    manager = low_memory
    y0, params = _make_inputs(system, precision)

    solver_a = Solver(
        system, algorithm="euler", dt=0.01, memory_manager=manager
    )
    solver_b = Solver(
        system, algorithm="euler", dt=0.01, memory_manager=manager
    )

    solver_a.solve(y0, params, duration=0.1)
    a_output_id = id(solver_a.kernel.output_arrays)
    assert manager.registry[a_output_id].allocated_bytes > 0

    solver_b.solve(y0, params, duration=0.1)
    assert manager.registry[a_output_id].allocated_bytes == 0

    result = solver_a.solve(y0, params, duration=0.1)
    assert manager.registry[a_output_id].allocated_bytes > 0
    assert np.isfinite(result.as_numpy["time_domain_array"]).all()


def test_host_arrays_spill_to_disk_and_results_match(
    system, precision, tmp_path
):
    """Outputs above the spill threshold are memmap-backed, bit-equal.

    A reference solve with default host arrays and a spill solve with
    a tiny threshold produce identical state trajectories; the spill
    solve's host state array is a ``numpy.memmap`` whose backing file
    lives in the configured spill directory.
    """
    y0, params = _make_inputs(system, precision)

    reference_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        save_every=0.05,
        memory_manager=MemoryManager(),
    )
    reference = reference_solver.solve(y0, params, duration=0.2)

    spill_solver = Solver(
        system,
        algorithm="euler",
        dt=0.01,
        save_every=0.05,
        memory_manager=MemoryManager(),
        host_spill_threshold=512,
        spill_directory=str(tmp_path),
    )
    spilled = spill_solver.solve(y0, params, duration=0.2)

    state_host = spill_solver.kernel.output_arrays.state
    assert isinstance(state_host, np.memmap)
    assert len(list(tmp_path.iterdir())) > 0

    np.testing.assert_array_equal(
        spilled.as_numpy["time_domain_array"],
        reference.as_numpy["time_domain_array"],
    )
