"""Solver and array-manager GPU-resource teardown.

These tests assert that a solver releases its memory-manager
registration and device allocations when it is garbage collected, when
its ``close`` is called, and when it is used as a context manager, and
that repeatedly building and dropping solvers does not accumulate
registry entries.
"""

import gc

import numpy as np

from cubie.batchsolving.solver import Solver


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


def _instance_ids(solver):
    """Return the ids of the three memory-manager clients of a solver."""
    kernel = solver.kernel
    return (
        id(kernel),
        id(kernel.input_arrays),
        id(kernel.output_arrays),
    )


def _still_registered(manager, ids):
    """Return the subset of ``ids`` still present in the registry."""
    return [instance_id for instance_id in ids if instance_id in
            manager.registry]


def _registered_bytes(manager, ids):
    """Total device bytes the registry keeps alive for ``ids``."""
    total = 0
    for instance_id in ids:
        settings = manager.registry.get(instance_id)
        if settings is not None:
            total += settings.allocated_bytes
    return total


def test_solver_releases_registry_on_gc(
    system, precision, thread_mem_manager
):
    """Dropping the last reference frees the registry and its buffers.

    No further registration happens after the ``del``, so this only
    passes if the finalizer runs at collection time rather than waiting
    for a later purge.
    """
    manager = thread_mem_manager
    solver = Solver(system, algorithm="euler", dt=0.01, memory_manager=manager)
    y0, params = _make_inputs(system, precision)
    solver.solve(y0, params, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)
    assert _registered_bytes(manager, ids) > 0

    del solver
    gc.collect()

    assert _still_registered(manager, ids) == []


def test_close_releases_registry_immediately(
    system, precision, thread_mem_manager
):
    """``close`` deregisters without waiting for garbage collection."""
    manager = thread_mem_manager
    solver = Solver(system, algorithm="euler", dt=0.01, memory_manager=manager)
    y0, params = _make_inputs(system, precision)
    solver.solve(y0, params, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)

    solver.close()

    assert _still_registered(manager, ids) == []
    # close is idempotent and safe to call again.
    solver.close()
    assert _still_registered(manager, ids) == []


def test_context_manager_releases_on_exit(
    system, precision, thread_mem_manager
):
    """Using the solver as a context manager frees it at block exit."""
    manager = thread_mem_manager
    y0, params = _make_inputs(system, precision)
    with Solver(
        system, algorithm="euler", dt=0.01, memory_manager=manager
    ) as solver:
        solver.solve(y0, params, duration=0.1)
        ids = _instance_ids(solver)
        assert _still_registered(manager, ids) == list(ids)

    assert _still_registered(manager, ids) == []


def test_repeated_solvers_do_not_grow_registry(
    system, precision, thread_mem_manager
):
    """A build/solve/drop loop leaves the registry no larger than before."""
    manager = thread_mem_manager
    y0, params = _make_inputs(system, precision)

    gc.collect()
    baseline = len(manager.registry)

    for _ in range(6):
        solver = Solver(
            system, algorithm="euler", dt=0.01, memory_manager=manager
        )
        solver.solve(y0, params, duration=0.1)
        del solver
        gc.collect()

    assert len(manager.registry) <= baseline
