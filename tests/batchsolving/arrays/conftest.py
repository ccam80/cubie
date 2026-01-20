import numpy as np
import pytest

from cubie import MemoryManager
from tests._utils import _build_solver_instance


class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def __init__(self, **kwargs):
        super().__init__()
        self._custom_limit = kwargs.get("forced_free_mem", 950)

    def get_memory_info(self):
        return int(self._custom_limit), int(8192)


@pytest.fixture(scope="function")
def forced_free_mem(request):
    if hasattr(request, "param"):
        return request.param
    return 950


@pytest.fixture(scope="function")
def low_memory(forced_free_mem):
    return MockMemoryManager(forced_free_mem=forced_free_mem)


@pytest.fixture(scope="function")
def low_mem_solver(
    system,
    solver_settings,
    driver_array,
    low_memory,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
        memory_manager=low_memory,
    )


@pytest.fixture(scope="session")
def unchunking_solver(
    system,
    solver_settings,
    driver_array,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
    )


@pytest.fixture(scope="function")
def chunked_solved_solver(system, precision, low_mem_solver, driver_settings):
    solver = low_mem_solver

    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    # This run has a combined request size of 1668b, with 1080 chunkable/588
    # unchunkable along the run axis.
    # For one run per chunk:
    #  - run axis: free > 588 + 1080/5 -> 850
    # Two runs per (2-2-1):
    #  - run axis: free > 588 + 1080/(5/2) -> 1024b
    # Three runs per (3-2):
    # - run axis: free > 588 + 1080/(5/3) -> 1240
    # Four runs per (4-1):
    # - run axis: free > 588 + 1080/(5/4) 0> 1460
    # Unchunked (5-0):
    # - 2048
    result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    
    yield solver, result
    
    # Cleanup: Ensure all writebacks complete and watcher properly shut down
    # Wait without timeout to prevent abandoning pending tasks
    if hasattr(solver.kernel.output_arrays, '_watcher'):
        solver.kernel.output_arrays.wait_pending(timeout=None)
        solver.kernel.output_arrays._watcher.shutdown()
    solver.kernel.output_arrays.reset()


@pytest.fixture(scope="session")
def unchunked_solved_solver(
    system,
    precision,
    driver_settings,
    unchunking_solver,
):
    solver = unchunking_solver
    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    # Run without chunking
    result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    return solver, result
