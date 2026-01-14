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
        return int(self._custom_limit), int(8192)  # 32kb free, total


@pytest.fixture(scope="session")
def forced_free_mem(request):
    if hasattr(request, "param"):
        return request.param
    return 950


@pytest.fixture(scope="session")
def low_memory(forced_free_mem):
    return MockMemoryManager(forced_free_mem=forced_free_mem)


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def chunk_axis(request):
    if hasattr(request, "param"):
        return request.param
    return "run"


@pytest.fixture(scope="session")
def chunked_solved_solver(
    system, precision, low_mem_solver, driver_settings, chunk_axis
):
    solver = low_mem_solver

    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    # This run has a combined request size of 1668b, with 1080 chunkable/588
    # unchunkable if axis is run, and 1300b chunkable/368b unchunkable if time.
    # For one run per chunk:
    #  - run axis: free > 588 + 1080/5 -> 850
    #  - time axis: free > 368 + 1300/5 -> 630b
    # Two runs per (2-2-1):
    #  - run axis: free > 588 + 1080/(5/2) -> 1024b
    #  - time axis: free > 368 + 1300*2/5 -> 890b
    # Three runs per (3-2):
    # - run axis: free > 588 + 1080/(5/3) -> 1240
    # - time axis: free > 368 + 1300*3/5 -> 1150b
    # Four runs per (4-1):
    # - run axis: free > 588 + 1080/(5/4) 0> 1460
    # - time axis: free > 368 + 1300*4/5 -> 1420
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
        chunk_axis=chunk_axis,
    )
    return solver, result


@pytest.fixture(scope="session")
def unchunked_solved_solver(
    system,
    precision,
    driver_settings,
    unchunking_solver,
    chunk_axis,
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
        chunk_axis=chunk_axis,
    )
    return solver, result
