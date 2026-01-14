import numpy as np
import pytest

from cubie import MemoryManager
from tests._utils import _build_solver_instance


class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def __init__(self, **kwargs):
        self._custom_limit = kwargs.get("forced_free_mem", 512)

    def get_memory_info(self):
        return int(self._custom_limit), int(8192)  # 32kb free, total


@pytest.fixture(scope="session")
def forced_free_mem(request):
    if hasattr(request, "param"):
        return request.param
    return 512


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

    # This run has a combined request size of 1968b, so a 512 limit forces
    # one run per chunk (4 chunks, then runs rounded to 1, then chunks=5)
    # A 1024 limit forces a  3 and a 2.
    # Run with forced chunking (low memory)
    result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
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
        chunk_axis=chunk_axis,
    )
    return solver, result
