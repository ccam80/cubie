"""Shared fixtures for integrated numerical tests.

These tests exercise composed systems (algorithm + controller + output
functions + metrics wired into a loop or kernel) and compare device
results against CPU reference implementations.
"""

import numpy as np
import pytest

from cubie import MemoryManager
from tests._utils import _build_solver_instance


class MockMemoryManager(MemoryManager):
    """Memory manager with controlled memory info for chunking tests."""

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
def chunked_solved_solver(
    system, precision, low_mem_solver, driver_settings
):
    solver = low_mem_solver

    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

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
