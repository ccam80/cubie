"""Pytest fixtures for Numba CUDASIM MWE tests."""

import pytest

from tests.numba_mwe.allocator_factory import AllocatorFactory
from tests.numba_mwe.kernel_factory import KernelFactory


@pytest.fixture(scope="session")
def settings_dict():
    """Return settings dictionary for MWE tests.

    Default values:
    - array_size: 32 (size of output array, larger to have more excess threads)
    - buffer_size: 5 (size of local array in allocator)
    """
    return {
        "array_size": 32,
        "buffer_size": 5,
    }


@pytest.fixture(scope="session")
def allocator_factory(settings_dict):
    """Return a session-scoped AllocatorFactory instance.

    Uses buffer_size from settings_dict.
    """
    return AllocatorFactory(buffer_size=settings_dict["buffer_size"])


@pytest.fixture(scope="session")
def kernel(allocator_factory):
    """Return a session-scoped compiled CUDA kernel.

    The kernel is reused across all tests to force potential race
    conditions when consecutive tests call it with different thread counts.
    """
    kernel_factory = KernelFactory(allocator_factory)
    return kernel_factory.build()
