"""Pytest fixtures for Numba CUDASIM MWE tests."""

import pytest

from tests.numba_mwe.allocator_factory import AllocatorFactory
from tests.numba_mwe.kernel_factory import KernelFactory


@pytest.fixture(scope="function")
def settings_dict(request):
    """Return settings dictionary for MWE tests.

    Default values:
    - array_size: 10 (size of output array)
    - buffer_size: 5 (size of local array in allocator)
    - n_threads: 7 (threads that do work, less than array_size)

    Accepts parametrization via request.param for overrides.
    """
    defaults = {
        "array_size": 10,
        "buffer_size": 5,
        "n_threads": 7,
    }

    if hasattr(request, "param") and request.param is not None:
        defaults.update(request.param)

    return defaults


@pytest.fixture(scope="function")
def allocator_factory(settings_dict):
    """Return a fresh AllocatorFactory instance.

    Uses buffer_size from settings_dict.
    """
    return AllocatorFactory(buffer_size=settings_dict["buffer_size"])


@pytest.fixture(scope="function")
def kernel(allocator_factory):
    """Return a compiled CUDA kernel.

    Creates a KernelFactory using the allocator_factory and
    returns the result of build().
    """
    kernel_factory = KernelFactory(allocator_factory)
    return kernel_factory.build()
