import numpy as np
import pytest

from numba import cuda
from cubie.integrators.driver_array import DriverArray


@pytest.fixture
def simple_driver():
    # 5 samples at times 0,1,2,3,4 -> 4 segments (0..3)
    times = np.linspace(0.0, 4.0, 5)
    drivers = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    return DriverArray(drivers, times, order=3, loop=False)


@pytest.fixture
def looping_driver():
    times = np.linspace(0.0, 4.0, 5)
    drivers = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    return DriverArray(drivers, times, order=3, loop=True)


def _run_partition(partition_device_fn, query_times):
    """Launch a CUDA kernel that applies the device partition() to each time sample."""
    n = query_times.size
    out = np.empty(n, dtype=np.int32)

    @cuda.jit
    def kernel(times, results):
        i = cuda.grid(1)
        if i < times.size:
            results[i] = partition_device_fn(times[i])

    d_times = cuda.to_device(query_times)
    d_out = cuda.to_device(out)

    threads_per_block = 64
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_out)
    d_out.copy_to_host(out)
    return out


def test_partition_clamp(simple_driver):
    partition_device, _ = simple_driver.build()

    query_times = np.array([
        -1.0,  # before start -> clamp 0
        0.0,   # start -> 0
        0.9999,  # just before 1.0 still in seg 0
        1.0,   # exactly boundary -> next segment
        3.5,   # interior last segment (segment 3)
        4.0,   # end -> clamp last segment (3)
        10.0,  # far beyond -> clamp last segment (3)
    ], dtype=np.float64)

    expected = np.array([0, 0, 0, 1, 3, 3, 3], dtype=np.int32)

    got = _run_partition(partition_device, query_times)
    assert np.array_equal(got, expected)


def test_partition_loop(looping_driver):
    partition_device, _ = looping_driver.build()

    query_times = np.array([
        0.0,  # start -> 0
        4.0,  # duration end wraps to 0
        5.0,  # 1 beyond end -> wraps to segment 1
    ], dtype=np.float64)

    expected = np.array([0, 0, 1], dtype=np.int32)

    got = _run_partition(partition_device, query_times)
    assert np.array_equal(got, expected)
