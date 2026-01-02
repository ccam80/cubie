"""Tests for reproducing Numba CUDASIM flaky bug.

The bug occurs when threads that should return early continue
executing and fail when calling cuda.local.array().

Error message: "module 'numba.cuda' has no attribute 'local'"

These tests repeatedly call a session-scoped kernel with varying
thread counts both within each test and across consecutive tests.
This attempts to trigger a race condition where one kernel run
isn't fully cleaned up before the next one starts.
"""

import numpy as np
from numba import cuda


def _run_kernel_single(kernel, array_size, n_threads):
    """Run kernel once with specified thread count.

    Parameters
    ----------
    kernel : callable
        Compiled CUDA kernel.
    array_size : int
        Size of the output array.
    n_threads : int
        Number of threads that should do work.
    """
    # Create output array
    output = np.zeros(array_size, dtype=np.float32)

    # Calculate grid dimensions
    threads_per_block = 32
    blocks = (array_size + threads_per_block - 1) // threads_per_block

    # Launch kernel
    kernel[(blocks,), (threads_per_block,)](output, n_threads)

    # Synchronize
    cuda.synchronize()

    # Verify results
    # First n_threads elements should be 1.0
    assert np.all(output[:n_threads] == 1.0), (
        f"Expected first {n_threads} elements to be 1.0, "
        f"got {output[:n_threads]}"
    )
    # Remaining elements should be 0.0
    if n_threads < array_size:
        assert np.all(output[n_threads:] == 0.0), (
            f"Expected elements [{n_threads}:] to be 0.0, "
            f"got {output[n_threads:]}"
        )


def _run_kernel_repeated(kernel, array_size, thread_counts):
    """Run kernel repeatedly with different thread counts.

    This is designed to trigger race conditions by rapidly
    calling the same kernel with different n_threads values.

    Parameters
    ----------
    kernel : callable
        Compiled CUDA kernel.
    array_size : int
        Size of the output array.
    thread_counts : list of int
        List of n_threads values to use in sequence.
    """
    for n_threads in thread_counts:
        _run_kernel_single(kernel, array_size, n_threads)


def test_mwe_case_01(kernel, settings_dict):
    """MWE test case 1 - ascending thread counts."""
    array_size = settings_dict["array_size"]
    thread_counts = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_02(kernel, settings_dict):
    """MWE test case 2 - descending thread counts."""
    array_size = settings_dict["array_size"]
    thread_counts = [30, 25, 20, 15, 10, 7, 5, 3, 1]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_03(kernel, settings_dict):
    """MWE test case 3 - alternating high/low."""
    array_size = settings_dict["array_size"]
    thread_counts = [1, 30, 2, 29, 3, 28, 4, 27, 5, 26]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_04(kernel, settings_dict):
    """MWE test case 4 - same count repeatedly."""
    array_size = settings_dict["array_size"]
    thread_counts = [7] * 20
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_05(kernel, settings_dict):
    """MWE test case 5 - edge cases."""
    array_size = settings_dict["array_size"]
    thread_counts = [1, array_size, 1, array_size, 1, array_size]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_06(kernel, settings_dict):
    """MWE test case 6 - rapid small variations."""
    array_size = settings_dict["array_size"]
    thread_counts = [10, 11, 10, 11, 10, 11, 12, 11, 12, 11, 12, 13]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_07(kernel, settings_dict):
    """MWE test case 7 - powers of two."""
    array_size = settings_dict["array_size"]
    thread_counts = [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_08(kernel, settings_dict):
    """MWE test case 8 - many single-thread runs."""
    array_size = settings_dict["array_size"]
    thread_counts = [1] * 30
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_09(kernel, settings_dict):
    """MWE test case 9 - many max-thread runs."""
    array_size = settings_dict["array_size"]
    thread_counts = [array_size] * 30
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_10(kernel, settings_dict):
    """MWE test case 10 - random-looking pattern."""
    array_size = settings_dict["array_size"]
    thread_counts = [7, 23, 4, 31, 2, 18, 9, 27, 1, 15, 6, 29, 3, 21, 8]
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_11(kernel, settings_dict):
    """MWE test case 11 - stress test with many iterations."""
    array_size = settings_dict["array_size"]
    thread_counts = list(range(1, array_size + 1)) * 3
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_12(kernel, settings_dict):
    """MWE test case 12 - reverse stress test."""
    array_size = settings_dict["array_size"]
    thread_counts = list(range(array_size, 0, -1)) * 3
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_13(kernel, settings_dict):
    """MWE test case 13 - zigzag pattern."""
    array_size = settings_dict["array_size"]
    thread_counts = []
    for i in range(1, array_size // 2 + 1):
        thread_counts.extend([i, array_size - i + 1])
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_14(kernel, settings_dict):
    """MWE test case 14 - rapid boundary transitions."""
    array_size = settings_dict["array_size"]
    thread_counts = [array_size - 1, array_size, array_size - 1] * 15
    _run_kernel_repeated(kernel, array_size, thread_counts)


def test_mwe_case_15(kernel, settings_dict):
    """MWE test case 15 - long run with all values."""
    array_size = settings_dict["array_size"]
    thread_counts = list(range(1, array_size + 1)) * 5
    _run_kernel_repeated(kernel, array_size, thread_counts)
