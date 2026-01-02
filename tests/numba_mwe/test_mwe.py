"""Tests for reproducing Numba CUDASIM flaky bug.

The bug occurs when threads that should return early continue
executing and fail when calling cuda.local.array().

Error message: "module 'numba.cuda' has no attribute 'local'"

Having multiple identical tests increases the probability of
triggering the flaky bug since each test creates fresh fixture
instances due to function scope.
"""

import numpy as np
from numba import cuda


def _run_kernel_test(kernel, settings_dict):
    """Helper to run kernel and verify output.

    Parameters
    ----------
    kernel : callable
        Compiled CUDA kernel.
    settings_dict : dict
        Test settings with array_size and n_threads.
    """
    array_size = settings_dict["array_size"]
    n_threads = settings_dict["n_threads"]

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


def test_mwe_case_01(kernel, settings_dict):
    """MWE test case 1."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_02(kernel, settings_dict):
    """MWE test case 2."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_03(kernel, settings_dict):
    """MWE test case 3."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_04(kernel, settings_dict):
    """MWE test case 4."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_05(kernel, settings_dict):
    """MWE test case 5."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_06(kernel, settings_dict):
    """MWE test case 6."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_07(kernel, settings_dict):
    """MWE test case 7."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_08(kernel, settings_dict):
    """MWE test case 8."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_09(kernel, settings_dict):
    """MWE test case 9."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_10(kernel, settings_dict):
    """MWE test case 10."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_11(kernel, settings_dict):
    """MWE test case 11."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_12(kernel, settings_dict):
    """MWE test case 12."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_13(kernel, settings_dict):
    """MWE test case 13."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_14(kernel, settings_dict):
    """MWE test case 14."""
    _run_kernel_test(kernel, settings_dict)


def test_mwe_case_15(kernel, settings_dict):
    """MWE test case 15."""
    _run_kernel_test(kernel, settings_dict)
