"""
Test script to explore Numba CUDA function detection methods.
"""

from numba import cuda


def is_numba_cuda_device_function(func):
    """
    Test whether a function handle is a Numba CUDA device function.

    Parameters
    ----------
    func : callable
        Function to test

    Returns
    -------
    bool
        True if the function is a Numba CUDA device function, False otherwise
    """
    is_cuda = False
    is_device = False
    # Method 2: Check for CUDA-specific attributes
    if hasattr(func, "targetoptions"):
        is_cuda = True
        if func.targetoptions.get("device", False):
            is_device = True

    return is_cuda, is_device


# Test functions
@cuda.jit(device=True)
def cuda_device_func(x, y):
    """A simple CUDA device function."""
    return x + y


@cuda.jit(device=False)
def cuda_kernel(x, y):
    """A regular Python function."""
    y = x


def noncuda_func(x, y):
    """A regular Python function."""
    return x + y


def main():
    """Test the detection function."""
    for callable in [cuda_device_func, cuda_kernel, noncuda_func]:
        is_cuda, is_device = is_numba_cuda_device_function(callable)
        print(
            f"{callable.__name__} is cuda: {is_cuda}, is device: {is_device}"
        )

main()