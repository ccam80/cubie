"""Tests for BatchSolverKernel build_kernel simplification."""

import inspect
import pytest


class TestBatchSolverKernelSimplification:
    """Tests for the BatchSolverKernel local_scratch removal."""

    def test_build_kernel_no_local_scratch(self, solverkernel):
        """Verify build_kernel does not use local_scratch variable.

        The simplified kernel passes persistent_local directly to the
        loopfunction instead of allocating a separate local_scratch buffer.
        """
        # Force kernel build by accessing the kernel property
        kernel = solverkernel.kernel

        # Get the source code of the build_kernel method
        source = inspect.getsource(solverkernel.build_kernel)

        # Verify local_scratch is not used in the build_kernel method
        assert "local_scratch" not in source, (
            "build_kernel should not reference local_scratch after "
            "simplification"
        )

        # Verify alloc_local_scratch is not used
        assert "alloc_local_scratch" not in source, (
            "build_kernel should not use alloc_local_scratch after "
            "simplification"
        )

    def test_kernel_uses_persistent_local(self, solverkernel):
        """Verify the compiled kernel passes persistent_local to loopfunction.

        After simplification, the kernel should pass persistent_local
        directly to the loop function instead of a separately allocated
        local_scratch buffer.
        """
        # Force kernel build by accessing the kernel property
        kernel = solverkernel.kernel

        # Get the source code of the build_kernel method
        source = inspect.getsource(solverkernel.build_kernel)

        # Verify persistent_local is used in the loopfunction call
        # The pattern should be: loopfunction call with persistent_local arg
        assert "persistent_local" in source, (
            "build_kernel should use persistent_local in loopfunction call"
        )

        # Verify the kernel was built successfully (no exceptions)
        assert kernel is not None, "Kernel should build successfully"
