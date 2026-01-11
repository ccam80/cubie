# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# Fetched from NVIDIA/numba-cuda repository on 2026-01-11
# Source: numba_cuda/numba/cuda/dispatcher.py
#
# NOTE: This is a large file. Only the caching-related portions are included
# below for reference. The full file is very large (~2500 lines).
# Key caching-related classes and code:

"""
Key caching-related imports in dispatcher.py:
    from numba.cuda.core.caching import Cache, CacheImpl, NullCache

Key caching-related classes:

class CUDACacheImpl(CacheImpl):
    def reduce(self, kernel):
        return kernel._reduce_states()

    def rebuild(self, target_context, payload):
        return _Kernel._rebuild(**payload)

    def check_cachable(self, cres):
        # CUDA Kernels are always cachable - the reasons for an entity not to
        # be cachable are:
        #
        # - The presence of lifted loops, or
        # - The presence of dynamic globals.
        #
        # neither of which apply to CUDA kernels.
        return True


class CUDACache(Cache):
    '''
    Implements a cache that saves and loads CUDA kernels and compile results.
    '''

    _impl_class = CUDACacheImpl

    def load_overload(self, sig, target_context):
        # Loading an overload refreshes the context to ensure it is initialized.
        with utils.numba_target_override():
            return super().load_overload(sig, target_context)


In CUDADispatcher:
    def __init__(self, py_func, targetoptions, pipeline_class=CUDACompiler):
        ...
        self._cache = NullCache()
        ...
    
    def enable_caching(self):
        self._cache = CUDACache(self.py_func)

    def compile(self, sig):
        '''
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        '''
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None or return_type == types.none

        # Do we already have an in-memory compiled kernel?
        if self.specialized:
            return next(iter(self.overloads.values()))
        else:
            kernel = self.overloads.get(argtypes)
            if kernel is not None:
                return kernel

        # Can we load from the disk cache?
        kernel = self._cache.load_overload(sig, self.targetctx)

        if kernel is not None:
            self._cache_hits[sig] += 1
        else:
            # We need to compile a new kernel
            self._cache_misses[sig] += 1
            if not self._can_compile:
                raise RuntimeError("Compilation disabled")

            kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
            # We call bind to force codegen, so that there is a cubin to cache
            kernel.bind()
            self._cache.save_overload(sig, kernel)

        self.add_overload(kernel, argtypes)

        return kernel
"""

# The full dispatcher.py file is approximately 2500 lines.
# See the NVIDIA/numba-cuda repository for the complete source:
# https://github.com/NVIDIA/numba-cuda/blob/main/numba_cuda/numba/cuda/dispatcher.py
