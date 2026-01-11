# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# 
# Fetched from NVIDIA/numba-cuda repository on 2026-01-11
# Source: numba_cuda/numba/cuda/dispatcher.py
#
# Note: This file is very large. The key relevant parts for CuBIE caching are:
# - CUDACacheImpl class (line ~1030)
# - CUDACache class (line ~1050)
# - CUDADispatcher.enable_caching method
# - The imports from numba.cuda.core.caching

"""
Key excerpts relevant to caching:

from numba.cuda.core.caching import Cache, CacheImpl, NullCache

class CUDACacheImpl(CacheImpl):
    def reduce(self, kernel):
        return kernel._reduce_states()

    def rebuild(self, target_context, payload):
        return _Kernel._rebuild(**payload)

    def check_cachable(self, cres):
        return True


class CUDACache(Cache):
    _impl_class = CUDACacheImpl

    def load_overload(self, sig, target_context):
        with utils.numba_target_override():
            return super().load_overload(sig, target_context)


class CUDADispatcher:
    def __init__(self, ...):
        ...
        self._cache = NullCache()
        ...

    def enable_caching(self):
        self._cache = CUDACache(self.py_func)
"""

# Simulator dispatcher is minimal - just a stub class:
# class CUDADispatcher:
#     """
#     Dummy class so that consumers that try to import the real CUDADispatcher
#     do not get an import failure when running with the simulator.
#     """
#     ...
