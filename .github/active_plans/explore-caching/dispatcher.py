# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Numba CUDA Dispatcher module - reference implementation for caching exploration.

This file contains key excerpts from numba_cuda/numba/cuda/dispatcher.py
relevant to understanding how caching is implemented in numba-cuda.

Key classes for caching:
- CUDACacheImpl: Implements reduce/rebuild for CUDA kernels
- CUDACache: The main cache class for CUDA kernels
- CUDADispatcher: Uses CUDACache for file-based caching
"""

# Key imports from the original file
from numba.cuda.core.caching import Cache, CacheImpl, NullCache


class CUDACacheImpl(CacheImpl):
    """
    Implementation class for CUDA kernel caching.
    
    This class provides:
    - reduce(): Serializes a kernel for storage
    - rebuild(): Deserializes a kernel from storage
    - check_cachable(): Determines if a kernel can be cached
    """
    
    def reduce(self, kernel):
        """Reduce kernel to serializable form for caching."""
        return kernel._reduce_states()

    def rebuild(self, target_context, payload):
        """Rebuild kernel from cached payload."""
        # _Kernel._rebuild reconstructs the kernel from serialized state
        from numba.cuda.dispatcher import _Kernel
        return _Kernel._rebuild(**payload)

    def check_cachable(self, cres):
        """Check if the compile result is cachable.
        
        CUDA Kernels are always cachable because:
        - They don't have lifted loops
        - They don't use dynamic globals
        """
        return True


class CUDACache(Cache):
    """
    Cache implementation for CUDA kernels.
    
    Saves and loads CUDA kernels and compile results to/from disk.
    Uses IndexDataCacheFile for the actual file operations.
    
    Key methods inherited from Cache:
    - load_overload(sig, target_context): Load cached kernel
    - save_overload(sig, data): Save kernel to cache
    - flush(): Clear the cache
    - enable()/disable(): Control caching behavior
    
    The cache key is computed from:
    - Function signature
    - Target architecture (codegen.magic_tuple())
    - Hash of function bytecode
    - Hash of closure variables (if any)
    """
    
    _impl_class = CUDACacheImpl

    def load_overload(self, sig, target_context):
        """Load overload with CUDA-specific context handling."""
        # Loading an overload refreshes the context to ensure it is initialized
        # numba_target_override ensures proper target context during load
        from numba.cuda import utils
        with utils.numba_target_override():
            return super().load_overload(sig, target_context)


# Key excerpt from CUDADispatcher showing cache usage:
"""
class CUDADispatcher:
    def __init__(self, py_func, targetoptions, pipeline_class=CUDACompiler):
        # ... initialization ...
        self._cache = NullCache()  # Default: no caching
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()
        
    def enable_caching(self):
        '''Enable file-based caching for this dispatcher.'''
        self._cache = CUDACache(self.py_func)
        
    @global_compiler_lock
    def compile(self, sig):
        '''Compile kernel for given signature, using cache when available.'''
        argtypes, return_type = sigutils.normalize_signature(sig)
        
        # Check memory cache first
        if self.specialized:
            return next(iter(self.overloads.values()))
        else:
            kernel = self.overloads.get(argtypes)
            if kernel is not None:
                return kernel
        
        # Try to load from disk cache
        kernel = self._cache.load_overload(sig, self.targetctx)
        
        if kernel is not None:
            self._cache_hits[sig] += 1
        else:
            # Compile new kernel
            self._cache_misses[sig] += 1
            if not self._can_compile:
                raise RuntimeError("Compilation disabled")
            
            kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
            # Force codegen to get cubin for caching
            kernel.bind()
            # Save to disk cache
            self._cache.save_overload(sig, kernel)
        
        self.add_overload(kernel, argtypes)
        return kernel
        
    @property
    def stats(self):
        return _CompileStats(
            cache_path=self._cache.cache_path,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )
"""

# Note: The _Kernel class has these methods for serialization:
"""
class _Kernel:
    def _reduce_states(self):
        '''Reduce for serialization. PTX form, no stream info.'''
        return dict(
            cooperative=self.cooperative,
            name=self.entry_name,
            signature=self.signature,
            codelibrary=self._codelibrary,
            debug=self.debug,
            lineinfo=self.lineinfo,
            call_helper=self.call_helper,
            extensions=self.extensions,
        )
    
    @classmethod
    def _rebuild(cls, cooperative, name, signature, codelibrary, 
                 debug, lineinfo, call_helper, extensions):
        '''Rebuild an instance from serialized state.'''
        instance = cls.__new__(cls)
        super(cls, instance).__init__()
        instance.entry_point = None
        instance.cooperative = cooperative
        instance.entry_name = name
        instance.signature = signature
        instance._type_annotation = None
        instance._codelibrary = codelibrary
        instance.debug = debug
        instance.lineinfo = lineinfo
        instance.call_helper = call_helper
        instance.extensions = extensions
        return instance
"""
