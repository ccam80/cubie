"""DEPRECATED: Buffer settings moved to ode_loop.py.

Import from cubie.integrators.loops.ode_loop instead.
This module is kept for backwards compatibility only.
"""

# Re-export from new location for backwards compatibility
from cubie.integrators.loops.ode_loop import (
    LoopBufferSettings,
    LoopLocalSizes,
    LoopSharedIndicesFromSettings,
    LoopSliceIndices,
)

__all__ = [
    "LoopBufferSettings",
    "LoopLocalSizes",
    "LoopSharedIndicesFromSettings",
    "LoopSliceIndices",
]
