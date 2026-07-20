"""Process-stable identity hashing for generic test kernels.

The precompile plugin caches every dispatcher the test suite creates,
including kernels defined inline in test files. Those have no cubie
system identity, so the plugin keys them by function identity: code
object, closure values, defaults, and target options. The helpers here
strip process- and checkout-specific detail (file paths, line offsets,
serialization order) so a kernel compiled on a CPU runner produces the
same key when the GPU runner looks it up.

Kept out of :mod:`cubie.cubie_cache`: production caches are keyed by
system and compile-settings hashes and never need function identity.
"""
import marshal
from hashlib import sha256
from types import CodeType, FunctionType
from typing import Optional

# Whichever backend serializer is installed; population and consumer
# runs share a pinned environment, so both resolve identically.
try:
    from numba.cuda.serialize import dumps as cache_dumps
except ImportError:
    from numba_cuda_mlir.numba_cuda.serialize import (
        dumps as cache_dumps,
    )


def _portable_const(value):
    """Return a deterministically ordered stand-in for a constant.

    The compiler folds ``in {...}`` and ``in (...)`` membership tests
    into frozenset constants, and before Python 3.12 ``marshal``
    serialized their members in hash-iteration order, which varies
    with each process's hash seed. Replace every frozenset with a
    marker tuple of its members sorted by their own marshal bytes so
    the code hash is identical across processes and machines. The
    result is only marshaled for hashing, never executed.
    """
    if isinstance(value, CodeType):
        return _portable_code(value)
    if isinstance(value, frozenset):
        members = sorted(
            (_portable_const(member) for member in value),
            key=marshal.dumps,
        )
        return ("__frozenset__",) + tuple(members)
    if isinstance(value, tuple):
        return tuple(_portable_const(member) for member in value)
    return value


def _portable_code(code: CodeType) -> CodeType:
    """Remove source locations and hash-order detail from a code object."""
    constants = tuple(
        _portable_const(value) for value in code.co_consts
    )
    return code.replace(
        co_filename="", co_firstlineno=1, co_consts=constants
    )


def _stable_value_key(value, active: set[int]):
    """Return a process-stable key for a captured value."""
    py_func = getattr(value, "py_func", None)
    if py_func is not None:
        return (
            "dispatcher",
            _function_key(py_func, active),
            _stable_value_key(value.targetoptions, active),
        )
    if isinstance(value, FunctionType):
        # The serialized fallback pickles plain functions by value,
        # embedding co_filename, which differs between the CPU
        # population checkout and the GPU consumer checkout.
        return ("function", _function_key(value, active))
    if isinstance(value, CodeType):
        code_hash = sha256(marshal.dumps(_portable_code(value))).hexdigest()
        return ("code", code_hash)
    if value.__class__.__name__ == "FastMathOptions":
        return ("fastmath", tuple(sorted(value.flags)))
    if isinstance(value, tuple):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("tuple", items)
    if isinstance(value, list):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("list", items)
    if isinstance(value, dict):
        items = (
            (_stable_value_key(key, active), _stable_value_key(item, active))
            for key, item in value.items()
        )
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        items = (_stable_value_key(item, active) for item in value)
        return ("set", tuple(sorted(items, key=repr)))
    return ("serialized", sha256(cache_dumps(value)).hexdigest())


def _stable_value_hash(value) -> str:
    """Hash a value without process-specific serialization order."""
    key = _stable_value_key(value, set())
    return sha256(cache_dumps(key)).hexdigest()


def _function_key(py_func, active: Optional[set[int]] = None):
    """Identify function code and compile-time captures."""
    if active is None:
        active = set()
    identity = id(py_func)
    if identity in active:
        return ("recursive", py_func.__module__, py_func.__qualname__)
    active.add(identity)
    try:
        closure = py_func.__closure__ or ()
        closure_key = tuple(
            _stable_value_key(cell.cell_contents, active) for cell in closure
        )
        defaults_key = _stable_value_key(
            (py_func.__defaults__, py_func.__kwdefaults__), active
        )
        code_hash = sha256(
            marshal.dumps(_portable_code(py_func.__code__))
        ).hexdigest()
        return (
            py_func.__module__,
            py_func.__qualname__,
            sha256(cache_dumps(closure_key)).hexdigest(),
            code_hash,
            sha256(cache_dumps(defaults_key)).hexdigest(),
        )
    finally:
        active.remove(identity)


def _portable_magic(value):
    """Normalize backend target values used in cache keys."""
    if hasattr(value, "major") and hasattr(value, "minor"):
        return (int(value.major), int(value.minor))
    if isinstance(value, tuple):
        return tuple(_portable_magic(item) for item in value)
    return value
