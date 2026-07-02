"""Lowering registrations that fill gaps in numba-cuda-mlir.

numba-cuda-mlir 0.4.0 registers bitwise binary operators
(``&``, ``|``, ``^`` and their in-place forms) for
``(Integer, Integer)`` signatures only; Boolean operands raise
``NotImplementedError`` during MLIR lowering. CuBIE device code uses
branch-free Boolean flag updates (``finished &= save_finished``),
so this module registers the missing Boolean signatures using the
same ``arith.andi``/``ori``/``xori`` code generation the package
uses for integers (all three operate on ``i1``).

Import this module before compiling any kernel; registrations are
picked up when the MLIR target context refreshes its registries.
This is a stop-gap that belongs upstream in numba-cuda-mlir.
"""

import operator

from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir._mlir.dialects import arith
from numba_cuda_mlir.lowering.math import registry as _math_registry
from numba_cuda_mlir.numba_cuda import types


def _make_bitwise_cg(mlir_op):
    """Build a code generator emitting ``mlir_op`` on cast operands.

    Parameters
    ----------
    mlir_op
        MLIR ``arith`` dialect builder (``andi``, ``ori`` or
        ``xori``) applied to the two operands.

    Returns
    -------
    Callable
        Lowering function with the ``(builder, target, args,
        kwargs)`` signature expected by the lowering registry.
    """

    def _cg(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        target_mlir_type = builder.get_mlir_type(target_type)
        lhs, rhs = args
        lhs = lowering_utilities.convert(
            builder.load_var(lhs), target_mlir_type
        )
        rhs = lowering_utilities.convert(
            builder.load_var(rhs), target_mlir_type
        )
        builder.store_var(target, mlir_op(lhs, rhs))

    return _cg


_BITWISE_OPS = {
    operator.and_: arith.andi,
    operator.iand: arith.andi,
    operator.or_: arith.ori,
    operator.ior: arith.ori,
    operator.xor: arith.xori,
    operator.ixor: arith.xori,
}

_BOOLEAN_SIGNATURES = (
    (types.Boolean, types.Boolean),
    (types.Boolean, types.Integer),
    (types.Integer, types.Boolean),
)


def _invert_cg(builder, target, args, kwargs):
    """Lower ``~x``: xor with all-ones (-1 for ints, true for bools)."""

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    operand = lowering_utilities.convert(
        builder.load_var(args[0]), target_mlir_type
    )
    fill = 1 if isinstance(target_type, types.Boolean) else -1
    ones = lowering_utilities.constant(fill, operand.type)
    builder.store_var(target, arith.xori(operand, ones))


def register_boolean_bitwise_lowerings() -> None:
    """Register bitwise lowerings absent from numba-cuda-mlir."""

    for op, mlir_op in _BITWISE_OPS.items():
        cg = _make_bitwise_cg(mlir_op)
        for lhs_type, rhs_type in _BOOLEAN_SIGNATURES:
            _math_registry.lower(op, lhs_type, rhs_type)(cg)
    _math_registry.lower(operator.invert, types.Boolean)(_invert_cg)
    _math_registry.lower(operator.invert, types.Integer)(_invert_cg)


register_boolean_bitwise_lowerings()
