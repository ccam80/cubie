"""Lowering registrations that fill gaps in numba-cuda-mlir.

numba-cuda-mlir 0.4.0 registers bitwise binary operators
(``&``, ``|``, ``^`` and their in-place forms) for
``(Integer, Integer)`` signatures only; Boolean operands raise
``NotImplementedError`` during MLIR lowering. CuBIE device code uses
branch-free Boolean flag updates (``finished &= save_finished``),
so this module registers the missing Boolean signatures using the
same ``arith.andi``/``ori``/``xori`` code generation the package
uses for integers (all three operate on ``i1``).

numba-cuda-mlir also widens signless integers with zero-extension
(``arith.extui``) by default, corrupting negative signed operands of
``%`` and ``//`` on runtime values, and its integer ``%`` emits a
truncated remainder (``arith.remsi``) rather than Python's floored
modulo. This module registers ``(Integer, Integer)`` lowerings for
``%`` and ``//`` with sign-aware widening and floored semantics.

numba-cuda-mlir also lowers shared memory to zero-sized
internal-linkage globals: ``cuda.shared.array(0)`` becomes a private
zero-length ``memref.global`` and ``gpu.dynamic_shared_memory``'s
base becomes ``llvm.mlir.global internal @__dynamic_shmem__N :
!llvm.array<0 x i8>``. Indexing a zero-length internal object is
undefined behaviour, so optimizers (libnvvm -O3 and the LTO cubin
link at opt > 0) legally sink or delete stores staged through
dynamic shared memory. This module reroutes zero-length shared
arrays through a ``gpu.dynamic_shared_memory`` view and rewrites the
``__dynamic_shmem__`` globals to external linkage, which is the
"LTO store erasure" fix (upstream warning in #117 blames the
linker's fp16 handling; the actual defect is this lowering).

Import this module before compiling any kernel; registrations are
picked up when the MLIR target context refreshes its registries.
These are stop-gaps that belong upstream in numba-cuda-mlir; patch
branches exist in the ccam80/numba-cuda-mlir fork
(fix-boolean-bitwise-invert-lowering, fix-boolean-comparison-lowering,
fix-numpy-scalar-constants, fix-nested-tuple-dynamic-getitem,
fix-integer-mod-floordiv-lowering, fix-dynamic-shared-memory-ub).
Remove the corresponding shim once each lands upstream. The LTO
opt-disable below stays as the conservative default; with the
shared-memory shim in place the only known divergence at
NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=0 is FMA-level float variance in the
second-derivative summary metrics.
"""

import copy
import operator
from os import environ

import numpy

from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_lowering as _mlir_lowering
from numba_cuda_mlir import mlir_optimization as _mlir_optimization
from numba_cuda_mlir._mlir import ir as _ir
from numba_cuda_mlir._mlir.dialects import arith, memref as _memref
from numba_cuda_mlir._mlir.extras import types as _T
from numba_cuda_mlir.lowering import builtins as _lowering_builtins
from numba_cuda_mlir.lowering import cuda as _lowering_cuda
from numba_cuda_mlir.lowering import numpy as _lowering_numpy
from numba_cuda_mlir.lowering.numpy import registry as _np_registry
from numba_cuda_mlir.lowering.math import (
    eq_cg,
    ge_cg,
    gt_cg,
    le_cg,
    lt_cg,
    ne_cg,
    registry as _math_registry,
)
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import config as _cuda_config


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


_COMPARISON_CGS = {
    operator.eq: eq_cg,
    operator.ne: ne_cg,
    operator.lt: lt_cg,
    operator.le: le_cg,
    operator.gt: gt_cg,
    operator.ge: ge_cg,
}


def register_boolean_comparison_lowerings() -> None:
    """Register comparisons on Boolean operands.

    Upstream registers eq/ne/lt/le/gt/ge for ``(Number, Number)``
    only; Boolean operands raise ``NotImplementedError`` during
    lowering. Route the same code generators through the Boolean
    signatures, mirroring upstream's own Boolean registrations for
    ``sub`` and ``mul``.
    """

    signatures = (
        (types.Boolean, types.Boolean),
        (types.Boolean, types.Number),
        (types.Number, types.Boolean),
    )
    for op, cg in _COMPARISON_CGS.items():
        for lhs_type, rhs_type in signatures:
            _math_registry.lower(op, lhs_type, rhs_type)(cg)


register_boolean_comparison_lowerings()


_original_try_extract_constant = lowering_utilities.try_extract_constant


def _try_extract_constant_numpy(value):
    """Unwrap numpy scalar constants before constant extraction.

    numba freezes closure constants such as ``numpy.bool_(True)`` into
    kernels, but numba-cuda-mlir 0.4.0's ``try_extract_constant`` only
    matches Python ``int``/``float``/``bool`` and crashes constructing
    ``ir.Value`` from a numpy scalar. Convert via ``.item()`` first.
    """

    if isinstance(value, numpy.generic):
        value = value.item()
    return _original_try_extract_constant(value)


def register_numpy_constant_shim() -> None:
    """Patch try_extract_constant in every module that imported it."""

    for module in (
        lowering_utilities,
        _lowering_builtins,
        _lowering_cuda,
        _lowering_numpy,
    ):
        module.try_extract_constant = _try_extract_constant_numpy


register_numpy_constant_shim()


_original_lower_const_assign = _mlir_lowering.MLIRLower.lower_const_assign


def _lower_const_assign_numpy(self, target, const):
    """Convert numpy scalar constants before const-assign lowering.

    ``lower_const_assign`` gates on ``isinstance(value, (bool, int,
    float, np.number))``, but ``numpy.bool_`` is not ``np.number``,
    so frozen numpy bool constants fall through to
    ``NotImplementedError``. Normalise to Python scalars first.
    """

    if isinstance(const.value, numpy.generic):
        const = copy.copy(const)
        const.value = const.value.item()
    return _original_lower_const_assign(self, target, const)


_mlir_lowering.MLIRLower.lower_const_assign = _lower_const_assign_numpy


_original_load_var = _mlir_lowering.MLIRLower.load_var


def _load_var_numpy(self, var):
    """Unwrap numpy scalars read out of the lowering varmap.

    Frozen closure/global numpy scalars reach the varmap unconverted
    and crash downstream utilities (``unverified_convert``,
    ``try_extract_constant``) that only accept Python scalars or MLIR
    values. Convert at the single read chokepoint.
    """

    result = _original_load_var(self, var)
    if isinstance(result, numpy.generic):
        result = result.item()
    return result


_mlir_lowering.MLIRLower.load_var = _load_var_numpy


@lowering_utilities.unverified_convert.register(numpy.generic)
def _unverified_convert_numpy_scalar(value, target_type, *, signed=False):
    """Convert numpy scalar values via their Python equivalents.

    ``unverified_convert`` dispatches on the value type and has no
    overload for numpy scalars, which reach it through frozen closure
    constants in multi-stage algorithm loops.
    """

    return lowering_utilities.unverified_convert(
        value.item(), target_type, signed=signed
    )


_original_tuple_getitem = _lowering_numpy.lower_uni_tuple_getitem


def _lower_tuple_getitem_nested(builder, target, args, kwargs):
    """Support dynamic getitem on tuples whose elements are tuples.

    Upstream's tuple getitem emits a single-result
    ``scf.index_switch``, which requires a scalar MLIR result type;
    tuple-of-tuple elements (Butcher tableau rows) crash it. Decompose
    the row selection into one scalar switch per column and store the
    resulting Python tuple of values, matching the varmap convention
    that tuples are always Python tuples.
    """
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.lowering_utilities import convert, index_of
    from numba_cuda_mlir.mlir.dialect_exts import scf
    from numba_cuda_mlir.numba_cuda.core import ir as numba_ir

    target_type = builder.get_numba_type(target.name)
    if not isinstance(target_type, types.BaseTuple):
        return _original_tuple_getitem(builder, target, args, kwargs)

    tup = builder.load_var(args[0])
    index = (
        builder.load_var(args[1])
        if isinstance(args[1], numba_ir.Var)
        else args[1]
    )
    if not isinstance(index, ir.Value):
        builder.store_var(target, tup[int(index)])
        return

    tup = builder.lower_literal_if_needed(tup)
    index = index_of(index)
    element_types = (
        [target_type.dtype] * target_type.count
        if isinstance(target_type, types.UniTuple)
        else list(target_type.types)
    )
    cases = ir.DenseI64ArrayAttr.get(range(len(tup)))
    selected = []
    for column, element_type in enumerate(element_types):
        column_mlir_type = builder.get_mlir_type(element_type)

        def default(op, _column=column, _type=column_mlir_type):
            scf.yield_([convert(tup[0][_column], _type)])

        def case_builder(
            op, case_index, case_value, _column=column,
            _type=column_mlir_type,
        ):
            scf.yield_([convert(tup[case_value][_column], _type)])

        selected.append(
            scf.index_switch(
                results=[column_mlir_type],
                arg=index,
                cases=cases,
                default_body_builder=default,
                case_body_builder=case_builder,
            )
        )
    builder.store_var(target, tuple(selected))


def register_nested_tuple_getitem_lowering() -> None:
    """Register the nested-tuple getitem over upstream's version."""

    for container in (types.UniTuple, types.Tuple):
        _np_registry.lower(operator.getitem, container, types.Number)(
            _lower_tuple_getitem_nested
        )


register_nested_tuple_getitem_lowering()


def _int_mod_cg(builder, target, args, kwargs):
    """Lower integer ``%`` with Python floored-modulo semantics.

    Upstream's ``mod_cg`` widens operands through a conversion that
    zero-extends signless integers, so negative ``int32`` dividends
    become huge positive ``int64`` values, and it emits a bare
    ``arith.remsi`` (truncated remainder) where Python requires
    floored modulo. Sign-extend signed operands and add the divisor
    when the remainder's sign disagrees with it.
    """

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    signed = getattr(target_type, "signed", True)
    lhs, rhs = args
    lhs = lowering_utilities.convert(
        builder.load_var(lhs), target_mlir_type, signed=signed
    )
    rhs = lowering_utilities.convert(
        builder.load_var(rhs), target_mlir_type, signed=signed
    )
    if signed:
        rem = arith.remsi(lhs, rhs)
        zero = lowering_utilities.constant(0, rem.type)
        sign_mismatch = arith.cmpi(
            arith.CmpIPredicate.slt, arith.xori(rem, rhs), zero
        )
        rem_nonzero = arith.cmpi(arith.CmpIPredicate.ne, rem, zero)
        needs_fix = arith.andi(sign_mismatch, rem_nonzero)
        correction = arith.select(needs_fix, rhs, zero)
        result = arith.addi(rem, correction)
    else:
        result = arith.remui(lhs, rhs)
    builder.store_var(target, result)


def _int_floordiv_cg(builder, target, args, kwargs):
    """Lower integer ``//`` with sign-aware operand widening.

    Upstream's ``floordiv_cg`` emits the correct floored division
    (``arith.floordivsi``) but widens operands with zero-extension,
    corrupting any negative runtime operand. Sign-extend signed
    operands; use ``arith.divui`` for unsigned ones.
    """

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    signed = getattr(target_type, "signed", True)
    lhs, rhs = args
    lhs = lowering_utilities.convert(
        builder.load_var(lhs), target_mlir_type, signed=signed
    )
    rhs = lowering_utilities.convert(
        builder.load_var(rhs), target_mlir_type, signed=signed
    )
    if signed:
        result = arith.floordivsi(lhs, rhs)
    else:
        result = arith.divui(lhs, rhs)
    builder.store_var(target, result)


def register_integer_division_lowerings() -> None:
    """Register floored ``%`` and sign-safe ``//`` for integers."""

    for op in (operator.mod, operator.imod):
        _math_registry.lower(op, types.Integer, types.Integer)(
            _int_mod_cg
        )
    for op in (operator.floordiv, operator.ifloordiv):
        _math_registry.lower(op, types.Integer, types.Integer)(
            _int_floordiv_cg
        )


register_integer_division_lowerings()


_original_static_shared = _lowering_cuda.cuda_static_shared_memory


def _dynamic_region_shared_memory(lower, target, dtype):
    """Lower ``cuda.shared.array(0)`` to the dynamic shared region.

    A private zero-length shared ``memref.global`` is undefined
    behaviour to index, so optimizers sink or delete stores staged
    through it. A view over ``gpu.dynamic_shared_memory`` at byte
    offset zero, sized at runtime from the region's extent, matches
    numba's convention that every zero-length shared array aliases
    the dynamic region base. The view does not advance the running
    byte offset used by runtime-shaped shared arrays.
    """

    element_type = lower.get_storage_type(
        _lowering_cuda._resolve_numba_dtype(lower, dtype)
    )
    mr_type = _ir.MemRefType.get(
        shape=[_ir.ShapedType.get_dynamic_size()],
        element_type=element_type,
        memory_space=lower._get_shared_address_space(),
    )
    with lower.alloca_insertion_point():
        shm_base = lower._get_shared_memory_base()
        zero = arith.constant(result=_T.index(), value=0)
        element_bytes = arith.constant(
            result=_T.index(), value=element_type.width // 8
        )
        num_elements = arith.divui(
            _memref.dim(shm_base, zero), element_bytes
        )
        view = _memref.view(
            result=mr_type,
            source=shm_base,
            byte_shift=zero,
            sizes=[num_elements],
        )
    lower.store_var(target, view)


def _static_shared_memory_shim(lower, target, static_shape, dtype, alignas):
    """Route zero-length 1-D shared arrays to the dynamic region."""

    if len(static_shape) == 1 and static_shape[0] == 0:
        return _dynamic_region_shared_memory(lower, target, dtype)
    return _original_static_shared(
        lower, target, static_shape, dtype, alignas
    )


def _request_shared_memory_shim(self, sizes, mr_type):
    """Emit runtime-shaped shared views at the current insertion point.

    Upstream inserts at the end of the entry block, which raises an
    insertion error once the block has a terminator (any shared
    request lowered after control flow) and cannot see size operands
    computed after a branch.
    """

    match mr_type.element_type:
        case _ir.IntegerType() | _ir.FloatType() as t:
            element_bytes = t.width // 8
        case _T.index:
            element_bytes = 8
        case _:
            raise NotImplementedError(
                f"NotImplemented shared memory type {mr_type}."
            )
    assert self.mlir_funcOp
    bytes_op = arith.constant(result=_T.index(), value=element_bytes)
    for size in sizes:
        size = self.mlir_convert(size, _T.index())
        bytes_op = arith.muli(lhs=bytes_op, rhs=size)
    shm_base = self._get_shared_memory_base()
    if self._total_shared_memory_bytes is None:
        self._total_shared_memory_bytes = arith.constant(
            result=_T.index(), value=0
        )
    view = _memref.view(
        result=mr_type,
        source=shm_base,
        byte_shift=self._total_shared_memory_bytes,
        sizes=sizes,
    )
    self._total_shared_memory_bytes = arith.addi(
        lhs=self._total_shared_memory_bytes, rhs=bytes_op
    )
    return view


def _make_dynamic_shared_memory_external(module):
    """Rewrite ``__dynamic_shmem__*`` globals to external linkage.

    With internal linkage the optimizer may assume the zero-length
    object really is zero bytes long, making every indexed access
    out of bounds; external linkage makes the size unknown and
    restores conservative aliasing, matching CUDA C's
    ``extern __shared__`` declaration.
    """

    external = _ir.Attribute.parse("#llvm.linkage<external>")

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    if child.operation.name == "llvm.mlir.global":
                        sym = str(child.attributes["sym_name"])
                        if "__dynamic_shmem__" in sym:
                            child.attributes["linkage"] = external
                    walk(child.operation)

    walk(module.operation)


_original_pre_codegen = _mlir_optimization.run_pre_codegen_patterns


def _pre_codegen_with_external_shmem(module, *args, **kwargs):
    result = _original_pre_codegen(module, *args, **kwargs)
    _make_dynamic_shared_memory_external(module)
    return result


def register_dynamic_shared_memory_shims() -> None:
    """Install the dynamic-shared-memory UB fixes."""

    _lowering_cuda.cuda_static_shared_memory = _static_shared_memory_shim
    _mlir_lowering.MLIRLower._request_shared_memory = (
        _request_shared_memory_shim
    )
    _mlir_optimization.run_pre_codegen_patterns = (
        _pre_codegen_with_external_shmem
    )


register_dynamic_shared_memory_shims()


# The dynamic-shared-memory shims above fix the store erasure that
# made LTO-link optimization unsafe. Opt stays disabled by default
# anyway as the conservative choice: at opt>0 the second-derivative
# summary metrics show FMA-level float variance against the CPU
# reference (their central difference is cancellation-prone). An
# explicit NUMBA_CUDA_MLIR_DISABLE_LTO_OPT in the environment wins,
# so LTO-link optimization can be enabled by setting it to 0.
if "NUMBA_CUDA_MLIR_DISABLE_LTO_OPT" not in environ:
    _cuda_config.CUDA_DISABLE_LTO_OPT = 1
