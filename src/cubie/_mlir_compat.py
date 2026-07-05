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

numba-cuda-mlir also materializes frozen numpy closure arrays with a
per-thread device ``malloc`` and no ``free`` (element-wise
``tensor.from_elements`` bufferized to a heap allocation). The 8 MB
default device heap is exhausted around 64k threads, so any kernel
capturing an array literal — every cubie save path freezes its
saved-index arrays — faulted with ``CUDA_ERROR_ILLEGAL_ADDRESS`` at
production batch sizes. This module reroutes array literals to
internal constant globals carrying the array's raw bytes.

Import this module before compiling any kernel; registrations are
picked up when the MLIR target context refreshes its registries.
These are stop-gaps that belong upstream in numba-cuda-mlir; patch
branches exist in the ccam80/numba-cuda-mlir fork
(fix-boolean-bitwise-invert-lowering, fix-boolean-comparison-lowering,
fix-numpy-scalar-constants, fix-nested-tuple-dynamic-getitem,
fix-integer-mod-floordiv-lowering, fix-dynamic-shared-memory-ub,
fix-frozen-array-device-malloc).
Remove the corresponding shim once each lands upstream. With the
shared-memory shim in place LTO-link optimization is safe and runs
at the upstream default (enabled); set
NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=1 to force opt_level=0 on the LTO
link. The only known divergence with optimization enabled is
FMA-level float variance in the second-derivative summary metrics.
"""

import copy
import operator

import numpy

from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_lowering as _mlir_lowering
from numba_cuda_mlir import mlir_optimization as _mlir_optimization
from numba_cuda_mlir._mlir import ir as _ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    builtin as _builtin,
    llvm as _llvm,
    memref as _memref,
)
from numba_cuda_mlir._mlir.extras import types as _T
from numba_cuda_mlir.mlir.dialect_exts import llvm as _llvm_ext
from numba_cuda_mlir.lowering_utilities.type_conversions import (
    to_numba_type as _to_numba_type,
)
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


_original_lower_array_literal = (
    _mlir_lowering.MLIRLower.lower_array_literal
)
_array_literal_count = 0


def _lower_array_literal_shim(self, value):
    """Lower frozen array literals as internal constant globals.

    The upstream lowering materializes each frozen numpy closure
    array with a per-thread device ``malloc`` and no ``free``,
    exhausting the 8 MB default device heap around 64k threads. The
    array's raw bytes are emitted instead as an internal constant
    ``llvm.mlir.global`` (the encoding string constants use, since
    dense-attribute initializers are dropped by the LLVM-7
    translation backend) wrapped in a memref descriptor with the
    same strided type the upstream lowering produced. Dtypes whose
    storage width differs from the numpy itemsize, and rank-0
    arrays, keep the upstream lowering.
    """

    global _array_literal_count
    dtype_numba = _to_numba_type(value.dtype)
    dtype = self.get_storage_type(dtype_numba)
    raw_byte_storage = (
        isinstance(dtype, (_ir.IntegerType, _ir.FloatType))
        and dtype.width == value.dtype.itemsize * 8
        and value.ndim > 0
    )
    if not raw_byte_storage:
        return _original_lower_array_literal(self, value)

    contiguous = numpy.ascontiguousarray(value)
    databytes = contiguous.tobytes()
    name = f"__cubie_array_literal_{_array_literal_count}"
    _array_literal_count += 1
    array_type = _ir.Type.parse(f"!llvm.array<{len(databytes)} x i8>")
    gpu_block = self.mlir_gpu_module.bodyRegion.blocks[0]
    with _ir.InsertionPoint.at_block_begin(gpu_block):
        _llvm.GlobalOp(
            array_type,
            name,
            _ir.Attribute.parse("#llvm.linkage<internal>"),
            addr_space=0,
            constant=True,
            value=_ir.StringAttr.get(databytes),
            alignment=value.dtype.itemsize,
        )

    ndim = contiguous.ndim
    strides, running = [], 1
    for dim in reversed(contiguous.shape):
        strides.insert(0, running)
        running *= dim
    dynamic = _ir.MemRefType.get_dynamic_stride_or_offset()
    layout = _ir.StridedLayoutAttr.get(
        offset=dynamic, strides=[dynamic] * ndim
    )
    memref_type = _ir.MemRefType.get(
        shape=contiguous.shape, element_type=dtype, layout=layout
    )
    struct_type = _ir.Type.parse(
        f"!llvm.struct<(ptr, ptr, i64, array<{ndim} x i64>,"
        f" array<{ndim} x i64>)>"
    )

    def insert(container, element, *position):
        return _llvm.insertvalue(
            container=container,
            value=element,
            position=_ir.DenseI64ArrayAttr.get(list(position)),
        )

    with self.alloca_insertion_point():
        pointer = _llvm_ext.addressof(name)
        descriptor = _llvm.UndefOp(struct_type).result
        descriptor = insert(descriptor, pointer, 0)
        descriptor = insert(descriptor, pointer, 1)
        descriptor = insert(descriptor, arith.constant(_T.i64(), 0), 2)
        for axis, dim in enumerate(contiguous.shape):
            descriptor = insert(
                descriptor, arith.constant(_T.i64(), dim), 3, axis
            )
        for axis, stride in enumerate(strides):
            descriptor = insert(
                descriptor, arith.constant(_T.i64(), stride), 4, axis
            )
        return _builtin.unrealized_conversion_cast(
            [memref_type], [descriptor]
        )


def register_array_literal_shim() -> None:
    """Install the constant-global array-literal lowering."""

    _mlir_lowering.MLIRLower.lower_array_literal = (
        _lower_array_literal_shim
    )


register_array_literal_shim()


# The dynamic-shared-memory shims above fix the store erasure that
# previously made LTO-link optimization unsafe, so it runs at the
# upstream default (enabled). Set NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=1
# to force opt_level=0 on the LTO link; the remaining difference at
# opt>0 is FMA-level float variance in the second-derivative summary
# metrics (their central difference is cancellation-prone).


# ------------------------------------------------------------------ #
# Compile-time performance patches (numba_cuda frontend)             #
# ------------------------------------------------------------------ #
# The shims below rebind the compiler-frontend performance changes
# carried on the cubie_patch branch of the ccam80/numba-cuda-mlir
# fork so they apply to the stock wheel: lazy PostProcessor liveness,
# string-only error markup, the per-class TargetConfig hash, SSA
# sweeps restricted to def/use blocks, memoised callee IR with a
# structural clone (including the preserve_ir form of inline_ir),
# CallConstraint re-resolution skipping, and bitset liveness fix
# points. All are behaviour-preserving; only compile time changes.
# Each group feature-detects the installed package and no-ops when
# the change is already present (a patched build, or a future release
# that merged it). Fork feature branches: perf-lazy-postproc-liveness,
# perf-lazy-error-markup, perf-targetconfig-hash-cache,
# perf-ssa-restricted-sweeps, perf-inline-callee-ir-cache (stacked on
# inline-ir-preserve), perf-callconstraint-memo, perf-liveness-bitsets.
# The numba-cuda lowering-side patches (call-type cache, linear
# singly-assigned scan) have no analogue here: MLIRBackend replaces
# the LLVM lowering entirely. The NumbaError double-highlight fix is
# also inapplicable: the vendored NumbaError inherits Exception
# directly, so no base class re-highlights the message.

import inspect
import itertools
import weakref
from collections import defaultdict

from numba_cuda_mlir.numba_cuda import types as _nb_types
from numba_cuda_mlir.numba_cuda.core import (
    analysis as _nb_analysis,
    errors as _nb_errors,
    inline_closurecall as _nb_icc,
    ir as _nb_ir,
    ir_utils as _nb_ir_utils,
    postproc as _nb_postproc,
    ssa as _nb_ssa,
    targetconfig as _nb_targetconfig,
    transforms as _nb_transforms,
    typeinfer as _nb_typeinfer,
)


_BYTE_BITS = tuple(
    tuple(bit for bit in range(8) if value & (1 << bit))
    for value in range(256)
)


def _compute_live_map(cfg, blocks, var_use_map, var_def_map):
    """
    Find variables that must be alive at the ENTRY of each block.

    The two fix points (forward definition reach, backward liveness)
    run on bitsets: every variable gets a bit index and per-block sets
    become arbitrary-size integers, so the union/intersection work in
    each sweep is machine-word bignum arithmetic instead of hash-set
    element traversal. Large flattened functions have tens of
    thousands of variables live across thousands of blocks, where set
    objects made this analysis dominate compilation.
    """
    index = {}
    names = []
    for use_def_map in (var_def_map, var_use_map):
        for name_set in use_def_map.values():
            for name in name_set:
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
    nbytes = (len(names) + 7) // 8

    def to_bits(name_set):
        buf = bytearray(nbytes)
        for name in name_set:
            i = index[name]
            buf[i >> 3] |= 1 << (i & 7)
        return int.from_bytes(buf, "little")

    offsets = list(blocks.keys())
    def_bits = {offset: to_bits(var_def_map[offset]) for offset in offsets}
    use_bits = {offset: to_bits(var_use_map[offset]) for offset in offsets}

    successors = {
        offset: [out_blk for out_blk, _ in cfg.successors(offset)]
        for offset in offsets
    }
    predecessors = {
        offset: [inc_blk for inc_blk, _ in cfg.predecessors(offset)]
        for offset in offsets
    }

    # Forward: definitions (and uses) of every block that can reach a
    # block, itself included. Ascending label order approximates a
    # topological order, so this converges in a couple of sweeps.
    def_reach_map = {
        offset: def_bits[offset] | use_bits[offset] for offset in offsets
    }
    changed = True
    while changed:
        changed = False
        for offset in offsets:
            cur = def_reach_map[offset]
            for out_blk in successors[offset]:
                merged = def_reach_map[out_blk] | cur
                if merged != def_reach_map[out_blk]:
                    def_reach_map[out_blk] = merged
                    changed = True

    # Backward: push variable usage to predecessors, restricted to
    # variables a definition can reach and not defined in the
    # predecessor itself. Reverse label order approximates a reverse
    # topological order for the same fast convergence.
    live_bits = {offset: use_bits[offset] for offset in offsets}
    changed = True
    while changed:
        changed = False
        for offset in reversed(offsets):
            live_vars = live_bits[offset]
            for inc_blk in predecessors[offset]:
                incoming = (
                    live_vars & def_reach_map[inc_blk]
                ) & ~def_bits[inc_blk]
                merged = live_bits[inc_blk] | incoming
                if merged != live_bits[inc_blk]:
                    live_bits[inc_blk] = merged
                    changed = True

    live_map = {}
    for offset in offsets:
        blob = live_bits[offset].to_bytes(nbytes, "little")
        live = set()
        for byte_pos, byte in enumerate(blob):
            if byte:
                base = byte_pos << 3
                for bit in _BYTE_BITS[byte]:
                    live.add(names[base + bit])
        live_map[offset] = live
    return live_map


def _patch_live_map():
    if hasattr(_nb_analysis, "_BYTE_BITS"):
        return
    stock = _nb_analysis.compute_live_map
    _nb_analysis._BYTE_BITS = _BYTE_BITS
    _nb_analysis.compute_live_map = _compute_live_map
    # ir_utils imports the function by name at module import time.
    if getattr(_nb_ir_utils, "compute_live_map", None) is stock:
        _nb_ir_utils.compute_live_map = _compute_live_map


def _patch_postproc():
    src = inspect.getsource(_nb_postproc.PostProcessor.run)
    if "Only generator info consumes" in src:
        return  # already lazy

    def run(self, emit_dels: bool = False, extend_lifetimes: bool = False):
        """
        Run the following passes over Numba IR:
        - canonicalize the CFG
        - emit explicit `del` instructions for variables
        - compute lifetime of variables
        - compute generator info (if function is a generator function)
        """
        self.func_ir.blocks = _nb_transforms.canonicalize_cfg(
            self.func_ir.blocks
        )
        vlt = _nb_postproc.VariableLifetime(self.func_ir.blocks)
        self.func_ir.variable_lifetime = vlt

        if self.func_ir.is_generator:
            # Only generator info consumes the entry-liveness result
            # (via get_block_entry_vars); non-generator consumers of
            # liveness use the lazily computed properties on
            # VariableLifetime instead, so the fix-point analyses are
            # not run eagerly for them.
            bev = _nb_analysis.compute_live_variables(
                vlt.cfg,
                self.func_ir.blocks,
                vlt.usedefs.defmap,
                vlt.deadmaps.combined,
            )
            for offset, ir_block in self.func_ir.blocks.items():
                self.func_ir.block_entry_vars[ir_block] = bev[offset]

            self.func_ir.generator_info = _nb_postproc.GeneratorInfo()
            self._compute_generator_info()
        else:
            self.func_ir.generator_info = None

        # Emit del nodes, do this last as the generator info parsing
        # generates and then strips dels as part of its analysis.
        if emit_dels:
            self._insert_var_dels(extend_lifetimes=extend_lifetimes)

    _nb_postproc.PostProcessor.run = run


def _patch_error_markup():
    scheme_cls = getattr(_nb_errors, "HighlightColorScheme", None)
    if scheme_cls is None or "ColorShell" not in inspect.getsource(
        scheme_cls._markup
    ):
        return
    from colorama import Style

    def _markup(self, msg, color=None, style=Style.BRIGHT):
        # This only builds a string; it does not write to a stream.
        # Wrapping the standard streams with colorama (ColorShell) is
        # unnecessary for that and was undone before anything printed
        # the string, yet its init/deinit dominated error
        # construction when typing speculatively instantiates many
        # exceptions. Emit the same bytes without touching the
        # streams.
        features = ""
        if color:
            features += color
        if style:
            features += style
        return features + msg + Style.RESET_ALL

    scheme_cls._markup = _markup


def _patch_targetconfig_hash():
    cfg_cls = _nb_targetconfig.TargetConfig
    if hasattr(cfg_cls, "_precomputed_hash"):
        return

    def _set_hash(cls):
        cls._precomputed_hash = hash(tuple(sorted(cls.options)))
        for sub in cls.__subclasses__():
            _set_hash(sub)

    _set_hash(cfg_cls)

    meta = type(cfg_cls)
    orig_meta_init = meta.__init__

    def meta_init(cls, name, bases, dct):
        orig_meta_init(cls, name, bases, dct)
        cls._precomputed_hash = hash(tuple(sorted(cls.options)))

    meta.__init__ = meta_init

    def __hash__(self):
        # Equal to hash(tuple(sorted(self.values()))): sorting the
        # values() mapping iterates option names only, so the hash is
        # the per-class constant precomputed above.
        return self._precomputed_hash

    cfg_cls.__hash__ = __hash__


def _ssa_find_defs_violators(blocks, cfg):
    """
    Returns
    -------
    res : Tuple[Dict[str, None], Mapping, Mapping]
        The SSA violators in a dictionary of variable names, the
        per-variable definition map (name -> [(assign, label)]) and
        the per-variable use-block map (name -> {label}).
    """
    defs = defaultdict(list)
    uses = defaultdict(set)
    states = dict(defs=defs, uses=uses)
    _nb_ssa._run_block_analysis(blocks, states, _nb_ssa._GatherDefsHandler())
    violators = {k: None for k, vs in defs.items() if len(vs) > 1}
    doms = cfg.dominators()
    for k, use_blocks in uses.items():
        if k not in violators:
            for label in use_blocks:
                dom = doms[label]
                def_labels = {label for _assign, label in defs[k]}
                if not def_labels.intersection(dom):
                    violators[k] = None
                    break
    return violators, defs, uses


def _ssa_run_block_rewrite(blocks, states, handler, relevant_labels=None):
    newblocks = {}
    for label, blk in blocks.items():
        if relevant_labels is not None and label not in relevant_labels:
            # The handler can only change statements that mention the
            # variable being processed, so blocks without a def/use
            # of it pass through unchanged.
            newblocks[label] = blk
            continue
        newblk = _nb_ir.Block(scope=blk.scope, loc=blk.loc)
        newbody = []
        states["label"] = label
        states["block"] = blk
        for stmt in _nb_ssa._run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks


def _ssa_fresh_vars(blocks, varname, def_labels):
    """Rewrite to put fresh variable names"""
    states = _nb_ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap = defaultdict(list)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _nb_ssa._FreshVarHandler(), def_labels
    )
    return newblocks, defmap


def _ssa_fix_ssa_vars(
    blocks, varname, defmap, cfg, df_plus, cache_list_vars, use_labels
):
    """Rewrite all uses to ``varname`` given the definition map"""
    states = _nb_ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap
    states["phimap"] = phimap = defaultdict(list)
    states["cfg"] = cfg
    states["phi_locations"] = _nb_ssa._compute_phi_locations(df_plus, defmap)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _nb_ssa._FixSSAVars(cache_list_vars), use_labels
    )
    # insert phi nodes
    for label, philist in phimap.items():
        curblk = newblocks[label]
        # Prepend PHI nodes to the block. Build a fresh block rather
        # than mutating in place: phi locations include pass-through
        # blocks, and input block objects must never be mutated.
        newblk = _nb_ir.Block(scope=curblk.scope, loc=curblk.loc)
        newblk.body = philist + curblk.body
        newblocks[label] = newblk
    return newblocks


def _ssa_run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function."""
    if not blocks:
        return {}
    cfg = _nb_ssa.compute_cfg_from_blocks(blocks)
    df_plus = _nb_ssa._iterated_domfronts(cfg)
    violators, defs, uses = _ssa_find_defs_violators(blocks, cfg)
    cache_list_vars = _nb_ssa._CacheListVars()

    for varname in violators:
        # Only blocks that define or use the variable can be changed
        # by its rewrite passes; every other block passes through
        # untouched. The def/use block sets collected up front stay
        # valid throughout: the passes rename assignment targets and
        # uses of the current variable only, and phi nodes introduce
        # only freshly versioned names. The uses map excludes a
        # variable's use on the RHS of an assignment to itself
        # (e.g. ``x = x + 1``), but such a use can only appear in a
        # statement that assigns the variable, so its block is always
        # a def block; the fix pass therefore visits the union.
        def_labels = {label for _assign, label in defs[varname]}
        use_labels = uses[varname] | def_labels
        blocks, defmap = _ssa_fresh_vars(blocks, varname, def_labels)
        blocks = _ssa_fix_ssa_vars(
            blocks,
            varname,
            defmap,
            cfg,
            df_plus,
            cache_list_vars,
            use_labels,
        )

    cfg_post = _nb_ssa.compute_cfg_from_blocks(blocks)
    if cfg_post != cfg:
        raise _nb_errors.CompilerError("CFG mutated in SSA pass")
    return blocks


def _patch_ssa():
    params = inspect.signature(_nb_ssa._fresh_vars).parameters
    if "def_labels" in params:
        return
    _nb_ssa._find_defs_violators = _ssa_find_defs_violators
    _nb_ssa._run_block_rewrite = _ssa_run_block_rewrite
    _nb_ssa._fresh_vars = _ssa_fresh_vars
    _nb_ssa._fix_ssa_vars = _ssa_fix_ssa_vars
    _nb_ssa._run_ssa = _ssa_run_ssa


_callee_ir_cache = weakref.WeakKeyDictionary()


def _clone_callee_ir(func_ir):
    """Structural clone of ``func_ir`` for use as an inline callee.

    Equivalent in effect to deep-copying the IR blocks, but far
    cheaper: a fresh single Scope is created (with its redefinition
    state), every Var is recreated in it, and every statement,
    expression and mutable container is rebuilt. Immutable leaves are
    shared: Loc objects, constant/global/freevar payloads, and any
    non-IR values held in expressions. The clone can be freely
    relabelled, renamed and spliced by ``inline_ir`` without mutating
    the source IR.
    """
    blocks = func_ir.blocks
    old_scope = next(iter(blocks.values())).scope
    new_scope = _nb_ir.Scope(parent=old_scope.parent, loc=old_scope.loc)
    new_scope.redefined.update(old_scope.redefined)
    for name, versions in old_scope.var_redefinitions.items():
        new_scope.var_redefinitions[name] = set(versions)

    varmap = {}
    for name, var in old_scope.localvars._con.items():
        varmap[name] = new_scope.define(name, var.loc)

    def clone_value(value):
        if isinstance(value, _nb_ir.Var):
            new_var = varmap.get(value.name)
            if new_var is None:
                new_var = new_scope.define(value.name, value.loc)
                varmap[value.name] = new_var
            return new_var
        if isinstance(value, _nb_ir.Expr):
            new_expr = copy.copy(value)
            new_expr._kws = {
                key: clone_value(item) for key, item in value._kws.items()
            }
            return new_expr
        if isinstance(value, list):
            return [clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_value(item) for item in value)
        if isinstance(value, dict):
            return {key: clone_value(item) for key, item in value.items()}
        return value

    def clone_stmt(stmt):
        new_stmt = copy.copy(stmt)
        for name, value in tuple(new_stmt.__dict__.items()):
            cloned = clone_value(value)
            if cloned is not value:
                new_stmt.__dict__[name] = cloned
        return new_stmt

    new_blocks = {}
    for label, block in blocks.items():
        new_block = _nb_ir.Block(scope=new_scope, loc=block.loc)
        new_block.body = [clone_stmt(stmt) for stmt in block.body]
        new_blocks[label] = new_block

    new_ir = copy.copy(func_ir)
    new_ir.blocks = new_blocks
    new_ir.block_entry_vars = {}
    return new_ir


def _make_inline_ir():
    def inline_ir(
        self, caller_ir, block, i, callee_ir, callee_freevars,
        arg_typs=None, preserve_ir=True,
    ):
        """Inlines the callee_ir in the caller_ir at statement index i
        of block `block`, callee_freevars are the free variables for
        the callee_ir. If the callee_ir is derived from a function
        `func` then this is `func.__code__.co_freevars`. If `arg_typs`
        is given and the InlineWorker instance was initialized with a
        typemap and calltypes then they will be appropriately updated
        based on the arg_typs. If `preserve_ir` is True, the callee_ir
        object will be copied before mutating, otherwise it will be
        mutated in place.
        """
        # Save a reference to the incoming callee_ir
        callee_ir_original = callee_ir

        if preserve_ir:
            def copy_ir(the_ir):
                kernel_copy = the_ir.copy()
                kernel_copy.blocks = {}
                for block_label, block in the_ir.blocks.items():
                    new_block = copy.deepcopy(the_ir.blocks[block_label])
                    kernel_copy.blocks[block_label] = new_block
                return kernel_copy

            callee_ir = copy_ir(callee_ir)

        if self.validator is not None:
            self.validator(callee_ir)

        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks

        # 1. relabel callee_ir by adding an offset
        max_label = max(
            _nb_ir_utils._the_max_label.next(),
            max(caller_ir.blocks.keys()),
        )
        callee_blocks = _nb_icc.add_offset_to_labels(
            callee_blocks, max_label + 1
        )
        callee_blocks = _nb_icc.simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        _nb_ir_utils._the_max_label.update(max_label)
        self.debug_print("After relabel")
        _nb_icc._debug_dump(callee_ir)

        # 2. rename all local variables in callee_ir with new locals
        # created in caller_ir
        callee_scopes = _nb_icc._get_all_scopes(callee_blocks)
        self.debug_print("callee_scopes = ", callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if var.name not in callee_freevars:
                inlined_name = _nb_icc._created_inlined_var_name(
                    callee_ir.func_id.unique_name, var.name
                )
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print("var_dict = ", var_dict)
        _nb_icc.replace_vars(callee_blocks, var_dict)
        self.debug_print("After local var rename")
        _nb_icc._debug_dump(callee_ir)

        # 3. replace formal parameters with actual arguments
        callee_func = callee_ir.func_id.func
        args = _nb_icc._get_callee_args(
            call_expr, callee_func, block.body[i].loc, caller_ir
        )

        # 4. Update typemap
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError("arg_typs should have a value not None")
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks

        self.debug_print("After arguments rename: ")
        _nb_icc._debug_dump(callee_ir)

        _nb_icc._replace_args_with(callee_blocks, args)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = _nb_ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1 :]
        new_label = _nb_icc.next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(_nb_ir.Jump(min_label, instr.loc))

        # 6. replace Return with assignment to LHS
        topo_order = _nb_icc.find_topo_order(callee_blocks)
        _nb_icc._replace_returns(callee_blocks, instr.target, new_label)

        if (
            instr.target.name in caller_ir._definitions
            and call_expr in caller_ir._definitions[instr.target.name]
        ):
            caller_ir._definitions[instr.target.name].remove(call_expr)

        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            block = callee_blocks[label]
            block.scope = scope
            _nb_icc._add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print("After merge in")
        _nb_icc._debug_dump(caller_ir)

        return callee_ir_original, callee_blocks, var_dict, new_blocks

    return inline_ir


def _patch_inline_worker():
    if hasattr(_nb_icc, "_clone_callee_ir"):
        return
    _nb_icc._clone_callee_ir = _clone_callee_ir
    _nb_icc._callee_ir_cache = _callee_ir_cache

    worker = _nb_icc.InlineWorker
    if "preserve_ir" not in inspect.signature(worker.inline_ir).parameters:
        worker.inline_ir = _make_inline_ir()

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        """Inlines the function in the caller_ir at statement index i
        of block `block`. If `arg_typs` is given and the InlineWorker
        instance was initialized with a typemap and calltypes then
        they will be appropriately updated based on the arg_typs.
        """
        callee_ir = self._fresh_callee_ir(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(
            caller_ir, block, i, callee_ir, freevars,
            arg_typs=arg_typs, preserve_ir=False,
        )

    def _fresh_callee_ir(self, function, enable_ssa=False):
        """Return callee IR that is safe for ``inline_ir`` to mutate.

        The canonical IR produced by the untyped pipeline for a given
        function and flags configuration is cached, and each call
        site receives a structural clone of it. Running the untyped
        pipeline is far more expensive than cloning, and deeply
        nested inline='always' functions otherwise recompile their
        whole subtree at every transitive call site.
        """
        try:
            per_func = _callee_ir_cache.setdefault(function, {})
        except TypeError:
            # Function is not weak-referenceable; skip caching.
            return self.run_untyped_passes(function, enable_ssa)
        key = (str(self.flags), enable_ssa)
        canonical_ir = per_func.get(key)
        if canonical_ir is None:
            canonical_ir = self.run_untyped_passes(function, enable_ssa)
            per_func[key] = canonical_ir
        return _clone_callee_ir(canonical_ir)

    worker.inline_function = inline_function
    worker._fresh_callee_ir = _fresh_callee_ir


def _patch_callconstraint():
    constraint = _nb_typeinfer.CallConstraint
    if "_resolved_key" in inspect.getsource(constraint.resolve):
        return

    orig_init = constraint.__init__

    def __init__(self, target, func, args, kws, vararg, loc):
        orig_init(self, target, func, args, kws, vararg, loc)
        # Input types of the last successful resolution, when that
        # resolution is provably repeatable (see resolve). The
        # propagation fix-point re-executes every constraint each
        # round; when the inputs are unchanged the resolution result
        # is identical, so the (expensive) template matching can be
        # skipped.
        self._resolved_key = None

    def resolve(self, typeinfer, typevars, fnty):
        assert fnty
        context = typeinfer.context

        r = _nb_typeinfer.fold_arg_vars(
            typevars, self.args, self.vararg, self.kws
        )
        if r is None:
            # Cannot resolve call type until all argument types are
            # known
            return
        pos_args, kw_args = r

        # Check argument to be precise
        for a in itertools.chain(pos_args, kw_args.values()):
            # Forbids imprecise type except array of undefined dtype
            if not a.is_precise() and not isinstance(a, _nb_types.Array):
                return

        # Resolve call type
        if isinstance(fnty, _nb_types.TypeRef):
            # Unwrap TypeRef
            fnty = fnty.instance_type

        resolve_key = (fnty, pos_args, tuple(sorted(kw_args.items())))
        if resolve_key == self._resolved_key:
            # Same inputs as the last successful resolution and that
            # resolution was repeatable: the signature, the target
            # type addition and the refinement bookkeeping would all
            # be identical, so there is nothing new to compute.
            return

        try:
            sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
        except _nb_typeinfer.ForceLiteralArg as e:
            # Adjust for bound methods
            folding_args = (
                (fnty.this,) + tuple(self.args)
                if isinstance(fnty, _nb_types.BoundFunction)
                else self.args
            )
            folded = e.fold_arguments(folding_args, self.kws)
            requested = set()
            unsatisfied = set()
            for idx in e.requested_args:
                maybe_arg = typeinfer.func_ir.get_definition(folded[idx])
                if isinstance(maybe_arg, _nb_ir.Arg):
                    requested.add(maybe_arg.index)
                else:
                    unsatisfied.add(idx)
            if unsatisfied:
                raise _nb_typeinfer.TypingError(
                    "Cannot request literal type.", loc=self.loc
                )
            elif requested:
                raise _nb_typeinfer.ForceLiteralArg(requested, loc=self.loc)
        if sig is None:
            # Note: duplicated error checking.
            #       See types.BaseFunction.get_call_type
            # Arguments are invalid => explain why
            headtemp = "Invalid use of {0} with parameters ({1})"
            args = [str(a) for a in pos_args]
            args += ["%s=%s" % (k, v) for k, v in sorted(kw_args.items())]
            head = headtemp.format(fnty, ", ".join(map(str, args)))
            desc = context.explain_function_type(fnty)
            msg = "\n".join([head, desc])
            raise _nb_typeinfer.TypingError(msg)

        typeinfer.add_type(self.target, sig.return_type, loc=self.loc)

        # If the function is a bound function and its receiver type
        # was refined, propagate it.
        if (
            isinstance(fnty, _nb_types.BoundFunction)
            and sig.recvr is not None
            and sig.recvr != fnty.this
        ):
            refined_this = context.unify_pairs(sig.recvr, fnty.this)
            if (
                refined_this is None
                and fnty.this.is_precise()
                and sig.recvr.is_precise()
            ):
                msg = "Cannot refine type {} to {}".format(
                    sig.recvr,
                    fnty.this,
                )
                raise _nb_typeinfer.TypingError(msg, loc=self.loc)
            if refined_this is not None and refined_this.is_precise():
                refined_fnty = fnty.copy(this=refined_this)
                typeinfer.propagate_refined_type(self.func, refined_fnty)

        # If the return type is imprecise but can be unified with the
        # target variable's inferred type, use the latter.
        # Useful for code such as::
        #    s = set()
        #    s.add(1)
        # (the set() call must be typed as int64(), not undefined())
        if not sig.return_type.is_precise():
            target = typevars[self.target]
            if target.defined:
                targetty = target.getone()
                if (
                    context.unify_pairs(targetty, sig.return_type)
                    == targetty
                ):
                    sig = sig.replace(return_type=targetty)

        self.signature = sig
        self._add_refine_map(typeinfer, typevars, sig)

        # Mark this resolution as repeatable only when re-running it
        # with identical inputs could not produce a different
        # outcome: a template-based BaseFunction resolves purely from
        # its registered templates (unlike e.g. a Dispatcher, whose
        # resolution consults the callee's still-refining inference
        # state during recursion), a precise return type skips the
        # target-unification branch, a non-BoundFunction skips
        # receiver refinement, and absence from the refine map means
        # no later refinement will call back into this constraint.
        if (
            isinstance(fnty, _nb_types.BaseFunction)
            and not isinstance(fnty, _nb_types.BoundFunction)
            and sig.return_type.is_precise()
            and typeinfer.refine_map.get(self.target) is not self
        ):
            self._resolved_key = resolve_key
        else:
            self._resolved_key = None

    constraint.__init__ = __init__
    constraint.resolve = resolve


_PERF_PATCH_GROUPS = {
    "liveness": _patch_live_map,
    "postproc": _patch_postproc,
    "errors": _patch_error_markup,
    "targetconfig": _patch_targetconfig_hash,
    "ssa": _patch_ssa,
    "inline": _patch_inline_worker,
    "callconstraint": _patch_callconstraint,
}


def apply_compiler_perf_patches() -> None:
    """Apply all frontend perf patch groups the installed wheel needs.

    Set CUBIE_DISABLE_NUMBA_PERF_PATCHES=1 to skip every group, for
    A/B benchmarking and for isolating suspected patch regressions.
    Set CUBIE_NUMBA_PERF_PATCH_GROUPS to a comma-separated subset of
    liveness, postproc, errors, targetconfig, ssa, inline,
    callconstraint to apply only those groups (per-feature A/B).
    """
    import os

    if os.environ.get("CUBIE_DISABLE_NUMBA_PERF_PATCHES", "0") == "1":
        return
    selected = os.environ.get("CUBIE_NUMBA_PERF_PATCH_GROUPS", "all")
    if selected.strip().lower() == "all":
        names = list(_PERF_PATCH_GROUPS)
    else:
        names = [n.strip() for n in selected.split(",") if n.strip()]
        unknown = [n for n in names if n not in _PERF_PATCH_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown perf patch group(s) {unknown}; valid: "
                f"{sorted(_PERF_PATCH_GROUPS)}"
            )
    for name in names:
        _PERF_PATCH_GROUPS[name]()


apply_compiler_perf_patches()
