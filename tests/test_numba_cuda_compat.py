"""Tests for the numba-cuda patches in :mod:`cubie._numba_cuda_compat`."""

import importlib.util
import inspect
import re
import sys

import pytest

import cubie  # noqa: F401  (applies the compat patches on import)
from cubie.cuda_backend import IS_MLIR

if IS_MLIR:
    pytest.skip(
        "the numba-cuda compat patches only load on the numba-cuda "
        "backend",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nocudasim

HELPER_SOURCE = """\
from numba import cuda


@cuda.jit(device=True, inline=True, lineinfo=True)
def scale(x):
    return x * 2.0 + 1.0
"""


def _file_ids_by_basename(ptx):
    """Map source-file basenames to their PTX .file table indices."""
    ids = {}
    for idx, path in re.findall(r'\.file\s+(\d+)\s+"([^"]*)"', ptx):
        basename = path.replace("\\\\", "\\").replace("\\", "/")
        ids[basename.rsplit("/", 1)[-1]] = idx
    return ids


def test_cross_file_lineinfo(tmp_path):
    """Lines inlined from another file keep that file's ``.file`` entry.

    The helper's body sits at a lower line number than the kernel's
    ``def`` line, so this also exercises the prologue-clamp guard:
    without it the helper's lines are renumbered to the kernel's
    ``def`` line, and without the file scoping they are attributed to
    the kernel's file.
    """
    helper_path = tmp_path / "lineinfo_helper.py"
    helper_path.write_text(HELPER_SOURCE)
    spec = importlib.util.spec_from_file_location(
        "lineinfo_helper", helper_path
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    try:
        spec.loader.exec_module(helper)
        scale = helper.scale

        from numba.core import types
        from numba.cuda import compile_ptx

        def kernel(x):
            x[0] = scale(x[0])

        ptx, _ = compile_ptx(
            kernel, types.void(types.float32[:]), lineinfo=True
        )
    finally:
        del sys.modules[spec.name]

    file_ids = _file_ids_by_basename(ptx)
    assert "lineinfo_helper.py" in file_ids, ptx
    assert "test_numba_cuda_compat.py" in file_ids, ptx

    # getsourcelines returns the line of the "@" decorator line (on
    # every supported Python it scans back to include decorators), so
    # the def is one line on and the body two.
    _, decorator_line = inspect.getsourcelines(scale.py_func)
    body_line = decorator_line + 2

    helper_id = file_ids["lineinfo_helper.py"]
    kernel_id = file_ids["test_numba_cuda_compat.py"]
    assert re.search(rf"\.loc\s+{helper_id}\s+{body_line}\b", ptx), ptx
    assert re.search(rf"\.loc\s+{kernel_id}\s+\d+", ptx), ptx


def _single_file_kernel_ptx(**compile_kwargs):
    from numba.core import types
    from numba.cuda import compile_ptx

    def kernel(x):
        x[0] = x[0] + 1

    ptx, _ = compile_ptx(
        kernel, types.void(types.float32[:]), **compile_kwargs
    )
    return ptx


def test_same_file_lineinfo_unchanged():
    """Single-file kernels keep stock lineinfo output under the patch.

    Exactly one ``.file`` entry (this file) and no lexical-block-file
    rescoping of any ``.loc``.
    """
    ptx = _single_file_kernel_ptx(lineinfo=True)
    file_ids = _file_ids_by_basename(ptx)
    assert list(file_ids) == ["test_numba_cuda_compat.py"], ptx
    kernel_id = file_ids["test_numba_cuda_compat.py"]
    loc_ids = set(re.findall(r"\.loc\s+(\d+)\s", ptx))
    assert loc_ids == {kernel_id}, ptx


def test_no_lineinfo_no_directives():
    """Without lineinfo, the patch adds no debug directives."""
    ptx = _single_file_kernel_ptx()
    assert ".file" not in ptx and ".loc" not in ptx, ptx


def test_relative_source_path_single_file_entry():
    """A relative co_filename does not split the kernel's own lines.

    The subprogram file is stored absolute, so the patch must
    normalize before comparing or every line lands in a spurious
    second file entry for the relative spelling.
    """
    from numba.core import types
    from numba.cuda import compile_ptx

    code = compile("def kernel(x):\n    x[0] += 1\n", "rel_kernel.py", "exec")
    namespace = {}
    exec(code, namespace)

    ptx, _ = compile_ptx(
        namespace["kernel"], types.void(types.float32[:]), lineinfo=True
    )
    assert len(re.findall(r"\.file\s+\d+", ptx)) == 1, ptx
