The patched MLIR backend wheel
==============================

CuBIE's ``mlir``, ``mlir-cuda12``, and ``mlir-cuda13`` extras install
``cubie-numba-cuda-mlir`` rather than NVIDIA's stock
``numba-cuda-mlir``. This page records what that distribution is, how
it is built, and how to publish a new build.

Why a patched wheel exists
--------------------------

CuBIE carries fixes for numba-cuda-mlir in two forms:

* **Python-side fixes** are applied at runtime by
  ``cubie._mlir_compat``, which is imported before any kernel is
  compiled. Each shim feature-detects the installed backend and
  no-ops when the fix is already present, so these fixes never need
  to be baked into a wheel.
* **Native-code fixes** change C++ that is compiled into the wheel's
  binary artifacts (the ``MLIRToLLVM70`` translation library and the
  ``_cext`` kernel launcher). Nothing imported at runtime can patch
  those, so environments that need them must install a rebuilt
  wheel.

``cubie-numba-cuda-mlir`` is that rebuilt wheel: stock
numba-cuda-mlir source plus only the native-code fixes, published
under a different distribution name because PyPI's
``numba-cuda-mlir`` name belongs to NVIDIA and because cubie is
itself published on PyPI, where dependencies must resolve by name
alone (direct URL requirements are rejected on upload).

The import package is unchanged: the wheel installs
``numba_cuda_mlir``, and ``cubie.cuda_backend`` detects it exactly as
it detects the stock package. Because both distributions own the
same import package, they must never be installed into the same
environment; uninstall one before installing the other.

What the wheel contains
-----------------------

The wheel is built from the ``cubie-wheel`` branch of
`ccam80/numba-cuda-mlir <https://github.com/ccam80/numba-cuda-mlir>`_;
that branch's ``CUBIE_WHEEL.md`` is the step-by-step maintenance
runbook (tracking upstream main, adding patches, building,
validating, publishing). The branch contains:
the upstream ``main`` the branch was last rebased onto, plus a union
merge of the native-code pull requests currently open against
NVIDIA/numba-cuda-mlir:

===== ================================================== ==============
PR    Fix                                                Binary
===== ================================================== ==============
#217  llvm70 typo and lit-check fixes                    MLIRToLLVM70
#219  Query libnvvm for the NVVM IR version              MLIRToLLVM70
#221  Selective fastmath via per-op MLIR attributes      MLIRToLLVM70
#225  Cross-file DWARF lineinfo attribution              MLIRToLLVM70
#233  Asynchronous launch for raise-free kernels         ``_cext``
===== ================================================== ==============

Python-side upstream pull requests are deliberately excluded — they
are already live through ``cubie._mlir_compat`` regardless of the
installed wheel.

Versioning follows ``<upstream release>.<patch iteration>``
(``0.4.1.1`` is the first patched build of the 0.4.1 era), kept in
``src/numba_cuda_mlir/VERSION`` on the branch. Provenance is recorded
in the branch's ``NOTICE`` file, as the Apache-2.0 license requires.

How the wheels are built
------------------------

The branch carries a workflow, ``.github/workflows/cubie-wheels.yml``,
that reproduces upstream's wheel build on GitHub-hosted runners:

1. The patches do not touch LLVM itself — the fork's C++ compiles
   against a stock LLVM/MLIR install — so the workflow's ``find-llvm``
   job locates the newest successful run of NVIDIA's public CI on
   ``main`` whose ``ci/llvm-version.env`` pins match the branch, and
   the build jobs download that run's LLVM install artifacts
   (uploaded on every upstream push, roughly 90-day retention)
   instead of spending hours rebuilding LLVM.
2. Each matrix job (CPython 3.11–3.14 on manylinux x86_64, manylinux
   aarch64, and win_amd64 — twelve wheels; the free-threaded 3.14t
   build is omitted because cubie does not support it) then runs the
   same pinned cibuildwheel invocation as upstream's
   ``build-wheel.yml`` and uploads the wheels as run artifacts.

The workflow runs on every push to ``cubie-wheel``. If the default
workflow token cannot download the cross-repository artifacts, add a
fine-grained personal access token with read access to public
repositories as the ``NVIDIA_ACTIONS_READ_TOKEN`` repository secret
on the fork.

Publishing a new build
----------------------

1. Update the branch: rebase onto the current upstream ``main`` if
   its LLVM pins have moved, re-merge the open native-code PR
   branches, and drop any branch whose PR has merged upstream.
2. Bump the fourth version component in
   ``src/numba_cuda_mlir/VERSION``.
3. Push; the workflow builds the full matrix. Verify one wheel
   locally against cubie's real-GPU test suite before publishing.
4. Run the workflow manually (*Actions → Build cubie wheels → Run
   workflow*) with ``publish`` ticked. Publishing uses PyPI trusted
   publishing: the ``cubie-numba-cuda-mlir`` project on PyPI must
   list the fork repository, workflow file ``cubie-wheels.yml``, and
   environment ``pypi`` as a trusted publisher.

When every native-code PR has merged and NVIDIA publishes a release
containing them, retire the patched wheel by pointing the ``mlir*``
extras in ``pyproject.toml`` back at ``numba-cuda-mlir`` with the
appropriate minimum version.
