"""Local Python x CUDA compatibility matrix for cubie.

Runs the real-GPU test suite serially across every supported Python
version against both the CUDA 12 and CUDA 13 toolkit wheels. Provision
the interpreters once with ``uv python install 3.10 3.11 3.12 3.13
3.14`` then run ``nox -s gpu`` for the full matrix, or a single cell
with e.g. ``nox -s "gpu-3.11(cuda='cuda13')"``.
"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = False

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
LANES = {"cuda12": "dev12", "cuda13": "dev13"}


@nox.session(python=PYTHONS)
@nox.parametrize("cuda", list(LANES))
def gpu(session, cuda):
    """Install the CUDA lane and run the real-GPU suite serially."""
    session.install("-e", f".[{LANES[cuda]}]")
    # Prove the resolved interpreter, numpy, toolkit, and binding for
    # this cell by querying numba-cuda directly. `numba -s` is unreliable
    # here: its sysinfo probe references a removed USE_NV_BINDING global
    # and misreports the (now mandatory) NVIDIA binding on cuda13.
    session.run(
        "python",
        "-c",
        "import sys, numpy, numba_cuda; from numba import cuda; "
        "from numba.cuda.cudadrv import runtime as rt, driver as d; "
        "print('py', sys.version.split()[0], "
        "'| numpy', numpy.__version__, "
        "'| numba_cuda', numba_cuda.__version__, "
        "'| available', cuda.is_available(), "
        "'| runtime', rt.get_version(), "
        "'| binding', d.binding.__name__)",
    )
    # Wipe the shared, cwd-relative codegen + compiled-kernel cache before
    # each cell. GENERATED_DIR = getcwd()/generated, and cubie_cache stores
    # compiled kernels under it keyed by a config_hash that does NOT include
    # the CUDA toolkit version — so without this wipe a later cell gets a
    # cache hit and runs an earlier cell's kernel instead of compiling its
    # own, silently cross-contaminating toolkits/interpreters.
    session.run(
        "python",
        "-c",
        "import shutil, pathlib; "
        "shutil.rmtree(pathlib.Path('generated'), ignore_errors=True)",
    )
    # nox runs the cells serially, so each cell keeps the project's
    # default xdist fan-out (-n8 from pyproject addopts) — matching the
    # normal and CI pytest invocation.
    session.run(
        "pytest",
        "-m",
        "not specific_algos and not sim_only",
        *session.posargs,
    )
