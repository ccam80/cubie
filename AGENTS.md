# CuBIE — CUDA Batch Integration Engine

CuBIE JIT-compiles CUDA kernels with Numba to integrate large batches of ODE/SDE systems in
parallel on NVIDIA GPUs — compiled-CUDA speed without writing CUDA, behind a SciPy/MATLAB-like
interface (`solve_ivp`, `Solver`). Domain-agnostic; built for large parameter/initial-condition
sweeps and summary-only extraction (e.g. likelihood-free inference).

## Documentation map
Architecture is documented per directory under `src/cubie/**/AGENTS.md` (each mirrored to a
`CLAUDE.md` symlink). **Start at `src/cubie/AGENTS.md`** — the package root, which defines the
`CUDAFactory` cached-compilation spine and the invariants every subpackage builds on. Device-code
optimisation conventions live in `src/cubie/writing_cuda_functions.md`.

## Setup
- `pip install -e .[dev]` from the repo root (use a venv; some deps are version-pinned).
- **Python 3.10–3.14**, **CUDA 12 or 13** (via the `cuda12`/`cuda13` extras, or a system
  toolkit), **NVIDIA GPU (compute capability ≥6.0)**.
- CPU-only dev/test without a GPU: set `NUMBA_ENABLE_CUDASIM=1` (Numba's CUDA simulator).
- Dev shell is **PowerShell on Windows** — chain with `;`, not `&&`. Staying Windows-compatible is
  a project goal; CI runs on Ubuntu (Python 3.10/3.11/3.12).

## Testing
Run `pytest` from the repo root. `pyproject.toml` `addopts` already applies coverage and
`-n logical` (xdist), so bare `pytest` is parallel + covered. Only run files relevant to your
change — the full suite is slow (run it as a pre-commit check only, and only when the user approves).
- **Simulator (CPU, matches nocuda CI) — a first pass only:**
  `NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy and not specific_algos"`
- **Real GPU (matches CUDA CI; CUDASIM off) — always run to verify results.** The simulator does
  not guarantee on-device correctness; a change is only verified once the real-GPU tests pass:
  `pytest -m "not specific_algos and not sim_only"`
- Markers (`pyproject.toml`): `nocudasim` (real-GPU only), `sim_only` (simulator-only debug),
  `cupy` (needs CuPy), `slow`, `specific_algos` (non-default tableau aliases).
- **Use the shared session-scoped fixtures in `tests/conftest.py`** with their default parameter
  sets unless the user explicitly excepts a case; don't hand-roll fixtures. **Mocks/patches may
  only be added with an explicit user exception.** Don't type-hint tests.
- **A failing test is a good test.** Never soften a test, loosen a tolerance, or use inexact/lax
  assertions to make it pass — even while developing. Fix the code, not the test; assert the exact
  intended behaviour.

## Lint & build
- `ruff` (line-length 79, max-doc-length 72, docstring-code-format) and `flake8`. CI's blocking
  gate: `flake8 . --select=E9,F63,F7,F82 --show-source`.
- Coverage config and pytest markers live in `pyproject.toml`.

## Code style
- PEP8: 79-char lines, 71-char comments. Descriptive names, not abbreviations.
- Type hints on function/method **signatures** only (PEP484) — no inline variable annotations, no
  `from __future__ import annotations` (min Python 3.10). numpydoc docstrings on public API.
- Comments describe current behaviour, not change history ("now", "no longer", "changed from" →
  removed). **Never edit `changelog.md`** (plugin-managed).

## Commits & PRs
- **Conventional Commit format**; description in **present-state changelog language** (describe the
  resulting state, e.g. "nested AGENTS.md files created…"). Types: `fix`, `feat` (rare), `test`,
  `docs`, `chore`.
- **Agents:** every fix or feature is developed on its own branch off `main`. When the work is
  done and verified, commit, push the branch, and open a PR.
- **Performance gate (every PR):** run `benchmarks/lorenz_mean_runtime.py` A/B — A on `main`,
  B on the PR branch (e.g. via `PYTHONPATH=<tree>/src`) — and include the results table in the
  PR message. Script defaults are the gate settings. The script outputs kernel runtime only
  (CUDA-event); one invocation per side suffices — means repeat to ~0.1% — but an invocation
  where a config's std exceeds ~5% of its mean was contaminated by outside interference:
  discard it and rerun. Run A (`main`) first, then pass A's printed mean/std to the B run via
  `--ref-fixed MEAN STD --ref-adaptive MEAN STD`; the script prints a Welch z per config and
  the verdict (`|z| >= 3` = the means differ; positive z on the PR branch = regression).

## Cross-cutting code rules (details in `src/cubie/AGENTS.md`)
- Never call a `CUDAFactory.build()` directly — access compiled functions via the cached properties.
- Never set/modify env vars in source (esp. `NUMBA_ENABLE_CUDASIM`); set them externally.
- Module-scoped imports belong in the file header only; deliberate lazy imports of optional deps
  (Qt) stay function-local. cupy/cupyx are required on a real GPU and imported
  once, conditionally, in `cuda_simsafe` — import them from there (`from cubie.cuda_simsafe
  import cupy, cupyx`), never directly and never lazily.
- In `CUDAFactory`/device-code files, use explicit imports with the project aliasing (`np_`,
  `attrsval_`, `attrs`-prefixed); store float config fields underscored and expose via a
  precision-casting property.
- Device-code optimisation patterns (predicated commit, warp-coherent loops, …) live in
  `src/cubie/writing_cuda_functions.md`.
- No backwards-compatibility burden — breaking changes are expected pre-1.0.

## Dependencies
- **Core:** numpy>=2.0, numba, numba-cuda, attrs, sympy>=1.13.0. cellmlmanip is vendored under
  `src/cubie/vendored/cellmlmanip` (its `lxml`/`networkx`/`Pint>=0.24`/`rdflib` runtime deps are core).
- **CUDA toolkit:** supplied by the `cuda12`/`cuda13` extras (`numba-cuda[cu12]`/`[cu13]`) or an
  existing system install; a bare `pip install cubie` uses whatever toolkit numba-cuda can find.
- **CuPy is required for real-GPU execution** — it is cubie's single device memory allocator.
  The `cuda12`/`cuda13` extras pull in the matching cupy build alongside the toolkit wheels.
  It is imported at `import cubie` through `cubie.cuda_simsafe`; the CUDA simulator
  (`NUMBA_ENABLE_CUDASIM=1`) never requires it.
- **Optional:** pandas (DataFrame output), matplotlib (driver plots).
- CI installs a patched `numba-cuda` fork (`ccam80/numba-cuda@cubie_patch`) for faster compile;
  stock `numba-cuda` works for local dev.
