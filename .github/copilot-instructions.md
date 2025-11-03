# Copilot Instructions for CuBIE

## Project Overview
CuBIE (CUDA Batch Integration Engine) is a Python library for high-performance batch integration of ODEs and SDEs using CUDA. It uses Numba to JIT-compile CUDA kernels, providing compiled CUDA speed without writing CUDA code directly. The library is designed for simulating large numbers of systems in parallel on NVIDIA GPUs.

## Development Environment

### Setup
- Install with: `pip install -e .[dev]` from the repository root
- Requires Python 3.8+ and CUDA Toolkit 12.9+ with NVIDIA GPU (compute capability 6.0+)
- For CPU-only development/testing: set environment variable `NUMBA_ENABLE_CUDASIM="1"`
- Default terminal is PowerShell - do not use `&&` to chain commands; use `;` or separate commands

### Platform
- Development environment is Windows with PowerShell
- CI runs on Ubuntu (Linux) with Python 3.10, 3.11, 3.12
- Must remain Windows-compatible (project goal)

## Code Style & Conventions

### General Style
- Follow PEP8: max line length 79 characters, comments 71 characters
- Use descriptive variable and function names (not minimal abbreviations)
- Type hints are required in function/method signatures (PEP484 format)
- Do NOT add inline variable type annotations in implementations
- Write numpydoc-style docstrings for all functions and classes
- Do NOT import from `__future__ import annotations` (assume Python 3.8+)
- Comments should explain complex operations to future developers, NOT narrate changes to users

### Commit Messages
- Use Conventional Commit format for all commit and PR messages
- `fix:` - when completely fixing a bug
- `feat:` - when implementing a whole feature (rare)
- `test:` - when modifying tests
- `docs:` - when completing a documentation task
- `chore:` - for everything else

### Architecture-Specific
- See `AGENTS.md` for detailed style guidelines and project structure
- Never call `build()` directly on CUDAFactory subclasses; access via properties (they auto-cache)
- No backwards compatibility enforcement - breaking changes expected during development

### Attrs Classes
- For floating-point attributes: save with leading underscore, add property returning `self.precision(self._attribute)`
- Never add aliases to underscored variables
- Never include underscore in `__init__` calls (attrs handles internally)

## Testing

### Running Tests
- Use `pytest` from command line (not `pytest tests/` - run from repo root)
- Only run tests in current test file (full test suite is slow)
- Mark tests as needed: `nocudasim`, `cupy`, `slow`, `specific_algos`
- Run without CUDA: `pytest -m "not nocudasim and not cupy"`
- Expect CUDA tests to fail without GPU; prompt user to run if needed

### Writing Tests
- Always use pytest fixtures (see `tests/conftest.py` for patterns)
- Parameterize fixtures with settings dictionaries
- Use indirect fixture overrides (observe pattern in conftest.py)
- **Strongly prefer** fixtures instantiating cubie objects over mocks/patches
- Do NOT use `mock` or `patch` unless absolutely unavoidable
- Do NOT shortcut `is_device` or implement patches for CUDA checks
- Do NOT type hint tests
- Failing tests are good tests - don't work around bugs, test intended behavior

## Building & Linting

### Linting
- Linter: flake8 and ruff
- Run: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- Optional: `flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics`
- Ruff config: line-length 79, max-doc-length 72, docstring-code-format enabled

### Testing with Coverage
- Tests run with coverage: `--cov=cubie --cov-report=xml:coverage.xml --cov-report=term-missing`
- Coverage config in `pyproject.toml` under `[tool.coverage]`

## Environment Variables
- **Never modify environment variables** in code (no monkeypatch or similar)
- Set `NUMBA_ENABLE_CUDASIM` externally for CPU simulation, never in source
- Only one AGENTS.md file exists (at repo root)

## Dependencies
- Core: numpy==1.26.4, numba, numba-cuda[cu12], attrs, sympy
- Dev: pytest, pytest-cov, pytest-durations, pytest-json-report, flake8, ruff, cupy-cuda12x, pandas, matplotlib, scipy
- Optional: cupy-cuda12x (for CuPy integration), pandas, matplotlib

## Common Pitfalls
- PowerShell doesn't support `&&` - use `;` or separate commands
- CUDA tests will fail without GPU - always check test markers
- Full test suite is slow - only run relevant test files
- Don't call `build()` on CUDAFactory subclasses directly
- Never modify environment variables in code
- Always use fixtures over mocks in tests