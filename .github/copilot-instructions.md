# Copilot Instructions for CuBIE

## Project Overview
CuBIE (CUDA Batch Integration Engine) is a Python library for high-performance batch integration of ODEs and SDEs using CUDA. It uses Numba to JIT-compile CUDA kernels, providing compiled CUDA speed without writing CUDA code directly. The library is designed for simulating large numbers of systems in parallel on NVIDIA GPUs.

## Available Agent Skills

This repository includes specialized agent skills stored in `.github/skills/`. Copilot will automatically load these skills when relevant to your request.

### Skills Overview

**Pipeline Execution** (`.github/skills/pipeline-execution/SKILL.md`)
- Orchestrates custom agents through the feature development workflow
- Coordinates plan_new_feature → detailed_implementer → taskmaster → run_tests → reviewer
- Manages pipeline termination rules and task group execution

### Request Recognition

Use the **Pipeline Execution** skill when the user requests:
- "run pipeline on issue #X"
- "execute pipeline for issue #X"
- "run the agent pipeline on #X, return after [level]"
- "pipeline issue #X"

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
- NEVER modify changelog.md, this is handled by a plugin.

### Comment Style
- Describe functionality and behavior, NOT implementation changes or history
- Bad: "now computed inline by operators, eliminating the need for a buffer"
- Good: "computed inline by operators; no dedicated buffer required"
- Comments are for understanding current code, not justifying past decisions
- Remove language like "now", "changed from", "no longer", "eliminated", etc.

### Commit Messages
- Use Conventional Commit format for all commit and PR messages
- `fix:` - when completely fixing a bug
- `feat:` - when implementing a whole feature (rare)
- `test:` - when modifying tests
- `docs:` - when completing a documentation task
- `chore:` - for everything else

### Architecture-Specific
- See `.github/context/cubie_internal_structure.md` for detailed project structure
- Never call `build()` directly on CUDAFactory subclasses; access via properties (they auto-cache)
- No backwards compatibility enforcement - breaking changes expected during development
- Use descriptive function names rather than minimal ones

### CUDA Device Code Patterns
- **Prefer predicated commit over conditional branching** in CUDA device functions
- Use compile-time branching when possible
- Avoid `if/else` statements; use predicated assignments instead
- Example pattern:
  ```python
  # Instead of:
  if condition:
      buffer[0] = new_value
  
  # Use predicated commit:
  update_flag = condition
  buffer[0] = selp(update_flag, new_value, buffer[0])
  ```
- This pattern improves warp efficiency by avoiding divergence
- Apply to all summary metrics and other CUDA device code

### Import Guidelines for CUDAFactory Files

Files that define CUDAFactory subclasses or contain CUDA device functions
should use explicit imports instead of whole-module imports:

- Use `from numpy import float32, float64, zeros` instead of `import numpy as np`
- Use `from attrs import define, field` instead of `import attrs`
- Use `from numba import cuda, int32, from_dtype` instead of `import numba`

This reduces the scope captured by Numba during CUDA JIT compilation,
potentially improving compilation time.

**Exception**: Complex modules like `sympy` may remain as whole-module
imports when many diverse symbols are used.

### Import Aliasing Conventions

To avoid name clashes with builtins, math functions, or numba functions:

- **NumPy functions**: Prefix with `np_` (e.g., `from numpy import ceil as np_ceil,
  array as np_array, sum as np_sum`). NumPy scalar types like `float32`, `float64`,
  `int32` also need prefixes (e.g., `from numpy import float32 as np_float32,
  int32 as np_int32, floating as np_floating`) since bare names clash with numba types.
- **Attrs validators**: Prefix with `attrsval_` (e.g., `from attrs.validators import
  instance_of as attrsval_instance_of, optional as attrsval_optional`)
- **Attrs utilities that may clash**: Prefix with `attrs` (e.g., `from attrs import
  Factory as attrsFactory`)
- **Attrs core decorators**: `define`, `frozen`, `field` do not need prefixes as
  they are unlikely to clash with other names.

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
- Style guidelines are consolidated in `.github/copilot-instructions.md`

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
