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
## Custom Agent Pipeline Commands

### Running the Agent Pipeline

When the user says "run pipeline on issue #X" or similar commands, interpret this as a request to execute the custom agent pipeline for that issue. The pipeline automates feature development through specialized agents.

**Important**: Pipeline coordination is handled by the **default Copilot agent** (you). Custom agents do NOT have the ability to invoke other custom agents. You are responsible for invoking each agent in the proper sequence based on the `return_after` parameter.

**CRITICAL: During pipeline execution, you (the default agent) should NOT read or edit source files directly**. Your role is purely coordination - invoke the appropriate agents and let them do the work. The exception is reading task_list.md to identify task groups for coordination. You do not have any control over which agents to run. The user will specify entry and return-after agents explicitly if they are different to the defaults. Never deviate from the explicitly ordered or default agents to call.

**Command variations to recognize:**
- "run pipeline on issue #X, return after [level]"
- "execute pipeline for issue #X"
- "run the agent pipeline on #X, return after [level]"
- "pipeline issue #X"

**How to handle pipeline commands:**

1. **Fetch the issue details** using GitHub tools to understand the request
2. **Invoke plan_new_feature agent** with the issue content
3. **Invoke detailed_implementer** with plan_new_feature outputs
4. **For each task group in task_list.md**:
   a. Invoke **taskmaster** for that specific task group
   b. Wait for taskmaster to complete
5. **Invoke run_tests** for complete test verification (once after all taskmasters complete)
6. **Invoke reviewer** to validate implementation
7. **If reviewer suggests edits**: invoke taskmaster for review edits (taskmaster_2)
8. **Invoke run_tests** at pipeline exit for final verification
9. **Report results and terminate** - summarize changes and any remaining test failures

**CRITICAL: Pipeline Termination Rules:**
- The pipeline has a **maximum of 2 taskmaster invocations** after initial task group execution:
  1. One optional taskmaster after first run_tests (step 5) if tests fail
  2. One taskmaster_2 after reviewer (step 7) if reviewer suggests edits
- **Do NOT loop** between run_tests and taskmaster. If tests still fail after taskmaster_2, report the failures to the user and terminate.
- After step 8 (final run_tests), the pipeline MUST terminate regardless of test results.
- Include any remaining test failures in the final summary for user review.

**Default return_after level**: Use `taskmaster_2` for complete implementation
**Default starting agent**: `plan_new_feature` unless specified otherwise

**Pipeline levels (in order):**
1. `plan_new_feature` - Creates user stories, overview, and architectural plan
2. `detailed_implementer` - Creates detailed task list with task groups
3. `taskmaster` - Executes one task group at a time (called per group)
4. `run_tests` - Runs tests once after all taskmaster invocations complete
5. `reviewer` - Reviews implementation against user stories
6. `taskmaster_2` - Applies review edits (final taskmaster invocation)
7. `run_tests_final` - Final test verification (no further fixes attempted)

**Example interpretation:**

User says: "run pipeline on issue #123"

Your action:
```
1. Use github/issue_read to get issue #123 details
2. Invoke plan_new_feature agent with:
   - Prompt: "Issue #123 content and context"
3. Wait for plan_new_feature to complete
4. Invoke detailed_implementer with plan_new_feature outputs
5. Wait for detailed_implementer to complete
6. Read task_list.md to identify task groups (N groups)
7. For each task group 1 to N:
   a. Invoke taskmaster with: "Execute Task Group [i] from task_list.md"
   b. Wait for taskmaster to complete
8. Invoke run_tests for full test suite verification
9. Invoke reviewer with all outputs
10. Wait for reviewer to complete
11. If reviewer suggests edits, invoke taskmaster with review edits
12. Invoke run_tests for final verification
13. Summarize changes and any remaining test failures, then return to user
    - Do NOT invoke additional taskmaster or run_tests calls
    - Report failures for user to examine
```

**Note**: Reading task_list.md to identify task groups is the only file reading you should do during pipeline execution. Do NOT read source files or make edits yourself.

### Running the Renamer Agent

When the user says "run renamer" or similar commands, interpret this as a request to execute the renamer agent to rationalize method, function, property, and attribute names in the codebase.

**Important**: The renamer agent manages a tracking file called `name_info.md` in the repository root. You coordinate the agent by invoking it with the appropriate operation and parameters.

**Command variations to recognize:**
- "run renamer"
- "run renamer on [file/directory]"
- "renamer update for [file/directory]"
- "renamer recommend for [file/directory]"
- "renamer rename [file/directory]"
- "renamer recommend [N] items"
- "renamer rename [N] items"

**Operations:**
1. `update_list` - Scan files and add all names to name_info.md
2. `recommend` - Analyze items and suggest better names
3. `rename` - Execute recommended renames in source files

**How to handle renamer commands:**

**Full workflow (no operation specified):**
When user says "run renamer on [target]":
1. Invoke renamer with operation="update_list" and target file/directory
2. Wait for completion
3. Invoke renamer with operation="recommend" and chunk_size=10 (or user-specified)
4. Wait for completion
5. Invoke renamer with operation="rename" and chunk_size=5 (or user-specified)
6. Wait for completion
7. Present summary to user

**Single operation:**
When user specifies an operation (e.g., "renamer recommend for src/cubie/integrators"):
1. Invoke renamer with the specified operation
2. Use reasonable chunk size (10 for recommend, 5 for rename) unless user specifies
3. Wait for completion
4. Present results to user

**Default chunk sizes:**
- recommend: 10 items per invocation
- rename: 5 items per invocation
- update_list: process all items

**Parameters to pass to renamer agent:**
- Target file or directory path (absolute path)
- Operation: "update_list", "recommend", or "rename"
- Chunk size (optional, for recommend/rename operations)

**Example interpretation:**

User says: "run renamer on src/cubie/integrators"

Your action:
```
1. Invoke renamer agent with:
   - Prompt: "Operation: update_list, Target: /home/runner/work/cubie/cubie/src/cubie/integrators"
2. Wait for renamer to complete
3. Invoke renamer agent with:
   - Prompt: "Operation: recommend, Target: /home/runner/work/cubie/cubie/src/cubie/integrators, Chunk size: 10"
4. Wait for renamer to complete
5. Invoke renamer agent with:
   - Prompt: "Operation: rename, Target: /home/runner/work/cubie/cubie/src/cubie/integrators, Chunk size: 5"
6. Wait for renamer to complete
7. Present summary showing:
   - Items added to tracking
   - Recommendations made
   - Renames executed
   - Status of name_info.md
```

User says: "renamer recommend 15 items"

Your action:
```
1. Invoke renamer agent with:
   - Prompt: "Operation: recommend, Chunk size: 15"
2. Wait for renamer to complete
3. Present recommendations to user
```
