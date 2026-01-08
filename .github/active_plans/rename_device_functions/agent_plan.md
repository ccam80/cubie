# Device Function Renaming - Agent Implementation Plan

## Objective

Systematically rename device function references from noun-based to verb-based names throughout the CuBIE codebase.

## Renaming Mappings

### Primary Renamings
- `dxdt_function` → `evaluate_f`
- `observables_function` → `evaluate_observables`
- `driver_function` → `evaluate_driver_at_t`

### Secondary References (Parameter Names in Constants)
- Update `ALL_ALGORITHM_STEP_PARAMETERS` set
- Update parameter filtering and validation logic
- Update error messages and docstrings

## Component-by-Component Changes

### 1. Base ODE System (`src/cubie/odesystems/baseODE.py`)

**Property Renames**:
- `@property dxdt_function` → `@property evaluate_f`
- `@property observables_function` → `@property evaluate_observables`

**Expected Behavior**:
- Properties return compiled CUDA device functions from cache
- No functional logic changes, pure renaming
- Docstrings updated to reflect new names

**Integration Points**:
- Called by algorithm steps to retrieve device functions
- Cached via `get_cached_output()` mechanism

### 2. Algorithm Step Base Classes (`src/cubie/integrators/algorithms/`)

**Files Requiring Changes**:
- `base_algorithm_step.py` - Base class properties and ALL_ALGORITHM_STEP_PARAMETERS constant
- `ode_explicitstep.py` - ExplicitStepConfig and build_step signature
- `ode_implicitstep.py` - Build method signature

**Changes Required**:
- Update `ALL_ALGORITHM_STEP_PARAMETERS` set to include new parameter names
- Remove old parameter names from the set
- Update config class field names
- Update property methods that expose these functions
- Update `build_step` method signatures

**Expected Behavior**:
- Config classes store functions under new field names
- Properties expose functions with new names
- Validation logic uses new parameter names

### 3. Individual Algorithm Implementations

**Files**:
- `explicit_euler.py`
- `backwards_euler.py`
- `backwards_euler_predict_correct.py`
- `crank_nicolson.py`
- `generic_erk.py`
- `generic_dirk.py`
- `generic_firk.py`
- `generic_rosenbrock_w.py`

**Changes Per File**:
- Constructor parameter names
- `build_step` method parameter names
- Local variable names within build methods
- Function call sites (e.g., `dxdt_function(...)` → `evaluate_f(...)`)
- Docstring parameter documentation

**Expected Behavior**:
- Device functions called with new names but identical signatures
- No changes to actual CUDA device code logic
- Configuration passing uses new parameter names

### 4. Integration Loops (`src/cubie/integrators/loops/ode_loop.py`)

**Changes Required**:
- Update any references to these function names
- Likely minimal changes as loops receive step functions, not individual device functions

**Expected Behavior**:
- No functional changes
- May need to update internal variable names for clarity

### 5. Single Integrator Run (`src/cubie/integrators/`)

**Files**:
- `SingleIntegratorRun.py` - Public wrapper with properties
- `SingleIntegratorRunCore.py` - Core implementation

**Changes Required**:
- Property names that expose device functions
- Validation logic comparing system functions to algorithm functions
- Internal variable references

**Expected Behavior**:
- Public properties expose functions with new names
- Validation continues to ensure system and algorithm use same functions

### 6. Batch Solving Components (`src/cubie/batchsolving/`)

**Files**:
- `BatchSolverKernel.py`
- `solver.py`

**Expected Changes**:
- Likely minimal direct references
- May reference through algorithm or system properties

### 7. Utility Functions (`src/cubie/_utils.py`)

**Changes Required**:
- Check docstrings and comments mentioning these function names
- Update any string literals or error messages

### 8. Tests (`tests/`)

**Scope**: 750+ references across all test files

**Systematic Approach**:
- Update fixture definitions that create/pass these functions
- Update test assertions checking function names
- Update instrumented algorithm copies in `tests/integrators/algorithms/instrumented/`
- Update CPU reference implementations if they reference these names

**Files Requiring Changes**:
- `tests/conftest.py` - Global fixtures
- `tests/system_fixtures.py` - ODE system fixtures
- `tests/integrators/` - All integrator tests
- `tests/odesystems/` - ODE system tests
- `tests/batchsolving/` - Solver tests
- `tests/integrators/algorithms/instrumented/` - Instrumented versions must stay in sync

**Expected Behavior**:
- Tests continue to validate same behavior under new names
- No test logic changes, pure renaming

### 9. Documentation (`docs/` and `.github/`)

**Files**:
- `docs/source/API_reference/` - API documentation RST files
- `.github/context/cubie_internal_structure.md` - Internal documentation
- `.github/agents/renamer.md` - Update examples in renamer agent docs
- Any markdown or RST files mentioning these names

**Changes Required**:
- Update code examples
- Update function name references in prose
- Update diagrams or tables showing function names

### 10. CHANGELOG (`CHANGELOG.md`)

**Entry to Add**:
```markdown
## [Version TBD] - Breaking Changes

### Changed
- **BREAKING**: Renamed device function references for improved clarity
  - `dxdt_function` → `evaluate_f`
  - `observables_function` → `evaluate_observables`
  - `driver_function` → `evaluate_driver_at_t`
  - Updated all algorithm steps, ODE systems, and integration loops
  - This is an internal API change; external users are unlikely to be affected
```

## Edge Cases to Consider

### 1. String Literals in Error Messages
Search for strings containing old names in:
- Exception messages
- Warning messages
- Logging statements

### 2. Dictionary Keys
Check for dictionaries using these names as keys:
- Configuration dictionaries
- Cache lookup keys
- Validation mappings

### 3. Comment References
Update comments that refer to these function names:
- Inline comments explaining function flow
- Docstring examples showing usage
- Architecture documentation

### 4. Symbolic Code Generation
Check `src/cubie/odesystems/symbolic/codegen/` for:
- Generated code that might reference these names
- Code templates with function name placeholders
- Hash computations that might include function names

### 5. Instrumented Test Copies
`tests/integrators/algorithms/instrumented/` must be kept in sync:
- These are logging-enabled copies of source algorithms
- Any changes to source must be replicated in instrumented versions
- Only difference should be additional logging, not functional changes

## Dependencies and Order

**Renaming Order** (bottom-up dependency chain):
1. Constants (ALL_ALGORITHM_STEP_PARAMETERS)
2. Config classes (BaseStepConfig, ExplicitStepConfig, etc.)
3. Base ODE properties (BaseODE.dxdt_function → evaluate_f)
4. Algorithm base classes (BaseAlgorithmStep, ODEExplicitStep, ODEImplicitStep)
5. Individual algorithm implementations
6. Integration loops
7. SingleIntegratorRun components
8. Batch solver components
9. Tests (can be done in parallel with source once source is complete)
10. Documentation

**Parallel Work Possible**:
- Tests can be updated in parallel once source changes are done
- Documentation can be updated in parallel
- Different algorithm files can be updated in parallel

## Validation Strategy

After renaming, validate that:
1. **No old names remain in source**: `grep -r "dxdt_function\|observables_function\|driver_function" src/`
2. **No old names remain in tests**: `grep -r "dxdt_function\|observables_function\|driver_function" tests/`
3. **Imports resolve**: No broken imports due to renaming
4. **Tests pass**: Full test suite passes (or at least the same tests that passed before)
5. **Linting passes**: flake8 and ruff checks pass

## Non-Functional Requirements

- Maintain line length limits (79 characters for code, 71 for comments)
- Update docstrings to match new parameter names
- Preserve existing type hints
- No changes to actual CUDA device code logic
- Comments describe current behavior, not the renaming history

