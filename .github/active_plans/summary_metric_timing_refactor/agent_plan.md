# Summary Metric Timing Refactor - Agent Plan

## Overview

This plan describes the architectural changes required to refactor the summary metric timing parameters. The refactor renames `dt_save` to `sample_summaries_every` throughout the summary metrics subsystem and updates test utilities to use consistent parameter naming.

---

## Component Changes

### 1. MetricConfig Class (`metrics.py`)

**Location**: `src/cubie/outputhandling/summarymetrics/metrics.py`

**Current State**:
- Has `_dt_save` attribute with `dt_save` property
- Used by derivative metrics for scaling finite differences

**Target State**:
- Rename `_dt_save` → `_sample_summaries_every`
- Rename property `dt_save` → `sample_summaries_every`
- Update docstrings to reflect new naming

**Integration Points**:
- SummaryMetric.__init__() passes this to MetricConfig
- Derivative metric build() methods access via compile_settings

### 2. SummaryMetric Base Class (`metrics.py`)

**Location**: `src/cubie/outputhandling/summarymetrics/metrics.py`

**Current State**:
- `__init__()` takes `dt_save` parameter
- Creates MetricConfig with `dt_save`

**Target State**:
- Rename parameter `dt_save` → `sample_summaries_every`
- Update MetricConfig instantiation
- Update docstrings

**Integration Points**:
- All metric subclasses inherit this behavior
- OutputFunctions.build() calls summary_metrics.update()

### 3. SummaryMetrics Registry Class (`metrics.py`)

**Location**: `src/cubie/outputhandling/summarymetrics/metrics.py`

**Current State**:
- `update()` method propagates `dt_save` to all registered metrics

**Target State**:
- `update()` method propagates `sample_summaries_every` to all metrics
- Existing behavior preserved with new parameter name

### 4. OutputFunctions Class (`output_functions.py`)

**Location**: `src/cubie/outputhandling/output_functions.py`

**Current State**:
- `ALL_OUTPUT_FUNCTION_PARAMETERS` includes `"save_every"`
- `build()` calls `summary_metrics.update(dt_save=config.save_every, ...)`

**Target State**:
- Add `"sample_summaries_every"` to `ALL_OUTPUT_FUNCTION_PARAMETERS`
- Update `build()` to use `sample_summaries_every` from config
- Note: This requires OutputConfig to have a `sample_summaries_every` property

**Integration Points**:
- OutputConfig provides the configuration values
- summary_metrics registry receives the update

### 5. OutputConfig Class (`output_config.py`)

**Location**: `src/cubie/outputhandling/output_config.py`

**Current State**:
- Has `_save_every` attribute and `save_every` property
- `from_loop_settings()` accepts `save_every` parameter

**Target State**:
- Add `_sample_summaries_every` attribute
- Add `sample_summaries_every` property
- Update `from_loop_settings()` to accept `sample_summaries_every`
- Default `sample_summaries_every` to `save_every` if not provided

**Integration Points**:
- OutputFunctions reads this for metric compilation
- IVPLoop provides this value during setup

### 6. Derivative Metric Implementations

**Locations**:
- `src/cubie/outputhandling/summarymetrics/dxdt_max.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_min.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_max.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_min.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**Current State**:
- `build()` methods access `self.compile_settings.dt_save`
- Comments reference `dt_save`

**Target State**:
- Update to access `self.compile_settings.sample_summaries_every`
- Update comments and docstrings
- Preserve numerical behavior (just rename)

---

## Test Utility Changes

### 7. calculate_expected_summaries (`tests/_utils.py`)

**Location**: `tests/_utils.py`

**Current State**:
- Parameters: `summarise_every`, `dt_save`
- `summarise_every` is number of samples per summary
- `dt_save` is time between saved samples

**Target State**:
- Rename `summarise_every` → `samples_per_summary` (this is a count, not a time)
- Rename `dt_save` → `sample_summaries_every` (time between summary samples)
- Update docstrings and comments

**Behavioral Notes**:
- The function calculates expected summaries from pre-generated arrays
- `samples_per_summary` determines how many input samples per summary output
- `sample_summaries_every` is used for derivative scaling

### 8. calculate_single_summary_array (`tests/_utils.py`)

**Location**: `tests/_utils.py`

**Current State**:
- Parameters: `summarise_every`, `dt_save`
- Same semantics as calculate_expected_summaries

**Target State**:
- Rename `summarise_every` → `samples_per_summary`
- Rename `dt_save` → `sample_summaries_every`
- Update all internal usages

### 9. run_reference_loop (`tests/integrators/cpu_reference/loops.py`)

**Location**: `tests/integrators/cpu_reference/loops.py`

**Current State**:
- Reads `sample_summaries_every` from solver_settings (with fallback to save_every)
- Calculates `samples_per_summary = summarise_every / sample_summaries_every`
- Calls `calculate_expected_summaries` with old parameter names

**Target State**:
- Update call to `calculate_expected_summaries` with new parameter names:
  - Pass `samples_per_summary` (already calculated correctly)
  - Pass `sample_summaries_every` instead of `dt_save=save_every`
- Preserve the `sample_summaries_every` extraction logic (already correct)

---

## Expected Interactions

### Compile-Time Flow
```
IVPLoop settings
    ↓
OutputConfig.from_loop_settings(sample_summaries_every=...)
    ↓
OutputFunctions.build()
    ↓
summary_metrics.update(sample_summaries_every=...)
    ↓
Each SummaryMetric.update_compile_settings()
    ↓
MetricConfig with new sample_summaries_every
    ↓
Derivative metric build() captures sample_summaries_every in closure
```

### Test Validation Flow
```
run_reference_loop()
    ↓
Extract sample_summaries_every from solver_settings
    ↓
Calculate samples_per_summary = summarise_every / sample_summaries_every
    ↓
Generate state/observable histories
    ↓
calculate_expected_summaries(
    samples_per_summary=...,
    sample_summaries_every=...,
)
    ↓
Compare against device output
```

---

## Edge Cases to Consider

1. **Default Values**: When `sample_summaries_every` is not provided, it should default to `save_every` to maintain backward-compatible behavior for simple cases.

2. **Precision Handling**: The `sample_summaries_every` value should be converted to the system's precision type when used in calculations.

3. **Validation**: The `sample_summaries_every` must be positive (> 0) similar to existing `save_every` validation.

4. **Metric Cache Invalidation**: Changes to `sample_summaries_every` must invalidate the metric cache, which happens automatically through CUDAFactory's compile_settings mechanism.

---

## Dependencies and Imports

### metrics.py
No new imports required. Existing imports sufficient.

### output_functions.py
No new imports required. Update string in `ALL_OUTPUT_FUNCTION_PARAMETERS`.

### output_config.py
No new imports required. Add new attribute and property.

### Derivative Metrics (all 6 files)
No new imports required. Only rename of compile_settings access.

### tests/_utils.py
No new imports required. Parameter renames only.

### tests/integrators/cpu_reference/loops.py
No new imports required. Parameter passing updates only.

---

## Data Structures

### MetricConfig (attrs class)
```python
@define
class MetricConfig:
    _precision: PrecisionDType
    _sample_summaries_every: float = field(default=0.01, ...)
    
    @property
    def sample_summaries_every(self) -> float: ...
    
    @property
    def precision(self) -> type[floating]: ...
```

### OutputConfig (attrs class)
Adds:
```python
_sample_summaries_every: Optional[float] = field(default=None, ...)

@property
def sample_summaries_every(self) -> float:
    # Return _sample_summaries_every if set, else save_every
```
