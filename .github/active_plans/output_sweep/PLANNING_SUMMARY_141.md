# Planning Summary: Issue #141 Output Functions Spree

## Task Completion

✅ **Planning Complete** - This document summarizes the comprehensive implementation plan for issue #141 and all child issues.

## Overview

Issue #141 consolidates 10 child issues requesting new output functions for the CUDA batch integration system. This planning exercise analyzed:
- Current architecture and constraints
- Data flow through device-function chains
- Signature requirements for uniform interfaces
- Buffer optimization opportunities
- Implementation complexity and ordering

## Quick Reference: Metrics to Implement

### Implementation Order by Dependency

#### Phase 1: Architecture Changes (Prerequisites)

| Issue | Feature | Type | Complexity | Dependencies |
|-------|---------|------|-----------|--------------|
| #76 | save_exit_state | State mgmt | High | None - modify kernel |
| #125 | iteration_counts | Diagnostics | High | None - modify algorithms |

#### Phase 2: Summary Metrics (Standard Architecture)

**Simple Metrics:**

| Issue | Metric | Complexity | Buffer | Output | Dependencies |
|-------|--------|-----------|--------|--------|--------------|
| #63 | min | Low | 1 | 1 | Architecture complete |
| #61 | max_magnitude | Low | 1 | 1 | Architecture complete |
| #62 | std | Medium | 2 | 1 | Architecture complete |

**Peak Detection Metrics:**

| Issue | Metric | Complexity | Buffer | Output | Dependencies |
|-------|--------|-----------|--------|--------|--------------|
| #64 | negative_peak | Medium | 3+n | n | Simple metrics tested |
| #65 | extrema | Medium | 6+2n | 2n | Simple metrics tested |

**Derivative Metrics:**

| Issue | Metric | Complexity | Buffer | Output | Dependencies |
|-------|--------|-----------|--------|--------|--------------|
| #66 | dxdt_extrema | High | 5+n | n | Peak metrics tested |
| #67 | d2xdt2_extrema | High | 8+n | n | Peak metrics tested |
| #68 | dxdt | Medium | 1 | 1 | Peak metrics tested |

## Key Architectural Decisions

### 1. Uniform Signature Mandate

**All summary_metric device functions must follow:**
```cuda
update(value, buffer, current_index, customisable_variable)
save(buffer, output_array, summarise_every, customisable_variable)
```

**Device functions:**
- Return nothing (mutate arrays in place)
- Support both float32 and float64
- Decorated with `@cuda.jit(..., device=True, inline=True)`

### 2. Buffer Optimization Strategy

**Combined Statistics Opportunity:**
When mean + rms + std requested together:
- Standard: 3 separate buffers → 3 slots total
- Optimized: Shared buffer → 2 slots total
- Implementation: Future enhancement, create `combined_stats.py`

**Benefit:**
- Reduces memory footprint
- Single pass through data
- Only compiled when combination requested

### 3. Data Source Mapping

**For all metrics except #76 and #125:**
- **Source:** Current state/observable values from integrator
- **Chain:** Integrator → update_summaries → metric.update() → buffer
- **Sink:** save_summaries → metric.save() → output_array

**For #76 (save_exit_state):**
- **Source:** Final integrator state values
- **Chain:** Integrator → save_exit_state() → d_inits array
- **Sink:** BatchInputArrays.fetch_inits() → host

**For #125 (iteration_counts):**
- **Source:** Newton/Krylov/Step-controller counters
- **Chain:** Algorithm → increment counter → accumulate
- **Sink:** Periodic save → output_array

### 4. Signature Changes Required

**For #76:**
```python
# New factory in save_state.py
save_exit_state_factory(num_states: int) -> Callable

# Kernel gets additional parameters
BatchSolverKernel(..., continuation: bool, d_inits: DeviceArray)

# BatchInputArrays gets new method
fetch_inits() -> HostArray
```

**For #125:**
```python
# Algorithm device functions get counter parameters
newton_iteration(..., iteration_counter: DeviceArray) -> int
krylov_solve(..., krylov_counter: DeviceArray) -> int

# Compile setting
enable_iteration_counting: bool = False  # default
```

## Buffer Requirements Summary

### Simple Metrics (1-2 slots)
- min, max_magnitude: 1 slot each
- std: 2 slots (sum, sum_of_squares)
- dxdt: 1 slot (previous value)

### Peak Detection (3+n to 8+n slots)
- negative_peak: 3 + n (state + counter + indices)
- extrema: 6 + 2n (state for both + counters + indices)
- dxdt_extrema: 5 + n (value history + derivative history + peaks)
- d2xdt2_extrema: 8 + n (extended history for second derivative)

### Non-Summary
- save_exit_state: No buffer (direct copy)
- iteration_counts: 3 counters per save window

## Implementation Risks

### High Risk
1. **Numerical accuracy (derivatives):** Finite differences accumulate errors
   - *Mitigation:* Document limitations, provide accuracy tests
   
2. **Buffer size explosion:** Large n in parameterized metrics
   - *Mitigation:* Add warnings, document recommended n values

### Medium Risk
3. **Performance degradation:** Many metrics = many function calls
   - *Mitigation:* Keep inline=True, profile, optimize combined_stats

4. **Breaking changes (#76, #125):** Kernel signature modifications
   - *Mitigation:* Backward compatibility, optional parameters, regression tests

## Recommended Implementation Order

### Phase 1: Architecture Changes (Foundation)
**Must be completed first - provides infrastructure for metrics**

1. **save_exit_state (#76)**
   - New device function in save_state.py
   - Kernel modifications for continuation mode
   - BatchInputArrays.fetch_inits() method
   - Continuation tests

2. **iteration_counts (#125)**
   - Counter propagation through solver chain
   - Compile-time flag for enabling/disabling
   - Performance validation
   - Integration tests

**Deliverable:** Architecture supports all planned metrics

### Phase 2: Simple Metrics (Validate Workflow)
**Dependencies:** Phase 1 complete and tested

1. min
2. max_magnitude
3. std
4. Update __init__.py registration
5. Unit tests for each

**Deliverable:** Working template for remaining metrics

### Phase 3: Peak Detection Metrics
**Dependencies:** Phase 2 complete and tested

1. negative_peak
2. extrema
3. Comprehensive peak detection tests

**Deliverable:** All peak variants complete

### Phase 4: Derivative Metrics
**Dependencies:** Phase 3 complete and tested

1. dxdt_extrema
2. d2xdt2_extrema
3. dxdt (optional, lower priority)
4. Numerical accuracy validation

**Deliverable:** All metrics complete, tested, documented

## Success Criteria

- [ ] 10/10 issues implemented
- [ ] All metrics registered in summary_metrics
- [ ] Unit tests: >95% coverage
- [ ] Integration tests: multi-metric usage validated
- [ ] No performance regression (<5% overhead)
- [ ] Documentation updated with examples

## Next Steps

1. **Implement Phase 1** - Architecture changes (#76, #125)
2. **Test Phase 1** - Validate continuation and iteration counting
3. **Implement Phase 2** - Simple metrics with new architecture
4. **Iterate through remaining phases** with continuous testing
5. **Document** each metric with usage examples

## Detailed Documentation

See `IMPLEMENTATION_PLAN_141.md` for:
- Complete technical specifications for each metric
- Detailed buffer layouts and algorithms
- Testing strategy (unit, integration, system)
- File-by-file checklist
- Code examples and pseudocode
- Risk mitigation details

---

**Status:** Planning complete, ready for implementation  
**Files to Create:** 8 new metrics + 11 test files  
**Files to Modify:** 5 existing files  
**Total LOC Estimate:** ~2000-2500 lines
