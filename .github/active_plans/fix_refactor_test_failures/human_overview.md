# Fix Refactor Test Failures: Human Overview

## User Stories

### US-1: Buffer Registration Must Correctly Map Field Names
**As a** CuBIE developer  
**I want** buffer registration to use correct field names when configuring IVPLoop  
**So that** summary metric buffers are properly sized and allocated

**Acceptance Criteria:**
- [ ] `state_summary_buffer_height` parameter maps to `state_summaries_buffer_height` field
- [ ] `observable_summary_buffer_height` parameter maps to `observable_summaries_buffer_height` field  
- [ ] All test_ode_loop tests pass with proper buffer allocation
- [ ] Summary metrics can access buffers with size > 0

### US-2: Step Controller Switching Must Handle Tolerance Arrays
**As a** CuBIE developer  
**I want** switching from fixed to adaptive controllers to properly initialize tolerances  
**So that** tolerance arrays match the system dimension `n`

**Acceptance Criteria:**
- [ ] Switching from fixed to adaptive controller uses correct `n` value
- [ ] Tolerance arrays are broadcast/resized to match new `n` 
- [ ] test_comprehensive_config_plumbing tests pass
- [ ] No ValueError for tolerance shape mismatch

### US-3: Ensure CUDA Simulation Compatibility 
**As a** CuBIE developer  
**I want** CUDA code to work in both real CUDA and simulation mode  
**So that** tests can run without GPU hardware

**Acceptance Criteria:**
- [ ] `cuda.local.array` calls are simulation-safe
- [ ] No AttributeError for `cuda.local` in simulation mode
- [ ] test_solveresult tests pass in CUDASIM mode

---

## Executive Summary

The recent refactor (merging buffer-settings-system into default-parameters) introduced three categories of test failures caused by parameter naming mismatches and tolerance handling issues during controller switching.

## Architecture Impact

```mermaid
flowchart TD
    subgraph IssueA["Issue 1: Field Name Mismatch"]
        A1[IVPLoop.__init__] --> A2["build_config(ODELoopConfig)"]
        A2 -->|"state_summary_buffer_height"| A3[build_config filters out]
        A2 -->|"Expected: state_summaries_buffer_height"| A4[ODELoopConfig field]
        A3 --> A5["Config gets default=0"]
        A5 --> A6["register_buffers() registers size=0"]
        A6 --> A7["IndexError in summary metrics"]
    end
    
    subgraph IssueB["Issue 2: Tolerance Shape Mismatch"]
        B1[_switch_controllers] --> B2["old_settings from FixedStepController"]
        B2 -->|"Contains: n=1, dt=..."| B3["No atol/rtol in old_settings"]
        B3 --> B4["AdaptiveController created"]
        B4 --> B5["atol default = [1e-6] shape (1,)"]
        B5 --> B6["tol_converter with n=3"]
        B6 --> B7["ValueError: shape (1,) != (3,)"]
    end
    
    subgraph IssueC["Issue 3: CUDA Simulation"]
        C1["cuda.local.array in buffer_registry"] --> C2["Simulator doesn't support cuda.local"]
        C2 --> C3["AttributeError"]
    end
```

## Root Cause Analysis

### Issue 1: Buffer Size = 0 (IndexError)

**Location:** `src/cubie/integrators/loops/ode_loop.py`, lines 197-199

**Problem:** Parameter names in `build_config()` call don't match `ODELoopConfig` field names:
- Passed: `'state_summary_buffer_height': state_summaries_buffer_height`
- Expected field: `state_summaries_buffer_height` (with 's')

**Fix:** Correct the parameter names in the `required` dict to match `ODELoopConfig` fields.

### Issue 2: Tolerance Array Shape Mismatch (ValueError)

**Location:** `src/cubie/integrators/SingleIntegratorRunCore.py`, method `_switch_controllers`

**Problem:** When switching from FixedStepController to AdaptiveController:
1. `old_settings` from fixed controller doesn't contain `atol`/`rtol`
2. New adaptive controller uses default `atol = np.asarray([1e-6])` with shape (1,)
3. New `n` value (e.g., 3) comes from `updates_dict`
4. `tol_converter` validates shape (1,) against n=3, raises ValueError

**Fix:** When switching controllers, ensure `n` is passed correctly and tolerance defaults are broadcast to the new `n` value. The issue is that `old_settings` doesn't include `n` from `updates_dict`.

### Issue 3: CUDA Simulation Mode (AttributeError)

**Diagnosis needed:** The error mentions `cuda.local` access. Need to verify if `buffer_registry.py` is accessing `cuda.local` directly instead of through simulation-safe wrappers.

**Note:** Looking at `buffer_registry.py` line 136, it uses `cuda.local.array(_local_size, _precision)` which should work in simulation mode. The error may be in a different location (solveresult.py tests suggest a different code path).

## Technical Decisions

1. **Minimal Change Approach:** Fix only the parameter naming issues without restructuring the configuration system
2. **Backward Compatibility:** Keep all existing public APIs unchanged
3. **Test-Driven Validation:** Use the failing tests to validate fixes

## Trade-offs Considered

| Approach | Pros | Cons |
|----------|------|------|
| Fix field names in ode_loop.py | Minimal change, quick fix | Symptom treatment |
| Refactor build_config | More robust | Larger change scope |
| Add field aliases in ODELoopConfig | Accepts both names | Adds complexity |

**Decision:** Fix field names directly - it's the smallest surgical fix with lowest risk.

## References

- Failing tests identified in issue statement
- `build_config` function in `src/cubie/_utils.py`
- Buffer registry pattern in `.github/context/cubie_internal_structure.md`
