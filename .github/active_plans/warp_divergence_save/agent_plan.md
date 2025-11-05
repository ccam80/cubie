# Agent Plan: Warp Divergence at Step-Save Investigation

## Problem Statement

The current implementation of the save logic in the CuBIE integration loop (`src/cubie/integrators/loops/ode_loop.py`) uses conditional branching based on the `do_save` flag, which can cause warp divergence when threads in the same warp reach save boundaries at different times during adaptive stepping.

**Key Question**: Is this warp divergence a performance problem that needs fixing, or is the current implementation already optimal for typical CuBIE workloads?

## Analysis Summary

Based on investigation of the codebase and the repository owner's analysis (issue #181 comment), the current implementation is likely already optimal. The issue asks to "investigate the feasibility" of alternatives, not necessarily to implement them.

**Owner's Hypothesis** (from issue comment):
- With many steps between saves (typical for likelihood-free inference), the save divergence is infrequent
- Predicated commit would increase workload overall by computing saves every step instead of every Nth step
- Warp sync would prevent threads that have already saved from continuing to the next step

This plan focuses on **documentation and analysis** rather than code changes, with optional benchmarking if validation is desired.

## Detailed Component Descriptions

### Component 1: Save Logic in Integration Loop

**Location**: `src/cubie/integrators/loops/ode_loop.py`, lines 454-483

**Current Behavior**:
```python
if do_save:
    save_state(...)
    if summarise:
        update_summaries(...)
        if (save_idx + 1) % saves_per_summary == 0:
            save_summaries(...)
        summary_idx += 1
    save_idx += 1
```

**Warp Divergence Point**: The `if do_save:` conditional causes branch divergence when different threads have different values of `do_save`.

**When Divergence Occurs**:
- **Fixed-step mode**: Never - all threads compute `do_save = (step_counter % steps_per_save) == 0` and get the same result
- **Adaptive-step mode**: Possible - threads may cross save boundaries at different times depending on their individual step sizes

**Save Frequency Calculation**:
- Fixed mode: `do_save = (step_counter % steps_per_save) == 0`
- Adaptive mode: `do_save = (t + dt[0] + equality_breaker) >= next_save`

### Component 2: Save State Device Function

**Location**: `src/cubie/outputhandling/save_state.py`

**Function**: `save_state_factory()` creates CUDA device functions that copy state, observables, and time values to output buffers.

**Performance Characteristics**:
- Simple memory copy operations
- Loop over `nstates` state variables and `nobs` observable variables
- Inline device function (low overhead)
- Fast execution (typically microseconds per save)

**Implication**: Save operations are fast, so warp divergence duration is short. Threads rejoin quickly.

### Component 3: Warp Synchronization Primitives

**Location**: `src/cubie/cuda_simsafe.py`

**Available Primitives**:
- `activemask()`: Get mask of active threads in warp
- `all_sync(mask, predicate)`: Test if all threads in mask satisfy predicate
- `selp(pred, true_value, false_value)`: Predicated select

**Current Usage**: These primitives are used in:
- Matrix-free solvers for convergence detection
- Main loop for detecting when all threads are finished

**Potential Usage for Save**: Could use `all_sync()` to detect if all threads in warp are ready to save, enabling coordinated save operations.

### Component 4: Summary Metrics System

**Location**: `src/cubie/outputhandling/summarymetrics/`

**Relationship to Save**: Summary updates occur during save operations (if enabled). The divergence affects both save and summary operations.

**Nested Divergence**: The summary save logic has an additional conditional:
```python
if (save_idx + 1) % saves_per_summary == 0:
    save_summaries(...)
```

This creates a second level of potential divergence, though it's even more infrequent than the save divergence.

## Expected Behavior of Recommendations

### Recommendation 1: Document the Architectural Decision (Primary)

**Changes Required**:
1. Add detailed comment block in `ode_loop.py` explaining why branching is used
2. Document the three alternative approaches and why they were rejected
3. Update `cubie_internal_structure.md` with warp divergence analysis

**Expected Files to Modify**:
- `src/cubie/integrators/loops/ode_loop.py` (add comments)
- `.github/context/cubie_internal_structure.md` (add section on warp divergence)
- Optionally: Create `.github/decisions/warp_divergence_save.md` as ADR (Architecture Decision Record)

**Behavior**: No runtime behavior changes. Only documentation improvements.

### Recommendation 2: Create Optional Benchmark (Secondary)

**Purpose**: Validate the hypothesis that current implementation is optimal.

**Location**: Create new file `docs/source/examples/save_divergence_benchmark.py`

**Benchmark Design**:
1. **Test Case 1**: Fixed-step integration with varying save frequencies
2. **Test Case 2**: Adaptive-step integration with varying save frequencies
3. **Test Case 3**: Adaptive-step with high divergence (different initial conditions)

**Measurements**:
- Total integration time
- Time spent in save operations
- Save frequency vs total time correlation

**Expected Result**: Current implementation performs well across typical workloads.

**Note**: This is optional and should only be created if the user wants empirical validation.

### Recommendation 3: Add Inline Comments (Minimal)

**Location**: `src/cubie/integrators/loops/ode_loop.py`, line 454

**Example Comment**:
```python
# Note: This conditional causes warp divergence in adaptive mode when
# threads reach save boundaries at different times. Alternative
# approaches (predicated commit or warp sync) were considered but
# rejected due to higher computational overhead or reduced throughput.
# See .github/decisions/warp_divergence_save.md for full analysis.
if do_save:
    save_state(...)
```

## Architectural Changes Required

**None** - This is an investigation and documentation task, not a refactoring task.

The current architecture is sound. The only change is adding clarity for future developers about why warp divergence is accepted in this location.

## Integration Points with Current Codebase

### Integration Point 1: IVPLoop Class

**Current Structure**: `IVPLoop` compiles the main integration loop via `build()` method.

**No Changes Needed**: The loop compilation logic remains unchanged.

**Documentation Enhancement**: Add properties or methods to expose save configuration for introspection/debugging.

### Integration Point 2: OutputFunctions

**Current Structure**: `OutputFunctions` factory creates save and summary device functions.

**No Changes Needed**: Save functions remain as-is.

**Potential Enhancement**: Could add timing instrumentation to measure save overhead (advanced feature, not required).

### Integration Point 3: CUDA Simulation Mode

**Current Structure**: `cuda_simsafe.py` provides simulation-safe warp primitives.

**Consideration**: In CUDASIM mode, `all_sync()` just returns the predicate without actual synchronization. Any warp-based optimization must work correctly in simulation mode.

**No Changes Needed**: Current implementation works in both modes.

## Expected Interactions Between Components

### Interaction 1: Loop → Save Functions

**Current**: Loop calls `save_state()` conditionally based on `do_save` flag.

**Flow**:
1. Loop computes `do_save` based on fixed or adaptive logic
2. If `do_save` is true, loop calls `save_state()` device function
3. Save function copies data to output buffers
4. Loop increments `save_idx` and updates `next_save`

**Warp Behavior**:
- Threads with `do_save=True`: Execute save path
- Threads with `do_save=False`: Predicated off, wait
- After save: All threads rejoin at next loop iteration

### Interaction 2: Loop → Summary Functions

**Current**: Summary updates occur during save operations (if enabled).

**Flow**:
1. If `do_save` is true and summarisation is enabled
2. Call `update_summaries()` to accumulate statistics
3. Check if summary save is due: `(save_idx + 1) % saves_per_summary == 0`
4. If yes, call `save_summaries()` to commit statistics

**Warp Behavior**: Nested conditionals create additional divergence points, but even more infrequent.

### Interaction 3: Loop → Warp Sync (Not Currently Used)

**Potential**: Could add warp sync primitives to coordinate saves.

**Example Pattern** (not recommended):
```python
mask = activemask()
warp_all_saving = all_sync(mask, do_save)
if warp_all_saving:
    # Coordinated save - all threads together
    save_state(...)
else:
    # Individual save - current approach
    if do_save:
        save_state(...)
```

**Why Not Implemented**: Added complexity with minimal benefit for typical workloads.

## Data Structures and Their Purposes

### Structure 1: LoopSharedIndices

**Purpose**: Defines slices into shared memory for different buffers.

**Relevance**: Save operations read from `state_shared_ind` and `obs_shared_ind` slices.

**No Changes Needed**: Structure is appropriate for current and alternative implementations.

### Structure 2: LoopLocalIndices

**Purpose**: Defines slices into persistent local memory (registers).

**Relevance**: Contains `dt` array used in `do_save` calculation for adaptive mode.

**No Changes Needed**: Structure is appropriate.

### Structure 3: Output Arrays

**Purpose**: Device arrays that receive saved state and observable data.

**Access Pattern**: `state_output[save_idx * save_state_bool, :]`

**Indexing**: Uses `save_idx` incremented only when `do_save=True`, ensuring each save goes to the correct output slot.

**No Changes Needed**: Array structure is optimal.

## Dependencies and Imports Required

**No New Dependencies**: All necessary primitives already exist in the codebase.

**Current Dependencies**:
- `cubie.cuda_simsafe`: For warp primitives (already imported)
- `numba.cuda`: For device function compilation (already imported)
- `cubie.outputhandling`: For save functions (already imported)

## Edge Cases to Consider

### Edge Case 1: Very High Save Frequency

**Scenario**: `dt_save` is very small, resulting in saves every few steps.

**Current Behavior**: More frequent warp divergence, but still less overhead than predicated commit.

**Analysis**: Even with save every 10 steps, current approach does 1/10 the save computations of predicated commit.

**Recommendation**: Document that high save frequency increases divergence but is still optimal.

### Edge Case 2: Extremely Divergent Adaptive Stepping

**Scenario**: Batch contains systems with vastly different stiffness, causing some threads to take tiny steps while others take large steps.

**Current Behavior**: Maximum warp divergence - threads save at different times.

**Worst Case Analysis** (from issue comment):
- 32 threads per warp
- Each thread saves at different time
- Result: 32 separate save executions per warp, 31 threads predicated off each time

**Counter-Analysis**:
- This worst case is unlikely in practice (systems in same batch usually have similar characteristics)
- Even if it occurs, threads quickly rejoin after each save
- Alternative approaches don't help: predicated commit still wastes computation, warp sync still forces waiting

**Recommendation**: Accept this edge case as inherent to adaptive stepping with divergent systems.

### Edge Case 3: Fixed-Step Mode

**Scenario**: Using fixed step size (`is_adaptive=False`).

**Current Behavior**: **Zero divergence** - all threads compute `do_save` identically using step counter.

**Analysis**: Current implementation is optimal for fixed-step mode.

### Edge Case 4: Summary Saves

**Scenario**: Both state saves and summary saves are enabled.

**Current Behavior**: Nested conditionals:
```python
if do_save:
    save_state(...)
    if summarise:
        update_summaries(...)
        if (save_idx + 1) % saves_per_summary == 0:
            save_summaries(...)
```

**Divergence Points**: Three levels of potential divergence (do_save, summarise flag, summary save check).

**Analysis**: Summary saves are even less frequent than state saves, so divergence impact is minimal.

## Performance Considerations

### Consideration 1: Save Operation Latency

**Typical Save Duration**: Microseconds (simple memory copies)

**Implication**: Even if warp diverges, threads rejoin quickly. Minimal performance impact.

### Consideration 2: Steps Between Saves

**Typical Ratio**: 10-1000+ steps per save (varies by application)

**Implication**: Divergence happens infrequently relative to total computation.

**Math**:
- If saves occur every 100 steps
- Divergence occurs at most once per 100 loop iterations
- 99% of iterations have no save divergence

### Consideration 3: Warp Scheduling Overhead

**CUDA Behavior**: Modern CUDA schedulers efficiently handle occasional divergence.

**Mechanism**: When threads diverge, GPU serializes execution paths, then reconverges.

**Cost**: Minimal for infrequent, short-duration divergence (like saves).

### Consideration 4: Alternative Approaches Cost

**Predicated Commit**:
- Cost: N save computations (where N = total steps)
- Benefit: No divergence
- Net: Much higher total cost for typical workloads

**Warp Sync**:
- Cost: Fast threads wait for slow threads at save boundary
- Benefit: Coordinated execution
- Net: Reduced throughput, negates advantage of adaptive stepping

## Testing Strategy

If benchmarking is implemented, the testing strategy should include:

### Test 1: Fixed-Step Baseline

**Purpose**: Verify zero divergence in fixed-step mode.

**Setup**: Simple ODE system, fixed time step, varying save frequencies.

**Expected Result**: Linear scaling with save frequency, no divergence overhead.

### Test 2: Adaptive-Step with Similar Systems

**Purpose**: Verify minimal divergence with homogeneous batch.

**Setup**: Same ODE system, different initial conditions, adaptive stepping.

**Expected Result**: Minimal divergence, performance similar to fixed-step.

### Test 3: Adaptive-Step with Divergent Systems

**Purpose**: Stress-test maximum divergence scenario.

**Setup**: Mix of stiff and non-stiff systems, adaptive stepping, moderate save frequency.

**Expected Result**: Some divergence, but performance still acceptable and better than alternatives.

### Test 4: High Save Frequency

**Purpose**: Test worst-case scenario for current approach.

**Setup**: Save every 5-10 steps.

**Expected Result**: Even with frequent saves, current approach outperforms predicated commit.

## Implementation Priority

### Priority 1: Documentation (Must Have)

Add clear comments and architectural decision documentation explaining the warp divergence issue and why the current implementation is optimal.

**Deliverables**:
1. Inline comments in `ode_loop.py`
2. Updated `cubie_internal_structure.md`
3. Optional: Architecture Decision Record (ADR)

### Priority 2: Benchmarking (Nice to Have)

Create optional benchmark to validate assumptions empirically.

**Deliverables**:
1. `docs/source/examples/save_divergence_benchmark.py`
2. Documentation of benchmark results

### Priority 3: Monitoring (Future Enhancement)

Add instrumentation to measure actual save overhead in production workloads.

**Deliverables** (not for this task):
1. Timing hooks in save functions
2. Performance profiling utilities

## Success Criteria

1. **Documentation**: Clear explanation of warp divergence at save points exists in codebase
2. **Decision Record**: Architectural decision is documented with rationale
3. **Understanding**: Future developers can understand why branching is used despite divergence
4. **Validation** (optional): Benchmark data supports the decision if benchmark is created

## Non-Goals

1. **Code Changes**: Do not modify the integration loop logic (unless benchmarking reveals unexpected issues)
2. **New Features**: Do not add warp sync or predicated commit alternatives
3. **Optimization**: Do not attempt to optimize save operations (they're already fast)

## Open Questions for User

1. **Benchmark Desired?**: Should we create a benchmark to validate the analysis, or is documentation sufficient?
2. **ADR Format?**: Should the architectural decision be documented as a formal ADR in `.github/decisions/` or as an extended comment in the code?
3. **Related Issues?**: Should this plan also address issue #149 (FSAL caching divergence) or keep them separate?

## Conclusion

The investigation points toward **keeping the current implementation** and adding **documentation** to explain the architectural decision. The current branching approach is likely optimal for CuBIE's typical workloads, where saves are infrequent relative to integration steps.

The plan focuses on making this decision explicit and well-documented for future developers, with optional empirical validation through benchmarking.
