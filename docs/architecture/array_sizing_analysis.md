# Array and Loop Iterator Sizing Architecture Analysis

This document provides a comprehensive analysis of the current state of array and loop iterator sizing sources in CuBIE, their flow through the system, and recommendations for architectural improvements.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Size Categories Overview](#size-categories-overview)
3. [Current Architecture Flow Diagrams](#current-architecture-flow-diagrams)
4. [Detailed Analysis by Size Category](#detailed-analysis-by-size-category)
5. [OutputSizes vs BufferSettings Analysis](#outputsizes-vs-buffersettings-analysis)
6. [Recommendations](#recommendations)

---

## Executive Summary

CuBIE's array sizing architecture has evolved organically, leading to some redundancy and confusion about the canonical source of truth for various sizes. The key points of confusion are:

1. **OutputSizes** (`output_sizes.py`) provides helper classes for computing output array shapes for memory allocation
2. **BufferSettings** (`BufferSettings.py` and `ode_loop.py`) provides configuration for internal loop buffers with shared/local memory selection
3. There is overlap in responsibility: both systems compute sizes from the same sources but for different purposes
4. Size information flows from multiple sources (system, output config) and gets processed in multiple places

The analysis below documents the current state and provides options for consolidation.

---

## Size Categories Overview

### Natural Sources of Size Information

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NATURAL SIZE SOURCES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BaseODE / ODEData                           OutputConfig                   │
│  ├── num_states (n)                          ├── n_saved_states            │
│  ├── num_observables                         ├── n_saved_observables       │
│  ├── num_parameters                          ├── n_summarised_states       │
│  ├── num_constants                           ├── n_summarised_observables  │
│  └── num_drivers                             ├── summary_types             │
│                                              ├── dt_save                   │
│       ↓                                      └── dt_summarise              │
│  SystemSizes (ODEData.py)                           ↓                       │
│  ├── states                                  SummaryMetrics                 │
│  ├── observables                             ├── buffer_height_per_var     │
│  ├── parameters                              └── output_height_per_var     │
│  ├── constants                                                              │
│  └── drivers                                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Terminology Distinction

| Term | Source | Purpose | Example |
|------|--------|---------|---------|
| `n_states` | System (BaseODE) | Total state variables in ODE system | 10 |
| `n_saved_states` | OutputConfig | States selected for time-domain output | 5 (if saving subset) |
| `n_summarised_states` | OutputConfig | States selected for summary metrics | 3 (if summarising subset) |
| `n_observables` | System (BaseODE) | Total observable variables | 8 |
| `n_saved_observables` | OutputConfig | Observables selected for output | 4 |
| `n_parameters` | System (BaseODE) | Runtime-tunable parameters | 6 |
| `n_constants` | System (BaseODE) | Compile-time constants | 2 |
| `n_drivers` | System (BaseODE) | External forcing signals | 1 |

---

## Current Architecture Flow Diagrams

### Size Information Flow: System to Device Loop

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                               SIZE FLOW TO DEVICE LOOP                                 │
└───────────────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────┐
                            │   BaseODE   │
                            │  (System)   │
                            └──────┬──────┘
                                   │
                         ┌─────────▼─────────┐
                         │   SystemSizes     │
                         │ states,observables│
                         │ parameters,drivers│
                         │ constants         │
                         └─────────┬─────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│ OutputFunctions │    │ SingleIntegratorRun │    │   Algorithm     │
│                 │    │       Core          │    │     Step        │
│ Uses:           │    │                     │    │                 │
│ - max_states    │    │ Coordinates:        │    │ Uses:           │
│ - max_observ.   │    │ - loop              │    │ - n_states      │
│                 │    │ - algorithm         │    │ for buffers     │
│ Produces:       │    │ - controller        │    │                 │
│ - compile_flags │    │ - output_functions  │    │ Provides:       │
│ - buffer heights│    │                     │    │ - shared_mem_req│
│ - output heights│    │                     │    │ - local_mem_req │
└────────┬────────┘    └──────────┬──────────┘    └────────┬────────┘
         │                        │                        │
         │                        ▼                        │
         │             ┌─────────────────────┐             │
         └─────────────►  LoopBufferSettings ◄─────────────┘
                       │                     │
                       │ Aggregates:         │
                       │ - n_states          │
                       │ - n_parameters      │
                       │ - n_drivers         │
                       │ - n_observables     │
                       │ - summary_buffer_h  │
                       │ - n_error           │
                       │ - n_counters        │
                       │                     │
                       │ + location settings │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │      IVPLoop        │
                       │                     │
                       │ Uses buffer_settings│
                       │ for:                │
                       │ - shared_indices    │
                       │ - local_sizes       │
                       │ - loop compilation  │
                       └─────────────────────┘
```

### Size Information Flow: Output Array Allocation

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                         SIZE FLOW TO OUTPUT ARRAYS                                     │
└───────────────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────┐              ┌─────────────────┐
     │  BatchSolver    │              │  OutputConfig   │
     │  Kernel         │              │                 │
     │                 │              │ - output_types  │
     │ - duration      │              │ - saved indices │
     │ - dt_save       │              │ - summary types │
     │ - num_runs      │              │                 │
     └────────┬────────┘              └────────┬────────┘
              │                                │
              │                                ▼
              │                       ┌─────────────────────┐
              │                       │  OutputFunctions    │
              │                       │                     │
              │                       │  Derives:           │
              │                       │  - n_saved_states   │
              │                       │  - n_saved_observ.  │
              │                       │  - summary heights  │
              │                       └────────┬────────────┘
              │                                │
              ▼                                ▼
     ┌─────────────────────────────────────────────────────┐
     │               output_sizes.py                        │
     │                                                      │
     │  ┌──────────────────────┐  ┌──────────────────────┐ │
     │  │ SummariesBufferSizes │  │  OutputArrayHeights  │ │
     │  │ - state              │  │  - state             │ │
     │  │ - observables        │  │  - observables       │ │
     │  │ - per_variable       │  │  - state_summaries   │ │
     │  └──────────────────────┘  │  - observ._summaries │ │
     │                            │  - per_variable      │ │
     │  ┌──────────────────────┐  └──────────────────────┘ │
     │  │  LoopBufferSizes     │                           │
     │  │  - state_summaries   │  ┌──────────────────────┐ │
     │  │  - observ._summaries │  │ SingleRunOutputSizes │ │
     │  │  - state             │  │ (time × variable)    │ │
     │  │  - observables       │  └──────────────────────┘ │
     │  │  - dxdt              │                           │
     │  │  - parameters        │  ┌──────────────────────┐ │
     │  │  - drivers           │  │  BatchOutputSizes    │ │
     │  └──────────────────────┘  │ (time × run × var)   │ │
     │                            └──────────────────────┘ │
     └─────────────────────────────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │     OutputArrays        │
                              │     InputArrays         │
                              │                         │
                              │  Uses BatchOutputSizes  │
                              │  to create ArrayRequest │
                              │  for MemoryManager      │
                              └─────────────────────────┘
```

---

## Detailed Analysis by Size Category

### 1. States and Derivatives

**Natural Home:** `BaseODE.num_states` → `SystemSizes.states`

**Flow:**
```
BaseODE.num_states
    ↓
ODEData.num_states
    ↓
SystemSizes.states  ←────────────────────────────┐
    │                                            │
    ├──► LoopBufferSettings.n_states             │
    │         ↓                                  │
    │    Loop device buffers:                    │
    │    - state_buffer[n_states]                │
    │    - state_proposal_buffer[n_states]       │
    │    - error[n_states] (if adaptive)         │
    │                                            │
    └──► OutputConfig.max_states ────────────────┘
              ↓
         n_saved_states (after index filtering)
              ↓
         OutputArrayHeights.state
              ↓
         BatchOutputSizes.state = (time_samples, n_runs, n_saved_states)
```

**Usage Inside Device Functions:**
- **n_states**: Used for ALL internal loop buffers (state, proposal, error, dxdt)
- **n_saved_states**: Used ONLY for output array dimensions

**Key Insight:** The loop always processes `n_states` internally, but may only output `n_saved_states` if the user requested a subset.

### 2. Observables

**Natural Home:** `BaseODE.num_observables` → `SystemSizes.observables`

**Flow:**
```
BaseODE.num_observables
    ↓
SystemSizes.observables
    │
    ├──► LoopBufferSettings.n_observables
    │         ↓
    │    Loop device buffers:
    │    - observables_buffer[n_observables]
    │    - observables_proposal_buffer[n_observables]
    │
    └──► OutputConfig.max_observables
              ↓
         n_saved_observables (after index filtering)
              ↓
         OutputArrayHeights.observables
              ↓
         BatchOutputSizes.observables = (time_samples, n_runs, n_saved_obs)
```

### 3. Parameters

**Natural Home:** `BaseODE.num_parameters` → `SystemSizes.parameters`

**Flow:**
```
BaseODE.num_parameters
    ↓
SystemSizes.parameters
    │
    ├──► LoopBufferSettings.n_parameters
    │         ↓
    │    Loop device buffer:
    │    - parameters_buffer[n_parameters]
    │
    └──► BatchInputSizes.parameters = (n_runs, n_parameters)
```

**Note:** Parameters are INPUT arrays, not output. They flow from user to device, not device to user.

### 4. Drivers (External Forcing)

**Natural Home:** `BaseODE.num_drivers` → `SystemSizes.drivers`

**Flow:**
```
BaseODE.num_drivers
    ↓
SystemSizes.drivers
    │
    ├──► LoopBufferSettings.n_drivers
    │         ↓
    │    Loop device buffers:
    │    - drivers_buffer[n_drivers]
    │    - drivers_proposal_buffer[n_drivers]
    │
    └──► BatchInputSizes.driver_coefficients = (segments, n_drivers, order)
```

**Note:** Drivers are INPUT arrays representing interpolated forcing functions.

### 5. Constants

**Natural Home:** `BaseODE.num_constants` → `SystemSizes.constants`

**Flow:**
```
BaseODE.num_constants
    ↓
SystemSizes.constants
    ↓
Compile-time baked into CUDA kernel (not runtime sized)
```

**Note:** Constants are special - they become compile-time literals, not runtime buffers.

### 6. Summary Metrics

**Natural Home:** `OutputConfig.summary_types` → `SummaryMetrics`

**Flow:**
```
OutputConfig.summary_types (e.g., ["max", "rms", "mean"])
    ↓
summary_metrics.summaries_buffer_height(types)
    ↓
OutputConfig.summaries_buffer_height_per_var  ←── Per-variable scratch space
    ↓
Multiplied by n_summarised_states/observables:
    │
    ├──► state_summaries_buffer_height = per_var × n_summarised_states
    │         ↓
    │    LoopBufferSettings.state_summary_buffer_height
    │         ↓
    │    state_summary_buffer[height] in loop
    │
    └──► observable_summaries_buffer_height = per_var × n_summarised_obs
              ↓
         LoopBufferSettings.observable_summary_buffer_height
              ↓
         observable_summary_buffer[height] in loop
```

**Output sizes follow a similar pattern:**
```
summary_metrics.summaries_output_height(types)
    ↓
OutputConfig.summaries_output_height_per_var
    ↓
state_summaries_output_height = per_var × n_summarised_states
    ↓
BatchOutputSizes.state_summaries = (summary_samples, n_runs, output_height)
```

---

## OutputSizes vs BufferSettings Analysis

### Current Responsibilities

| Class | Location | Primary Purpose | Problems |
|-------|----------|-----------------|----------|
| `SummariesBufferSizes` | output_sizes.py | Compute scratch buffer heights for summaries | Duplicates info in OutputConfig |
| `LoopBufferSizes` | output_sizes.py | Compute loop staging dimensions | Overlaps with LoopBufferSettings |
| `OutputArrayHeights` | output_sizes.py | Compute heights for output arrays | Clean, well-scoped |
| `SingleRunOutputSizes` | output_sizes.py | 2D output shapes per run | Clean, well-scoped |
| `BatchOutputSizes` | output_sizes.py | 3D output shapes for batch | Clean, well-scoped |
| `BatchInputSizes` | output_sizes.py | Input array dimensions | Clean, well-scoped |
| `LoopBufferSettings` | ode_loop.py | Configure loop buffer sizes + memory locations | Good but overlaps with LoopBufferSizes |
| `LoopLocalSizes` | ode_loop.py | Local array sizes with nonzero guarantees | Focused, clean |
| `LoopSliceIndices` | ode_loop.py | Shared memory slice layout | Focused, clean |

### The Core Confusion

The confusion stems from two parallel systems:

**1. output_sizes.py classes** - Originally designed for:
- Computing output array shapes for memory allocation
- Providing sizing info to MemoryManager and ArrayRequest
- Supporting the BatchSolver's output array allocation

**2. BufferSettings hierarchy** - Originally designed for:
- Configuring internal loop buffers
- Supporting shared/local memory selection
- Providing slice indices and local sizes for CUDA kernels

**The overlap:** `LoopBufferSizes` in output_sizes.py duplicates sizing logic that is now better handled by `LoopBufferSettings.local_sizes` in ode_loop.py.

### Historical Context

Looking at the code evolution:
1. `output_sizes.py` classes were created to support output array allocation
2. `LoopBufferSettings` was later created to support shared/local memory selection
3. The newer `LoopBufferSettings` subsumed some responsibilities that `LoopBufferSizes` had

---

## Recommendations

### Option A: Consolidate Into BufferSettings (Recommended)

**Goal:** Make BufferSettings the single source of truth for internal buffer sizing, while keeping output_sizes.py focused on output array dimensions.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED ARCHITECTURE (Option A)                      │
└───────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐         ┌─────────────────┐
  │   SystemSizes   │         │  OutputConfig   │
  │ (from ODEData)  │         │                 │
  └────────┬────────┘         └────────┬────────┘
           │                           │
           ▼                           ▼
  ┌────────────────────────────────────────────────┐
  │            LoopBufferSettings                  │
  │  (Single source for internal buffer sizing)    │
  │                                                │
  │  From SystemSizes:          From OutputConfig: │
  │  - n_states                 - summary_height_s │
  │  - n_parameters             - summary_height_o │
  │  - n_drivers                - n_counters       │
  │  - n_observables            - n_error          │
  │                                                │
  │  Provides:                                     │
  │  - shared_memory_elements                      │
  │  - local_memory_elements                       │
  │  - local_sizes (LoopLocalSizes)                │
  │  - shared_indices (LoopSliceIndices)           │
  └────────────────────┬───────────────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────────────┐
  │                 IVPLoop                        │
  │  Uses LoopBufferSettings exclusively           │
  └────────────────────────────────────────────────┘


  ┌─────────────────┐         ┌─────────────────┐
  │ BatchSolverKernel│         │  OutputConfig   │
  │ - duration      │         │                 │
  │ - num_runs      │         │                 │
  └────────┬────────┘         └────────┬────────┘
           │                           │
           ▼                           ▼
  ┌────────────────────────────────────────────────┐
  │              output_sizes.py                   │
  │  (Focused ONLY on output array shapes)         │
  │                                                │
  │  - OutputArrayHeights                          │
  │  - SingleRunOutputSizes                        │
  │  - BatchOutputSizes                            │
  │  - BatchInputSizes                             │
  │                                                │
  │  REMOVED:                                      │
  │  - SummariesBufferSizes (into OutputConfig)    │
  │  - LoopBufferSizes (into LoopBufferSettings)   │
  └────────────────────────────────────────────────┘
```

**Changes Required:**
1. Remove `SummariesBufferSizes` - properties already exist in `OutputConfig`
2. Remove `LoopBufferSizes` - functionality now in `LoopBufferSettings.local_sizes`
3. Update `from_output_fns` methods to use `OutputConfig` directly
4. Add docstrings clarifying the distinct purposes

### Option B: Create Unified SizingContext

**Goal:** Create a single context object that aggregates all sizing information, passed throughout the system.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED ARCHITECTURE (Option B)                      │
└───────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐         ┌─────────────────┐
  │   SystemSizes   │         │  OutputConfig   │
  └────────┬────────┘         └────────┬────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
          ┌─────────────────────────────────┐
          │        SizingContext            │
          │  (Single aggregation point)     │
          │                                 │
          │  @property                      │
          │  system_sizes -> SystemSizes    │
          │                                 │
          │  @property                      │
          │  output_sizes -> OutputSizes    │
          │                                 │
          │  @property                      │
          │  loop_buffer_settings ->        │
          │      LoopBufferSettings         │
          │                                 │
          │  @property                      │
          │  batch_output_sizes(num_runs,   │
          │      duration) -> BatchOutputs  │
          └─────────────┬───────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   OutputFunctions  IVPLoop    BatchSolverKernel
```

**Pros:**
- Single object to pass around
- Clear dependency injection
- Easy to reason about

**Cons:**
- Larger refactor
- May over-couple components
- Runtime-dependent sizes (duration, num_runs) complicate the design

### Option C: Minimal Cleanup (Conservative)

**Goal:** Minimal changes - just add clear docstrings explaining the current architecture.

**Changes Required:**
1. Add comprehensive docstrings to each sizing class explaining its role
2. Add deprecation warnings to `SummariesBufferSizes` and `LoopBufferSizes`
3. Document the recommended usage patterns in this analysis document

---

## Recommendation Summary

**Recommended: Option A (Consolidate Into BufferSettings)**

This option:
- Removes redundant classes (`SummariesBufferSizes`, `LoopBufferSizes`)
- Preserves clean separation: BufferSettings for internal loop buffers, output_sizes.py for output arrays
- Minimal breaking changes (internal refactor)
- Clearer architecture for future development

The key insight is that the current confusion exists because `output_sizes.py` was created before `LoopBufferSettings`, and the newer class has subsumed the older one's responsibilities. Cleaning up by removing the obsolete classes and clarifying docstrings will resolve the architectural confusion.

---

## Appendix: Complete Size Flow Matrix

| Size | Natural Source | Consumers | Purpose |
|------|---------------|-----------|---------|
| n_states | BaseODE.num_states | LoopBufferSettings, Algorithm, Controller | All internal buffers |
| n_saved_states | OutputConfig | OutputArrayHeights, BatchOutputSizes | Output array widths |
| n_summarised_states | OutputConfig | OutputConfig (buffer heights) | Summary buffer dimensions |
| n_observables | BaseODE.num_observables | LoopBufferSettings | Internal observables buffer |
| n_saved_observables | OutputConfig | OutputArrayHeights, BatchOutputSizes | Output array widths |
| n_summarised_observables | OutputConfig | OutputConfig (buffer heights) | Summary buffer dimensions |
| n_parameters | BaseODE.num_parameters | LoopBufferSettings, BatchInputSizes | Parameter buffers |
| n_constants | BaseODE.num_constants | Compile-time | Baked into kernel |
| n_drivers | BaseODE.num_drivers | LoopBufferSettings, BatchInputSizes | Driver buffers |
| n_error | Algorithm.is_adaptive | LoopBufferSettings | Error buffer (0 if fixed-step) |
| n_counters | OutputCompileFlags.save_counters | LoopBufferSettings | Counter buffer (0 or 4) |
| summary_buffer_height | summary_metrics | OutputConfig, LoopBufferSettings | Scratch buffer for summaries |
| summary_output_height | summary_metrics | OutputArrayHeights | Output array heights |
| output_length | BatchSolverKernel | BatchOutputSizes | Time dimension of outputs |
| num_runs | BatchSolverKernel | BatchOutputSizes | Batch dimension of outputs |
