# Errorless Tableau with Adaptive Controller Validation

## User Stories

### Story 1: Prevent Silent Failures with Incompatible Configurations
**As a** CuBIE user  
**I want** the library to raise a clear error when I try to use an errorless algorithm with an adaptive step controller  
**So that** I don't encounter silent failures or mysterious zero step sizes  

**Acceptance Criteria:**
- When a user creates a `Solver` or calls `solve_ivp` with an errorless algorithm (e.g., explicit Euler, classical RK4) and an adaptive controller (e.g., "pi", "pid"), a `ValueError` is raised
- The error message clearly explains the incompatibility
- The error message suggests using a fixed-step controller instead
- This validation occurs during integrator initialization, not during kernel execution

### Story 2: Automatic Controller Selection Based on Algorithm Capability
**As a** CuBIE user  
**I want** algorithms to automatically select appropriate default controllers  
**So that** I don't need to manually ensure compatibility when using default settings  

**Acceptance Criteria:**
- Errorless algorithms (e.g., explicit Euler, classical RK4, Heun's method) default to fixed-step controllers
- Adaptive algorithms with error estimates (e.g., Dormand-Prince, Bogacki-Shampine) default to adaptive controllers
- When a user explicitly specifies a controller, the system validates compatibility
- The default controller choice is transparent and documented

### Story 3: Clear and Actionable Error Messages
**As a** CuBIE developer debugging integration issues  
**I want** error messages to identify which algorithm and controller are incompatible  
**So that** I can quickly fix the configuration  

**Acceptance Criteria:**
- Error message includes the algorithm name
- Error message includes the controller type
- Error message explains why they are incompatible
- Error message suggests a fix (use fixed controller or adaptive algorithm)

## Overview

This feature adds validation to prevent incompatible algorithm-controller pairings that currently result in silent failures (zero step sizes). The implementation leverages existing `is_adaptive` properties on both algorithms and controllers to perform early validation.

### Current Problem

```
User Request: ERK with classical RK4 tableau + PI controller
             ↓
SingleIntegratorRunCore.__init__
             ↓
Algorithm created: is_adaptive = False (no error estimate)
             ↓
Controller created: is_adaptive = True
             ↓
Loop created with controller.is_adaptive = True
             ↓
KERNEL EXECUTION: Zero step size (uncaught division by zero)
             ↓
SILENT FAILURE
```

### Proposed Solution

```
User Request: ERK with classical RK4 tableau + PI controller
             ↓
SingleIntegratorRunCore.__init__
             ↓
Algorithm created: is_adaptive = False
             ↓
Controller created: is_adaptive = True
             ↓
VALIDATION CHECK: algorithm.is_adaptive vs controller.is_adaptive
             ↓
MISMATCH DETECTED
             ↓
ValueError raised with clear message
             ↓
USER INFORMED BEFORE KERNEL COMPILATION
```

## Key Technical Decisions

### 1. Validation Location
**Decision:** Add validation call in `SingleIntegratorRunCore.__init__` after both algorithm and controller are instantiated (around line 128-129)

**Rationale:**
- The `check_compatibility()` method already exists but is never called
- This is the earliest point where both components are available
- Prevents wasted compilation if configuration is invalid
- Fails fast with clear user feedback

### 2. Dynamic Controller Defaults
**Decision:** Make algorithm controller defaults depend on `is_adaptive` property of the algorithm

**Rationale:**
- Tableaus like `CLASSICAL_RK4_TABLEAU`, `HEUN_21_TABLEAU`, `RALSTON_33_TABLEAU` have no error estimates (`b_hat=None`)
- These should default to fixed-step controllers
- Tableaus like `DORMAND_PRINCE_54_TABLEAU`, `BOGACKI_SHAMPINE_32_TABLEAU` have error estimates
- These should default to adaptive controllers
- Preserves backward compatibility for users not specifying controllers

### 3. Error Message Content
**Decision:** Include algorithm name, controller type, and actionable suggestion

**Example:**
```
ValueError: Adaptive step controller 'pi' cannot be used with 
fixed-step algorithm 'erk'. Algorithm 'erk' with tableau 
'CLASSICAL_RK4' does not provide an error estimate required 
for adaptive step control. Use a fixed-step controller 
('fixed') or an algorithm with an embedded error estimate.
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ SingleIntegratorRunCore.__init__                            │
│                                                               │
│  1. Algorithm Step Creation                                  │
│     ┌────────────────────────────────────────┐              │
│     │ get_algorithm_step(settings)           │              │
│     │   ↓                                     │              │
│     │ ERKStep (or other algorithm)           │              │
│     │   ↓                                     │              │
│     │ is_adaptive property computed from:    │              │
│     │   - tableau.has_error_estimate (ERK)   │              │
│     │   - hardcoded False (Euler, BE)        │              │
│     └────────────────────────────────────────┘              │
│                        ↓                                      │
│  2. Controller Defaults Retrieval                            │
│     ┌────────────────────────────────────────┐              │
│     │ algorithm.controller_defaults          │              │
│     │   ↓                                     │              │
│     │ StepControlDefaults with:              │              │
│     │   - Fixed controller (if not adaptive) │              │
│     │   - Adaptive controller (if adaptive)  │              │
│     └────────────────────────────────────────┘              │
│                        ↓                                      │
│  3. User Settings Override                                   │
│     ┌────────────────────────────────────────┐              │
│     │ controller_settings.update(            │              │
│     │     step_control_settings)             │              │
│     └────────────────────────────────────────┘              │
│                        ↓                                      │
│  4. Controller Creation                                      │
│     ┌────────────────────────────────────────┐              │
│     │ get_controller(controller_settings)    │              │
│     │   ↓                                     │              │
│     │ Controller with is_adaptive property   │              │
│     └────────────────────────────────────────┘              │
│                        ↓                                      │
│  5. VALIDATION (NEW)                                         │
│     ┌────────────────────────────────────────┐              │
│     │ self.check_compatibility()             │              │
│     │   ↓                                     │              │
│     │ if not algo.is_adaptive and            │              │
│     │    controller.is_adaptive:             │              │
│     │     raise ValueError(...)              │              │
│     └────────────────────────────────────────┘              │
│                        ↓                                      │
│  6. Loop Creation (only if valid)                            │
│     ┌────────────────────────────────────────┐              │
│     │ self.instantiate_loop(...)             │              │
│     └────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points with Current Codebase

### Modified Components

1. **SingleIntegratorRunCore.__init__**
   - Location: `src/cubie/integrators/SingleIntegratorRunCore.py:85-160`
   - Change: Add call to `self.check_compatibility()` after controller creation (line ~129)
   
2. **SingleIntegratorRunCore.check_compatibility**
   - Location: `src/cubie/integrators/SingleIntegratorRunCore.py:169-184`
   - Change: Enhance error message to include algorithm and controller names

3. **Algorithm Controller Defaults**
   - Locations:
     - `src/cubie/integrators/algorithms/generic_erk.py` (ERK_DEFAULTS)
     - Other algorithm files with static defaults
   - Change: Make defaults conditional on whether algorithm has error estimation

### Unchanged Components (Referenced)

- `BaseAlgorithmStep.is_adaptive` (abstract property, line 452-454)
- `ButcherTableau.has_error_estimate` (property, line 96-101)
- `BaseStepController.is_adaptive` (property, line 154-157)
- Loop creation and CUDA compilation (no changes needed)

## Expected Impact on Architecture

**Minimal architectural changes:**
- No new classes or modules
- No changes to CUDA kernel generation
- No changes to public API signatures
- No performance impact (validation is O(1) at initialization)

**Behavioral changes:**
- Some configurations that previously failed silently will now raise errors early
- Users will receive clear guidance on fixing configurations
- Default behavior preserved for compatible configurations
- Breaking change: Invalid configurations that were previously allowed (but broken) will now error

## Trade-offs and Alternatives Considered

### Alternative 1: Validate in Loop or Kernel
**Rejected:** Too late; kernel may already be compiled

### Alternative 2: Automatically Coerce Controller
**Rejected:** Silently changing user intent could hide bugs; explicit errors are better

### Alternative 3: Make All Algorithms Adaptive by Padding Errors with Zeros
**Rejected:** Would waste memory and computation; doesn't solve the underlying problem

### Selected Approach: Early Validation + Smart Defaults
**Benefits:**
- Fail fast with clear errors
- Zero runtime overhead (validation at initialization only)
- Preserves user control (can still specify controllers)
- Backward compatible for valid configurations
