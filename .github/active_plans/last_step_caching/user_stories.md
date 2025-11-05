# User Stories: Last-Step Caching for RODAS*P and RadauIIA5

## User Personas

### Performance-Conscious Scientist
- **Who**: A computational scientist running large-scale ODE/SDE simulations on GPU
- **Goal**: Minimize computational overhead and maximize throughput
- **Context**: Uses RODAS or Radau methods for stiff problems

### Algorithm Developer
- **Who**: A numerical methods researcher implementing or testing Rosenbrock and FIRK methods
- **Goal**: Ensure implementation is mathematically correct and efficient
- **Context**: Needs accurate FSAL property detection

## User Stories

### Story 1: Automatic Performance Optimization
**As a** performance-conscious scientist  
**I want** the RODAS*P and RadauIIA5 methods to automatically skip redundant accumulation  
**So that** my simulations run faster without any code changes  

**Acceptance Criteria**:
- RODAS4P, RODAS5P, and RadauIIA5 tableaus correctly identify when last row of `a` equals `b`
- When this property is detected, the proposed state calculation skips accumulation and copies directly from storage
- When embedded error estimates exist and second-to-last row of `a` equals `b_hat`, error calculation also uses direct copy
- The optimization is transparent to users (no API changes required)
- Results remain numerically identical to within floating-point precision

### Story 2: Correct FSAL Detection
**As an** algorithm developer  
**I want** the FSAL property to be rigorously checked based on row equality  
**So that** I can trust the implementation matches the mathematical specification  

**Acceptance Criteria**:
- Tableau validation checks for exact row equality between `a` and `b`/`b_hat`
- FSAL property is correctly distinguished from the new last-step caching property
- Documentation clearly explains the difference between FSAL and last-step caching
- The current lax FSAL check is replaced with a rigorous implementation

### Story 3: Extensibility for Future Tableaus
**As an** algorithm developer  
**I want** the optimization to apply automatically to any new tableau with matching properties  
**So that** I don't need to manually optimize each new method  

**Acceptance Criteria**:
- Tableau base class provides properties/methods to detect row equality
- Generic algorithm implementations check these properties at compile time
- New tableaus with matching properties automatically benefit from optimization
- No hard-coded tableau-specific logic in step implementations

## Success Metrics

1. **Performance**: Measurable reduction in step execution time for affected tableaus (target: 5-10% reduction based on stage count)
2. **Correctness**: All existing tests pass with identical results
3. **Code Quality**: Zero hard-coded tableau names in algorithm implementations
4. **Coverage**: All RODAS*P and RadauIIA5 methods use optimized code path
