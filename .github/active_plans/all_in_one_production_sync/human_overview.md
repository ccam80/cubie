# User Stories

1. **Debugger aligns with production drivers**
   - *As a developer*, I want `tests/all_in_one.py` to use the exact driver evaluation and derivative device functions used in production so that lineinfo debugging matches shipped kernels.
   - *Acceptance criteria*: Inline driver and driver-derivative factories mirror the production ArrayInterpolator device code (shape handling, wrap/clamp logic, Horner evaluation); no placeholders remain; driver buffers are seeded identically to production loop setup.
   - *Success metrics*: Driver outputs in the debug kernel match production runs for identical inputs and time grids.

2. **Observables parity with generated systems**
   - *As a developer*, I need the all-in-one debug file to expose the same observables device function as the generated Lorenz system so I can validate new observable-driven systems without divergence.
   - *Acceptance criteria*: `observables_factory` returns the production observable computation for the Lorenz fixture (no `pass` stubs) with correct typing and signature; observables are refreshed wherever production kernels expect them.
   - *Success metrics*: Observables buffers populated identically to production kernels during single- and multi-stage steps.

3. **Loop seeding matches solver behavior**
   - *As a tester*, I need driver/observable seeding inside the inline loop to follow production initialization so proposed buffers start from the same state as batch solver kernels.
   - *Acceptance criteria*: Loop entry seeds drivers, observables, states, and caches exactly as `SingleIntegratorRun`/`IVPLoop` do (including reuse of accepted values and fallback to driver function); behavior holds for both driverless and driver-enabled configurations.
   - *Success metrics*: First-step values in debug runs equal production solver traces for the same configuration.

# Overview

This plan aligns `tests/all_in_one.py` with production driver and observable implementations to keep the lineinfo debug kernel faithful to shipped CUDA kernels. The file currently carries placeholders (notably the observables stub and missing loop seeding parity). We will copy the production device functions verbatim from their canonical sources (generated Lorenz observables, ArrayInterpolator inline kernels, and loop seeding used in `IVPLoop`) and ensure the inline loop stages use identical initialization paths.

```mermaid
flowchart TD
    A[Host config\n(constants, tables, drivers)] --> B[ArrayInterpolator\ncoefficients]
    B --> C[Driver device fn\n+ derivative]
    A --> D[Generated dxdt/observables]
    C --> E[Inline loop seeding\n(state/drivers/observables)]
    D --> E
    E --> F[Step kernels\n(ERK/DIRK/FIRK/Rosenbrock)]
    F --> G[Outputs\nstate/observables/counters]
```

Key technical decisions and rationale:
- **Source fidelity**: Use production ArrayInterpolator device implementations and generated Lorenz observables to eliminate divergence between debug and shipped kernels.
- **Seeding parity**: Mirror `IVPLoop` seeding of driver, observable, and proposal buffers so first-step behavior aligns with batch solver execution.
- **Observable refresh points**: Ensure observable calls appear at the same locations as production step factories (pre/post stage, FSAL paths) to support future observable-heavy systems.

Trade-offs and alternatives:
- Could retain simplified stubs for faster edits, but copying production code is required to debug new driver/observable systems accurately.
- Introducing feature flags was considered but rejected to keep the debug kernel identical to production.

Expected impact:
- Debug runs will faithfully reproduce production behavior for drivers/observables, enabling validation of new systems with driver inputs and observable outputs.
- Future additions of observable-rich systems will need no extra debug scaffolding; the inline kernel already matches production semantics.
