# System override parametrization report

Default `system_override` (when not provided or falsy) resolves to the
`nonlinear` three-state model via `tests/conftest.py`.

## Summary by system
- **linear**: 41 parametrized cases (heavily concentrated in
  `test_batch_grid_builder.py::test_call_input_types`, 27 cases).
- **three_chamber**: 8 cases.
- **stiff**: 3 cases.
- **constant_deriv**: 1 case.
- **default/nonlinear**: 2 cases (explicit parametrization with `{}`).

## Per-test usage and justification
- `tests/batchsolving/test_system_interface.py::test_update` — `linear`
  (3 cases). Uses a simple linear model to check state/parameter updates;
  behavior is model-agnostic, so it could likely standardize to default.
- `tests/batchsolving/test_batch_grid_builder.py::test_call_input_types`
  — `linear` (27 cases). Exercises input type combinations; not tied to
  model dynamics, so a default system would likely suffice and cut
  recompiles substantially.
- `tests/batchsolving/test_batch_grid_builder.py::test_call_outputs` —
  `linear` (1). Output formatting checks; could standardize.
- `tests/batchsolving/test_SolverKernel.py::test_run` — `three_chamber`
  (2). Full integration smoke/regression; complex system choice is
  justified to cover richer observables and summaries.
- `tests/batchsolving/test_solver.py::test_solve_basic` — `{}` (default
  nonlinear) and `three_chamber` (1 each). Verifies solve on default and a
  more realistic cardiovascular model; keeping both seems reasonable.
- `tests/batchsolving/test_solver.py::test_solver_with_different_systems`
  — `three_chamber`, `stiff`, `linear` (1 each). Explicit cross-system
  compatibility check; justified.
- `tests/batchsolving/test_solveresult.py::TestSolveResultFromSolver::test_time_domain_legend_from_solver`
  — `linear` (1). Legend building
  is system-agnostic; could standardize.
- `tests/batchsolving/test_solveresult.py::TestNaNProcessing` (class) —
  `linear` applied to four tests. NaN handling is generic; likely safe to
  use the default system to reduce recompiles.
- `tests/batchsolving/arrays/test_batchinputarrays.py::`
  `test_input_arrays_with_different_systems` — `three_chamber`, `stiff`,
  `linear` (1 each). Explicit shape/size compatibility check; justified.
- `tests/batchsolving/arrays/test_batchoutputarrays.py::`
  `test_output_arrays_with_different_systems` — `three_chamber`, `stiff`,
  `linear` (1 each). Also explicit cross-system coverage; justified.
- `tests/integrators/algorithms/test_step_algorithms.py::test_against_euler`
  — `constant_deriv` (1). Requires constant-derivative system to validate
  Euler equivalence; justified.
- `tests/integrators/algorithms/test_rosenbrock_tableaus.py::`
  `test_rosenbrock_step_accepts_registry_tableau` — `{}` (default). Uses
  default system; parametrization not needed.
- `tests/integrators/loops/test_ode_loop.py::`
  `test_all_summary_metrics_numerical_check` — `linear` (1). Focuses on
  summary metrics; likely safe to standardize.
- `tests/integrators/step_control/test_controller_equivalence_sequences.py`
  `::TestControllerEquivalence` — `three_chamber` applied to two tests.
  Uses multi-state cardiovascular model; could potentially run on default
  nonlinear system but retains a more realistic stiffness/size mix.
- `tests/integrators/test_SingleIntegratorRun.py::`
  `test_update_routes_to_children` — `linear` (1). Update propagation
  logic is model-agnostic; could standardize.

## Opportunities to reduce recompilation
- The dominant cost comes from `linear` parametrization in
  `test_call_input_types` (27 cases). Switching to the default system
  would remove most recompiles without reducing coverage.
- Other likely standardization candidates: `test_call_outputs`,
  `test_update` (system interface), `TestSolveResultFromSolver`
  legend test, `TestNaNProcessing`, `test_all_summary_metrics_numerical_check`,
  and `test_update_routes_to_children`, all of which assert
  model-agnostic behaviors.
- Cross-system coverage tests (explicitly checking multiple systems),
  `constant_deriv` Euler equivalence, and the `three_chamber` integration
  smoke tests should remain parametrized.
