# Docstring Sweep — Parallel Agent Batch Plan

## Overview

7 agents, launched in parallel as background tasks. Each agent:

1. Reads `.claude/reference/repo_knowledge.md` and `.claude/reference/docstring_template.py`
2. Reads every file in its batch
3. Inventories every public class, method, function, and property
4. For each, records: name, file:line, whether a docstring exists, whether it conforms to the template
5. Writes its results to `.claude/sweep_results/agent_N.md`

Output format per item:
```
### `ClassName.method_name` — `src/cubie/module.py:42`
- **Has docstring**: Yes / No
- **Conforms**: Yes / No / Partial
- **Issues**: (list of deviations from template, or "None")
```

## Agent Batches

### A1 — Core + ODE + symbolic (~8810 lines)
Files:
- `src/cubie/_utils.py`
- `src/cubie/cuda_simsafe.py`
- `src/cubie/buffer_registry.py`
- `src/cubie/CUDAFactory.py`
- `src/cubie/time_logger.py`
- `src/cubie/cubie_cache.py`
- `src/cubie/odesystems/ODEData.py`
- `src/cubie/odesystems/SystemValues.py`
- `src/cubie/odesystems/baseODE.py`
- `src/cubie/odesystems/symbolicODE.py`
- `src/cubie/odesystems/sym_utils.py`
- `src/cubie/odesystems/indexedbasemaps.py`
- `src/cubie/odesystems/odefile.py`
- `src/cubie/odesystems/auxiliary_caching.py`
- `src/cubie/odesystems/jvp_equations.py`

### A2 — Parsing + codegen + CellML (~6686 lines)
Files:
- `src/cubie/odesystems/parser.py`
- `src/cubie/odesystems/function_inspector.py`
- `src/cubie/odesystems/function_parser.py`
- `src/cubie/codegen/numba_cuda_printer.py`
- `src/cubie/codegen/dxdt.py`
- `src/cubie/codegen/time_derivative.py`
- `src/cubie/codegen/jacobian.py`
- `src/cubie/codegen/linear_operators.py`
- `src/cubie/codegen/preconditioners.py`
- `src/cubie/codegen/nonlinear_residuals.py`
- `src/cubie/codegen/_stage_utils.py`
- `src/cubie/cellml/cellml.py`
- `src/cubie/cellml/cellml_cache.py`

### A3 — Algorithms + tableaus (~6867 lines)
Files:
- `src/cubie/integrators/algorithms/base_algorithm_step.py`
- `src/cubie/integrators/algorithms/ode_explicitstep.py`
- `src/cubie/integrators/algorithms/ode_implicitstep.py`
- `src/cubie/integrators/algorithms/explicit_euler.py`
- `src/cubie/integrators/algorithms/generic_erk.py`
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- `src/cubie/integrators/algorithms/backwards_euler.py`
- `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- `src/cubie/integrators/algorithms/crank_nicolson.py`
- `src/cubie/integrators/tableaus/butcher_tableau.py`
- `src/cubie/integrators/tableaus/firk_tableau.py`
- `src/cubie/integrators/tableaus/rosenbrock_w_tableau.py`
- `src/cubie/integrators/tableaus/tableau_library.py`

### A4 — Controllers + solvers + loop (~6200 lines)
Files:
- `src/cubie/integrators/controllers/base_controller.py`
- `src/cubie/integrators/controllers/deadbeat_controller.py`
- `src/cubie/integrators/controllers/elementary_controller.py`
- `src/cubie/integrators/controllers/h211_controller.py`
- `src/cubie/integrators/controllers/pi_controller.py`
- `src/cubie/integrators/controllers/pid_controller.py`
- `src/cubie/integrators/controllers/predictive_controller.py`
- `src/cubie/integrators/solvers/base_solver.py`
- `src/cubie/integrators/solvers/linear_solver.py`
- `src/cubie/integrators/solvers/newton_krylov.py`
- `src/cubie/integrators/solvers/norms.py`
- `src/cubie/integrators/ode_loop_config.py`
- `src/cubie/integrators/ode_loop.py`
- `src/cubie/integrators/array_interpolator.py`

### A5 — Output handling (~6254 lines)
Files:
- `src/cubie/outputhandling/output_config.py`
- `src/cubie/outputhandling/output_sizes.py`
- `src/cubie/outputhandling/output_functions.py`
- `src/cubie/outputhandling/save_state.py`
- `src/cubie/outputhandling/save_summaries.py`
- `src/cubie/outputhandling/update_summaries.py`
- `src/cubie/outputhandling/metrics/metrics.py`
- All files in `src/cubie/outputhandling/metrics/` (21 metric files)

### A6 — Memory + batch solving (~10231 lines)
Files:
- `src/cubie/memory/mem_manager.py`
- `src/cubie/memory/stream_groups.py`
- `src/cubie/memory/cupy_emm.py`
- `src/cubie/memory/chunk_buffer_pool.py`
- `src/cubie/memory/array_requests.py`
- `src/cubie/batchsolving/solver.py`
- `src/cubie/batchsolving/solve_ivp.py`
- `src/cubie/batchsolving/parameter_grid.py`
- `src/cubie/batchsolving/batch_problem.py`
- `src/cubie/batchsolving/batch_config.py`
- `src/cubie/batchsolving/batch_compile.py`
- `src/cubie/batchsolving/batch_run.py`
- `src/cubie/batchsolving/batch_launch.py`
- `src/cubie/batchsolving/batch_results.py`
- `src/cubie/batchsolving/batch_output.py`
- `src/cubie/batchsolving/batch_arrays.py`
- `src/cubie/batchsolving/memory_estimate.py`
- `src/cubie/batchsolving/precision.py`
- `src/cubie/batchsolving/problem_size.py`

### A7 — Init files + misc (~1014 lines)
Files:
- All `__init__.py` files under `src/cubie/`
- `src/cubie/integrators/IntegratorRunSettings.py`
- `src/cubie/vendored/numba_cuda_cache.py`

## Instructions for launching agent

Launch all 7 agents as background Task agents (subagent_type=general-purpose) in a single message. Each agent prompt should:

1. State: "You are inventorying docstrings. Read `.claude/reference/docstring_template.py` first, then read every file listed below. For each public class, method, function, and property, record whether a docstring exists and whether it conforms to the template. Write results to `.claude/sweep_results/agent_N.md`."
2. List the files from the batch
3. Specify the output format shown above

After all agents complete, read all 7 result files and compile a summary.
