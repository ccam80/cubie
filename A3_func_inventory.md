# A3 Functionality Inventory — `integrators/algorithms/`

---

## 1. `base_algorithm_step.py`

### `ButcherTableau` — Attrs Class

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 1 | Validates `b_hat` length matches `stage_count` when `b_hat` is not None |
| 2 | Raises `ValueError` when `b_hat` length != stage_count |
| 3 | Raises `ValueError` when `b_hat` does not sum to one (tolerance 1e-8) |
| 4 | Raises `ValueError` when `b` does not sum to one (tolerance 1e-8) |
| 5 | Passes validation when `b_hat` is None |

#### Properties

| # | Functionality |
|---|--------------|
| 6 | `d` returns `None` when `b_hat` is None |
| 7 | `d` returns tuple of `(b_i - b_hat_i)` differences when `b_hat` is set |
| 8 | `stage_count` returns `len(self.b)` |
| 9 | `has_error_estimate` returns `False` when `d` is None |
| 10 | `has_error_estimate` returns `False` when all `d` weights are 0.0 |
| 11 | `has_error_estimate` returns `True` when any `d` weight != 0.0 |
| 12 | `first_same_as_last` returns `True` when `c[0]==0.0`, `c[-1]==1.0`, and `a[-1]==b` |
| 13 | `first_same_as_last` returns `False` when `c` is empty |
| 14 | `first_same_as_last` returns `False` when conditions not met |
| 15 | `can_reuse_accepted_start` returns `True` when `c[0]==0.0` |
| 16 | `can_reuse_accepted_start` returns `False` when `c` is empty |
| 17 | `can_reuse_accepted_start` returns `False` when `c[0]!=0.0` |
| 18 | `accumulates_output` returns `True` when `b_matches_a_row` is None |
| 19 | `accumulates_output` returns `False` when `b_matches_a_row` is not None |
| 20 | `accumulates_error` returns `True` when `b_hat_matches_a_row` is None |
| 21 | `accumulates_error` returns `False` when `b_hat_matches_a_row` is not None |
| 22 | `b_matches_a_row` returns row index where `a[row]` matches `b` within 1e-15 |
| 23 | `b_matches_a_row` returns None when no row matches `b` |
| 24 | `b_matches_a_row` prefers the last matching row when multiple match |
| 25 | `b_hat_matches_a_row` returns row index where `a[row]` matches `b_hat` within 1e-15 |
| 26 | `b_hat_matches_a_row` returns None when `b_hat` is None |
| 27 | `b_hat_matches_a_row` returns None when no row matches |

#### `_find_matching_row`

| # | Functionality |
|---|--------------|
| 28 | Returns None immediately when `target_weights` is None |
| 29 | Returns None when no row in `a` matches `target_weights` within tolerance 1e-15 |
| 30 | Returns last matching row index when multiple rows match |
| 31 | Compares only up to `stage_count` elements from each row |

#### `typed_rows`

| # | Functionality |
|---|--------------|
| 32 | Pads rows shorter than `stage_count` with zeros |
| 33 | Converts each entry using `numba_precision` |
| 34 | Returns tuple of tuples |

#### `typed_columns`

| # | Functionality |
|---|--------------|
| 35 | Returns column-major transposition of `typed_rows` output |

#### `a_flat`

| # | Functionality |
|---|--------------|
| 36 | Returns flattened 1D row-major tuple of `a` matrix, precision-typed |

#### `explicit_terms`

| # | Functionality |
|---|--------------|
| 37 | Returns column-major `a` matrix with diagonal and upper elements zeroed |

#### `typed_vector`

| # | Functionality |
|---|--------------|
| 38 | Returns precision-typed tuple from input vector |

#### `error_weights`

| # | Functionality |
|---|--------------|
| 39 | Returns None when `has_error_estimate` is False |
| 40 | Returns precision-typed `d` vector when `has_error_estimate` is True |

### `StepControlDefaults` — Attrs Class

| # | Functionality |
|---|--------------|
| 41 | `copy()` returns a deep copy with a new `step_controller` dict |

### `BaseStepConfig` — Attrs Class

#### Properties

| # | Functionality |
|---|--------------|
| 42 | `settings_dict` returns dict with `n`, `n_drivers`, `precision` |
| 43 | `first_same_as_last` returns `False` when no `tableau` attribute |
| 44 | `first_same_as_last` delegates to `tableau.first_same_as_last` when tableau present |
| 45 | `can_reuse_accepted_start` returns `False` when no `tableau` attribute |
| 46 | `can_reuse_accepted_start` delegates to `tableau.can_reuse_accepted_start` when tableau present |
| 47 | `stage_count` returns 1 when no `tableau` attribute |
| 48 | `stage_count` delegates to `tableau.stage_count` when tableau present |

### `StepCache` — Attrs Class

| # | Functionality |
|---|--------------|
| 49 | Stores `step` device function (required) |
| 50 | Stores optional `nonlinear_solver` device function (default None) |

### `BaseAlgorithmStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 51 | Deep-copies `_controller_defaults` |
| 52 | Calls `setup_compile_settings(config)` |
| 53 | Sets `is_controller_fixed = False` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 54 | Default implementation is a no-op (pass) |

#### `update`

| # | Functionality |
|---|--------------|
| 55 | Returns empty set when `updates_dict` is None and no kwargs |
| 56 | Returns empty set when `updates_dict` is empty dict and no kwargs |
| 57 | Merges `kwargs` into `updates_dict` |
| 58 | Forwards to `update_compile_settings(silent=True)` |
| 59 | Forwards to `buffer_registry.update(self, silent=True)` |
| 60 | Calls `register_buffers()` after buffer update |
| 61 | Valid-but-inapplicable params (in `ALL_ALGORITHM_STEP_PARAMETERS` but not recognized) marked as recognized |
| 62 | Warning emitted for valid-but-inapplicable params when `silent=False` |
| 63 | No warning for valid-but-inapplicable when `silent=True` |
| 64 | `KeyError` raised for truly invalid params when `silent=False` |
| 65 | No `KeyError` for truly invalid when `silent=True` |
| 66 | Returns union of all recognized keys |

#### Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 67 | `n_drivers` | `compile_settings.n_drivers` (cast to int) |
| 68 | `n` | `compile_settings.n` |
| 69 | `controller_defaults` | `_controller_defaults.copy()` |
| 70 | `tableau` | `getattr(compile_settings, 'tableau', None)` |
| 71 | `first_same_as_last` | `compile_settings.first_same_as_last` |
| 72 | `can_reuse_accepted_start` | `compile_settings.can_reuse_accepted_start` |
| 73 | `step_function` | `get_cached_output('step')` |
| 74 | `settings_dict` | `compile_settings.settings_dict` |
| 75 | `evaluate_f` | `compile_settings.evaluate_f` |
| 76 | `evaluate_observables` | `compile_settings.evaluate_observables` |
| 77 | `get_solver_helper_fn` | `compile_settings.get_solver_helper_fn` |
| 78 | `stage_count` | `compile_settings.stage_count` |

#### Abstract Properties

| # | Functionality |
|---|--------------|
| 79 | `threads_per_step` — abstract, raises NotImplementedError |
| 80 | `is_multistage` — abstract, raises NotImplementedError |
| 81 | `is_adaptive` — abstract, raises NotImplementedError |
| 82 | `is_implicit` — abstract, raises NotImplementedError |
| 83 | `order` — abstract, raises NotImplementedError |

---

## 2. `ode_explicitstep.py`

### `ExplicitStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `BaseStepConfig` with no additional fields |

### `ODEExplicitStep` — Class

#### `build`

| # | Functionality |
|---|--------------|
| 2 | Unpacks `evaluate_f`, `numba_precision`, `n`, `evaluate_observables`, `evaluate_driver_at_t`, `n_drivers` from config |
| 3 | Delegates to `build_step()` with those arguments |

#### `build_step`

| # | Functionality |
|---|--------------|
| 4 | Abstract method, raises NotImplementedError |

#### Properties

| # | Functionality |
|---|--------------|
| 5 | `is_implicit` returns `False` |

---

## 3. `ode_implicitstep.py`

### `ImplicitStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | `beta` property returns `self.precision(self._beta)` |
| 2 | `gamma` property returns `self.precision(self._gamma)` |
| 3 | `settings_dict` extends parent with `beta`, `gamma`, `M`, `preconditioner_order`, `get_solver_helper_fn` |

### `ODEImplicitStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Raises `ValueError` when `solver_type` is not 'newton' or 'linear' |
| 5 | Filters `kwargs` into linear solver params (ignoring None values) |
| 6 | Filters `kwargs` into Newton-Krylov params (ignoring None values) |
| 7 | Creates `LinearSolver` with precision, n, and linear kwargs |
| 8 | Creates `NewtonKrylov` wrapping `LinearSolver` when `solver_type='newton'` |
| 9 | Assigns `LinearSolver` directly when `solver_type='linear'` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 10 | Default implementation is a no-op (pass) |

#### `update`

| # | Functionality |
|---|--------------|
| 11 | Returns empty set when no updates |
| 12 | Delegates to `self.solver.update(silent=True)` |
| 13 | Injects `solver_function` from `self.solver.device_function` into updates |
| 14 | Delegates remaining to `super().update(silent=True)` |
| 15 | Returns union of recognized keys from solver and parent |

#### `build`

| # | Functionality |
|---|--------------|
| 16 | Calls `build_implicit_helpers()` first |
| 17 | Unpacks config and delegates to `build_step()` with `solver_function` included |

#### `build_step`

| # | Functionality |
|---|--------------|
| 18 | Abstract method, raises NotImplementedError |

#### `build_implicit_helpers`

| # | Functionality |
|---|--------------|
| 19 | Calls `get_solver_helper_fn("neumann_preconditioner", ...)` |
| 20 | Calls `get_solver_helper_fn("stage_residual", ...)` |
| 21 | Calls `get_solver_helper_fn("linear_operator", ...)` |
| 22 | Updates solver with operator, preconditioner, residual |
| 23 | Stores compiled `solver.device_function` in compile settings |

#### Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 24 | `is_implicit` | returns `True` |
| 25 | `beta` | `compile_settings.beta` |
| 26 | `gamma` | `compile_settings.gamma` |
| 27 | `mass_matrix` | `compile_settings.M` |
| 28 | `preconditioner_order` | `compile_settings.preconditioner_order` (cast to int) |
| 29 | `krylov_atol` | `solver.krylov_atol` |
| 30 | `krylov_rtol` | `solver.krylov_rtol` |
| 31 | `krylov_max_iters` | `solver.krylov_max_iters` (cast to int) |
| 32 | `linear_correction_type` | `solver.linear_correction_type` |
| 33 | `newton_atol` | `getattr(solver, 'newton_atol', None)` |
| 34 | `newton_rtol` | `getattr(solver, 'newton_rtol', None)` |
| 35 | `newton_max_iters` | `getattr(solver, 'newton_max_iters', None)` with int cast |
| 36 | `newton_damping` | `getattr(solver, 'newton_damping', None)` |
| 37 | `newton_max_backtracks` | `getattr(solver, 'newton_max_backtracks', None)` with int cast |
| 38 | `settings_dict` | merges `super().settings_dict` with `solver.settings_dict` |

---

## 4. `explicit_euler.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `EE_DEFAULTS` sets `step_controller='fixed'` and `dt=1e-3` |

### `ExplicitEulerStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 2 | Builds `ExplicitStepConfig` via `build_config` with required params |
| 3 | Passes `EE_DEFAULTS.copy()` as controller defaults |

#### `build_step`

| # | Functionality |
|---|--------------|
| 4 | Device step evaluates `evaluate_f` to get `dxdt` |
| 5 | Computes `proposed_state[i] = state[i] + dt * dxdt[i]` |
| 6 | Branch: calls `evaluate_driver_at_t` when available (`has_evaluate_driver_at_t`) |
| 7 | Branch: skips driver evaluation when not available |
| 8 | Calls `evaluate_observables` on proposed state |
| 9 | Returns `StepCache(step=step, nonlinear_solver=None)` |

#### Properties

| # | Functionality |
|---|--------------|
| 10 | `threads_per_step` returns 1 |
| 11 | `is_multistage` returns `False` |
| 12 | `is_adaptive` returns `False` |
| 13 | `order` returns 1 |

---

## 5. `generic_erk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ERK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `ERK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `ERKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` field defaults to `DEFAULT_ERK_TABLEAU` |
| 4 | `stage_rhs_location` field defaults to `'local'`, validated in `['local','shared']` |
| 5 | `stage_accumulator_location` field defaults to `'local'`, validated in `['local','shared']` |
| 6 | `first_same_as_last` property delegates to `self.tableau.first_same_as_last` |

### `ERKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 7 | Builds `ERKStepConfig` via `build_config` |
| 8 | Selects `ERK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 9 | Selects `ERK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 10 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 11 | Registers `stage_rhs` buffer with size `n`, persistent=True |
| 12 | Registers `stage_accumulator` buffer with size `max(stage_count-1,0) * n` |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 13 | Stage 0: skips RHS evaluation when FSAL cache usable (`first_same_as_last`, all threads accepted) |
| 14 | Stage 0: evaluates RHS when cache not usable |
| 15 | FSAL warp-sync: uses `activemask()` + `all_sync` to check warp-wide acceptance |
| 16 | Accumulates output via weighted sum when `accumulates_output` is True |
| 17 | Assigns output directly from accumulator when `b_row` matches stage |
| 18 | Accumulates error via weighted sum when `accumulates_error` is True |
| 19 | Assigns error directly from accumulator when `b_hat_row` matches stage |
| 20 | Scales accumulated output by `dt_scalar` and adds `state` |
| 21 | Scales accumulated error by `dt_scalar` |
| 22 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 23 | Evaluates drivers at stage times when `has_evaluate_driver_at_t` |
| 24 | Evaluates drivers at end time when `has_evaluate_driver_at_t` |
| 25 | Evaluates observables at each stage and at end time |
| 26 | Returns `int32(0)` status code |

#### Properties

| # | Functionality |
|---|--------------|
| 27 | `is_multistage` returns `tableau.stage_count > 1` |
| 28 | `is_adaptive` returns `tableau.has_error_estimate` |
| 29 | `order` returns `tableau.order` |
| 30 | `threads_per_step` returns 1 |

---

## 6. `generic_erk_tableaus.py`

### `ERKTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `ButcherTableau` with no additional fields (type tag) |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 2 | `HEUN_21_TABLEAU` — 2-stage, order 2, no error estimate |
| 3 | `RALSTON_33_TABLEAU` — 3-stage, order 3, no error estimate |
| 4 | `BOGACKI_SHAMPINE_32_TABLEAU` — 4-stage, order 3, with `b_hat` |
| 5 | `DORMAND_PRINCE_54_TABLEAU` — 7-stage, order 5, with `b_hat` |
| 6 | `CLASSICAL_RK4_TABLEAU` — 4-stage, order 4, no error estimate |
| 7 | `CASH_KARP_54_TABLEAU` — 6-stage, order 5, with `b_hat` |
| 8 | `FEHLBERG_45_TABLEAU` — 6-stage, order 5, with `b_hat` |
| 9 | `DORMAND_PRINCE_853_TABLEAU` — 12-stage, order 8, with `b_hat` |
| 10 | `TSITOURAS_54_TABLEAU` — 7-stage, order 5, with `b_hat` |
| 11 | `VERNER_76_TABLEAU` — 10-stage, order 7, with `b_hat` |
| 12 | `DEFAULT_ERK_TABLEAU` aliases `DORMAND_PRINCE_54_TABLEAU` |
| 13 | `ERK_TABLEAU_REGISTRY` maps string aliases to tableau instances |

---

## 7. `generic_dirk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `DIRK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `DIRK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `DIRKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_DIRK_TABLEAU` |
| 4 | `stage_increment_location` defaults `'local'`, validated `['local','shared']` |
| 5 | `stage_base_location` defaults `'local'`, validated `['local','shared']` |
| 6 | `accumulator_location` defaults `'local'`, validated `['local','shared']` |
| 7 | `stage_rhs_location` defaults `'local'`, validated `['local','shared']` |

### `DIRKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 8 | Creates identity mass matrix `eye(n, dtype=precision)` |
| 9 | Builds `DIRKStepConfig` via `build_config` with fixed `beta=1.0`, `gamma=1.0` |
| 10 | Selects `DIRK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 11 | Selects `DIRK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 12 | Passes `**kwargs` to `super().__init__` (solver params) |
| 13 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 14 | Clears existing buffer registrations via `buffer_registry.clear_parent(self)` |
| 15 | Registers solver child allocators |
| 16 | Registers `stage_increment` (size n, persistent=True) |
| 17 | Registers `accumulator` (size `max(stage_count-1,0)*n`) |
| 18 | Registers `stage_base` (size n, aliases `accumulator`) |
| 19 | Registers `stage_rhs` (size n, persistent=True) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 20 | Calls `get_solver_helper_fn` for preconditioner, residual, operator |
| 21 | Updates solver with operator, preconditioner, residual (no `n` param unlike parent) |
| 22 | Stores `solver.device_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 23 | Stage 0 FSAL: reuses cached RHS when `first_same_as_last`, multistage, all warp threads accepted |
| 24 | Stage 0: copies accepted drivers when `can_reuse_accepted_start` and not FSAL |
| 25 | Stage 0: evaluates drivers at stage time when neither FSAL nor reuse applies |
| 26 | Stage 0 implicit: calls nonlinear solver when `stage_implicit[0]` is True |
| 27 | Stage 0 explicit: skips solver when `stage_implicit[0]` is False |
| 28 | Stages 1..s: streams previous stage RHS into accumulators |
| 29 | Stages 1..s: calls nonlinear solver for implicit stages |
| 30 | Stages 1..s: skips solver for explicit stages |
| 31 | Accumulates output or assigns directly from `b_row` |
| 32 | Accumulates error or assigns directly from `b_hat_row` |
| 33 | Scales accumulated output by `dt_scalar` + adds `state` |
| 34 | Scales accumulated error by `dt_scalar` |
| 35 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 36 | Evaluates drivers and observables at end time |
| 37 | Returns `status_code` encoding solver results |

#### Properties

| # | Functionality |
|---|--------------|
| 38 | `is_multistage` returns `tableau.stage_count > 1` |
| 39 | `is_adaptive` returns `tableau.has_error_estimate` |
| 40 | `is_implicit` returns `True` |
| 41 | `order` returns `tableau.order` |
| 42 | `threads_per_step` returns 1 |

---

## 8. `generic_dirk_tableaus.py`

### `DIRKTableau` — Attrs Class

#### `diagonal`

| # | Functionality |
|---|--------------|
| 1 | Extracts diagonal entries `a[idx][idx]` for each stage |
| 2 | Returns precision-typed tuple via `typed_vector` |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 3 | `IMPLICIT_MIDPOINT_TABLEAU` — 1-stage, order 2 |
| 4 | `TRAPEZOIDAL_DIRK_TABLEAU` — 2-stage, order 2 (ESDIRK) |
| 5 | `LOBATTO_IIIC_3_TABLEAU` — 3-stage, order 4 |
| 6 | `SDIRK_2_2_TABLEAU` — 2-stage, order 2, L-stable |
| 7 | `L_STABLE_DIRK3_TABLEAU` — 3-stage, order 3, L-stable |
| 8 | `L_STABLE_SDIRK4_TABLEAU` — 5-stage, order 4, with `b_hat` |
| 9 | `DIRK_TABLEAU_REGISTRY` maps string aliases to tableau instances |
| 10 | `DEFAULT_DIRK_TABLEAU` aliases `LOBATTO_IIIC_3_TABLEAU` |

---

## 9. `generic_firk.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `FIRK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `FIRK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `FIRKStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_FIRK_TABLEAU` |
| 4 | `stage_increment_location` defaults `'local'`, validated `['local','shared']` |
| 5 | `stage_driver_stack_location` defaults `'local'`, validated `['local','shared']` |
| 6 | `stage_state_location` defaults `'local'`, validated `['local','shared']` |
| 7 | `stage_count` property delegates to `tableau.stage_count` |
| 8 | `all_stages_n` property returns `stage_count * n` |

### `FIRKStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 9 | Creates identity mass matrix |
| 10 | Builds `FIRKStepConfig` via `build_config` |
| 11 | Selects `FIRK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 12 | Selects `FIRK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 13 | Updates solver `n` to `config.all_stages_n` (coupled system) |
| 14 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 15 | Registers solver child allocators |
| 16 | Registers `stage_increment` (size `all_stages_n`, persistent=True) |
| 17 | Registers `stage_driver_stack` (size `stage_count * n_drivers`) |
| 18 | Registers `stage_state` (size `n`) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 19 | Calls `get_solver_helper_fn("n_stage_residual", ...)` with `stage_coefficients` and `stage_nodes` |
| 20 | Calls `get_solver_helper_fn("n_stage_linear_operator", ...)` |
| 21 | Calls `get_solver_helper_fn("n_stage_neumann_preconditioner", ...)` |
| 22 | Updates solver with `n=config.all_stages_n` |
| 23 | Stores `solver.device_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 24 | Initializes `proposed_state` from `state` when `accumulates_output` |
| 25 | Pre-fills `stage_driver_stack` with driver evaluations for all stage times |
| 26 | Calls single coupled `nonlinear_solver` for all stages |
| 27 | Reconstructs stage states from increments and coupling matrix |
| 28 | Direct assignment of `proposed_state` when `b_row` matches stage |
| 29 | Direct assignment of `error` when `b_hat_row` matches stage |
| 30 | Kahan summation for accumulated output when `accumulates_output` |
| 31 | Kahan summation for accumulated error when `accumulates_error` |
| 32 | Skips end-time driver evaluation when `ends_at_one` is True |
| 33 | Evaluates drivers at end time when `ends_at_one` is False |
| 34 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 35 | Evaluates observables at end time |

#### Properties

| # | Functionality |
|---|--------------|
| 36 | `is_multistage` returns `stage_count > 1` |
| 37 | `is_adaptive` returns `tableau.has_error_estimate` |
| 38 | `stage_count` returns `compile_settings.stage_count` |
| 39 | `is_implicit` returns `True` |
| 40 | `order` returns `tableau.order` |
| 41 | `threads_per_step` returns 1 |

---

## 10. `generic_firk_tableaus.py`

### `FIRKTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Subclass of `ButcherTableau` with no additional fields (type tag) |

### `compute_embedded_weights_radauIIA`

| # | Functionality |
|---|--------------|
| 2 | Defaults `order` to `s` (number of stages) when None |
| 3 | Raises `ValueError` when `order > s` |
| 4 | Uses `linalg.solve` when `order == s` (square system) |
| 5 | Uses `linalg.lstsq` when `order < s` (underdetermined) |
| 6 | Builds Vandermonde-like system from collocation nodes |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 7 | `GAUSS_LEGENDRE_2_TABLEAU` — 2-stage, order 4, no `b_hat` |
| 8 | `RADAU_IIA_5_TABLEAU` — 3-stage, order 5, with computed `b_hat` |
| 9 | `DEFAULT_FIRK_TABLEAU` aliases `GAUSS_LEGENDRE_2_TABLEAU` |
| 10 | `FIRK_TABLEAU_REGISTRY` maps string aliases to tableau instances |

---

## 11. `generic_rosenbrock_w.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ROSENBROCK_ADAPTIVE_DEFAULTS` sets PID controller with adaptive settings |
| 2 | `ROSENBROCK_FIXED_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `RosenbrockWStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `tableau` defaults to `DEFAULT_ROSENBROCK_TABLEAU` |
| 4 | `time_derivative_function` optional device function field |
| 5 | `prepare_jacobian_function` optional device function field |
| 6 | `driver_del_t` optional device function field |
| 7 | `stage_rhs_location` defaults `'local'`, validated `['local','shared']` |
| 8 | `stage_store_location` defaults `'local'`, validated `['local','shared']` |
| 9 | `cached_auxiliaries_location` defaults `'local'`, validated `['local','shared']` |
| 10 | `base_state_placeholder_location` defaults `'local'`, validated `['local','shared']` |
| 11 | `krylov_iters_out_location` defaults `'local'`, validated `['local','shared']` |

### `GenericRosenbrockWStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 12 | Creates identity mass matrix |
| 13 | Sets `gamma` from `tableau.gamma` |
| 14 | Builds `RosenbrockWStepConfig` via `build_config` |
| 15 | Initializes `_cached_auxiliary_count = None` |
| 16 | Selects `ROSENBROCK_ADAPTIVE_DEFAULTS` when `tableau.has_error_estimate` is True |
| 17 | Selects `ROSENBROCK_FIXED_DEFAULTS` when `tableau.has_error_estimate` is False |
| 18 | Passes `solver_type='linear'` to parent (not Newton) |
| 19 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 20 | Registers `stage_rhs` (size `n`) |
| 21 | Registers `stage_store` (size `stage_count * n`) |
| 22 | Registers `cached_auxiliaries` initially with size 0 (updated later) |
| 23 | Registers `stage_increment` (size `n`, persistent=True, aliases `stage_store`) |
| 24 | Registers `base_state_placeholder` (size 1, precision=int32) |
| 25 | Registers `krylov_iters_out` (size 1, precision=int32) |

#### `build_implicit_helpers` (override)

| # | Functionality |
|---|--------------|
| 26 | Calls `get_solver_helper_fn("neumann_preconditioner_cached", ...)` |
| 27 | Calls `get_solver_helper_fn("linear_operator_cached", ...)` |
| 28 | Calls `get_solver_helper_fn("prepare_jac", ...)` |
| 29 | Calls `get_solver_helper_fn("cached_aux_count")` to get count |
| 30 | Updates `cached_auxiliaries` buffer size via `buffer_registry.update_buffer` |
| 31 | Calls `get_solver_helper_fn("time_derivative_rhs")` |
| 32 | Updates linear solver with `use_cached_auxiliaries=True` |
| 33 | Stores `solver_function`, `time_derivative_function`, `prepare_jacobian_function` in compile settings |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 34 | Calls `prepare_jacobian` to cache Jacobian info |
| 35 | Evaluates `driver_del_t` when `has_evaluate_driver_at_t` |
| 36 | Zeros `proposed_drivers` when no `evaluate_driver_at_t` |
| 37 | Evaluates `time_derivative_rhs` and scales by `dt_scalar` |
| 38 | Stage 0: evaluates `f(state)` and forms RHS with `gamma_stages[0] * time_derivative` |
| 39 | Stage 0: calls linear solver |
| 40 | Stage 0: accumulates output and error if needed |
| 41 | Stages 1..s: accumulates predecessor contributions via `a_coeffs` |
| 42 | Stages 1..s: evaluates drivers, observables, and f at stage state |
| 43 | Stages 1..s: captures direct output when `b_row` matches stage |
| 44 | Stages 1..s: captures direct error when `b_hat_row` matches stage |
| 45 | Last stage recalculates time derivative before forming RHS |
| 46 | Stages 1..s: forms RHS with C-correction + gamma-derivative terms |
| 47 | Uses previous stage solution as initial guess for linear solver |
| 48 | Forms error as `proposed_state - error` when `not accumulates_error` |
| 49 | Evaluates drivers at end time and observables on proposed state |

#### Properties

| # | Functionality |
|---|--------------|
| 50 | `is_multistage` returns `tableau.stage_count > 1` |
| 51 | `is_adaptive` returns `tableau.has_error_estimate` |
| 52 | `cached_auxiliary_count` lazily builds implicit helpers when `_cached_auxiliary_count` is None |
| 53 | `cached_auxiliary_count` returns cached value when already computed |
| 54 | `is_implicit` returns `True` |
| 55 | `order` returns `tableau.order` |
| 56 | `threads_per_step` returns 1 |

---

## 12. `generic_rosenbrockw_tableaus.py`

### `RosenbrockTableau` — Attrs Class

| # | Functionality |
|---|--------------|
| 1 | Adds `C` field (lower-triangular Jacobian update coefficients) |
| 2 | Adds `gamma` field (diagonal shift, default 0.25) |
| 3 | Adds `gamma_stages` field (per-stage diagonal shifts) |
| 4 | `typed_gamma_stages` returns precision-typed tuple via `typed_vector` |

### Module-Level Functions

| # | Functionality |
|---|--------------|
| 5 | `_ros3p_tableau()` constructs ROS3P tableau with computed C matrix from gamma |
| 6 | `_rodas3p_tableau()` constructs 5-stage RODAS3P tableau |
| 7 | `_rosenbrock_23_sciml_tableau()` constructs 3-stage SciML Rosenbrock23 tableau |

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 8 | `ROS3P_TABLEAU` — 3-stage, order 3, with `b_hat` |
| 9 | `RODAS3P_TABLEAU` — 5-stage, order 3, with `b_hat` |
| 10 | `ROSENBROCK_23_SCIML_TABLEAU` — 3-stage, order 3, with `b_hat` |
| 11 | `ROSENBROCK_TABLEAUS` maps string aliases to tableau instances |
| 12 | `DEFAULT_ROSENBROCK_TABLEAU` aliases `ROS3P_TABLEAU` |

---

## 13. `backwards_euler.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ALGO_CONSTANTS` sets `beta=1.0`, `gamma=1.0`, `M=eye` |
| 2 | `BE_DEFAULTS` sets fixed controller with `dt=1e-3` |

### `BackwardsEulerStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `increment_cache_location` defaults `'local'`, validated `['local','shared']` |

### `BackwardsEulerStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Creates identity mass matrix from `ALGO_CONSTANTS['M'](n, dtype=precision)` |
| 5 | Builds `BackwardsEulerStepConfig` via `build_config` |
| 6 | Passes `BE_DEFAULTS.copy()` as controller defaults |
| 7 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 8 | Registers solver child allocators under name `'solver_scratch'` |
| 9 | Registers `increment_cache` (size n, persistent=True) |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 10 | Initializes `proposed_state` from `increment_cache` (warm start) |
| 11 | Evaluates drivers at `next_time` when `has_evaluate_driver_at_t` |
| 12 | Calls Newton-Krylov solver at `next_time` |
| 13 | Stores increment in `increment_cache` for next step warm start |
| 14 | Computes `proposed_state = increment + state` |
| 15 | Evaluates observables on proposed state |
| 16 | Returns solver status code |

#### Properties

| # | Functionality |
|---|--------------|
| 17 | `is_multistage` returns `False` |
| 18 | `is_adaptive` returns `False` |
| 19 | `threads_per_step` returns 1 |
| 20 | `order` returns 1 |

---

## 14. `backwards_euler_predict_correct.py`

### `BackwardsEulerPCStep` — Class (subclass of `BackwardsEulerStep`)

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 1 | Evaluates `f(state)` to compute explicit predictor |
| 2 | Sets `proposed_state = dt * predictor` as initial guess |
| 3 | Evaluates drivers at `next_time` when `has_evaluate_driver_at_t` |
| 4 | Calls Newton-Krylov solver with predictor as initial guess |
| 5 | Computes `proposed_state = increment + state` |
| 6 | Evaluates observables on proposed state |
| 7 | Returns solver status code |
| 8 | No warm-start cache (unlike parent `BackwardsEulerStep`) |

---

## 15. `crank_nicolson.py`

### Module-Level Constants

| # | Functionality |
|---|--------------|
| 1 | `ALGO_CONSTANTS` sets `beta=1.0`, `gamma=1.0`, `M=eye` |
| 2 | `CN_DEFAULTS` sets PID controller with adaptive settings |

### `CrankNicolsonStepConfig` — Attrs Class

| # | Functionality |
|---|--------------|
| 3 | `dxdt_location` defaults `'local'`, validated `['local','shared']` |

### `CrankNicolsonStep` — Class

#### `__init__`

| # | Functionality |
|---|--------------|
| 4 | Creates identity mass matrix |
| 5 | Builds `CrankNicolsonStepConfig` via `build_config` |
| 6 | Passes `CN_DEFAULTS.copy()` as controller defaults |
| 7 | Calls `register_buffers()` |

#### `register_buffers`

| # | Functionality |
|---|--------------|
| 8 | Registers solver child allocators |
| 9 | Registers `cn_dxdt` (size n, aliases `solver_shared`) |

#### `build_step` — Device Function Logic

| # | Functionality |
|---|--------------|
| 10 | Evaluates `f(state)` to compute `dxdt` at current time |
| 11 | Forms CN base state: `base_state[i] = state[i] + half_dt * dxdt[i]` |
| 12 | Aliases `error` as `base_state` (disjoint lifetimes) |
| 13 | Evaluates drivers at end time when `has_evaluate_driver_at_t` |
| 14 | Solves CN implicit system with `stage_coefficient=0.5` |
| 15 | Computes CN solution: `proposed_state = base_state + 0.5 * increment` |
| 16 | Stores CN increment into `base_state` |
| 17 | Solves BE implicit system with `be_coefficient=1.0` using CN increment as guess |
| 18 | Computes error as `proposed_state - (state + BE_increment)` |
| 19 | Evaluates observables on proposed state |
| 20 | Returns combined status from both solves (bitwise OR) |

#### Properties

| # | Functionality |
|---|--------------|
| 21 | `is_multistage` returns `False` |
| 22 | `is_adaptive` returns `True` |
| 23 | `threads_per_step` returns 1 |
| 24 | `order` returns 2 |
