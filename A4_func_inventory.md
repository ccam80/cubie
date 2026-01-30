# A4 Functionality Inventory

Covers step controllers, matrix-free solvers, norms, loop config/loop, and array interpolator.

---

## `step_control/base_step_controller.py`

### `ControllerCache` (attrs)

| # | Functionality |
|---|--------------|
| 1 | `device_function` defaults to `-1` |

### `BaseStepControllerConfig` (attrs, abstract)

| # | Functionality |
|---|--------------|
| 2 | `n` defaults to 1, validated as int >= 0 |
| 3 | `timestep_memory_location` defaults to `"local"`, validated in `["local", "shared"]` |
| 4 | `__attrs_post_init__` calls super |
| 5 | `dt_min` abstract property |
| 6 | `dt_max` abstract property |
| 7 | `dt0` abstract property |
| 8 | `is_adaptive` abstract property |
| 9 | `settings_dict` abstract property returns dict with `"n"` |

### `BaseStepController`

| # | Functionality |
|---|--------------|
| 10 | `__init__` calls `super().__init__()` |
| 11 | `register_buffers` registers `"timestep_buffer"` with size from `local_memory_elements`, location from `compile_settings.timestep_memory_location`, persistent=True |
| 12 | `build` abstract, returns `ControllerCache` |
| 13 | `n` forwards to `compile_settings.n` |
| 14 | `dt_min` forwards to `compile_settings.dt_min` |
| 15 | `dt_max` forwards to `compile_settings.dt_max` |
| 16 | `dt0` forwards to `compile_settings.dt0` |
| 17 | `is_adaptive` forwards to `compile_settings.is_adaptive` |
| 18 | `local_memory_elements` abstract property |
| 19 | `settings_dict` forwards to `compile_settings.settings_dict` |

#### `BaseStepController.update`

| # | Functionality |
|---|--------------|
| 20 | `updates_dict` defaults to `{}` when None |
| 21 | `kwargs` merged into `updates_dict` |
| 22 | Empty dict returns empty set immediately |
| 23 | `update_compile_settings` called with `silent=True` |
| 24 | Unrecognised keys split into valid-but-inapplicable (in `ALL_STEP_CONTROLLER_PARAMETERS`) vs truly invalid |
| 25 | Valid-but-inapplicable keys added to recognised set |
| 26 | Warning emitted for valid-but-inapplicable keys naming the controller class and listing params |
| 27 | No warning when valid-but-inapplicable is empty |
| 28 | `KeyError` raised for truly invalid keys when `silent=False` |
| 29 | No `KeyError` when `silent=True` with truly invalid keys |
| 30 | Returns full recognised set |

---

## `step_control/fixed_step_controller.py`

### `FixedStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 31 | `_dt` defaults to 1e-3, validated as float > 0 |
| 32 | `__attrs_post_init__` calls super and `_validate_config` |
| 33 | `dt` property returns `precision(self._dt)` |
| 34 | `dt_min` returns `self.dt` |
| 35 | `dt_max` returns `self.dt` |
| 36 | `dt0` returns `self.dt` |
| 37 | `is_adaptive` returns `False` |
| 38 | `settings_dict` extends super with `"dt"` key |

### `FixedStepController`

| # | Functionality |
|---|--------------|
| 39 | `__init__` creates `FixedStepControlConfig` via `build_config`, calls `setup_compile_settings` and `register_buffers` |
| 40 | `build` returns `ControllerCache` with device function that sets `accept_out[0] = 1` and returns `0` |
| 41 | `local_memory_elements` returns `0` |
| 42 | `dt` property forwards to `compile_settings.dt` |

---

## `step_control/adaptive_step_controller.py`

### `AdaptiveStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 43 | `_dt_min` defaults 1e-6, validated float > 0 |
| 44 | `_dt_max` defaults 1.0, validated float > 0 |
| 45 | `atol` defaults `[1e-6]`, converter `tol_converter` |
| 46 | `rtol` defaults `[1e-6]`, converter `tol_converter` |
| 47 | `algorithm_order` defaults 1, validated int >= 1 |
| 48 | `_min_gain` defaults 0.3, validated float in (0, 1) |
| 49 | `_max_gain` defaults 2.0, validated float > 1 |
| 50 | `_safety` defaults 0.9, validated float in (0, 1) |
| 51 | `_deadband_min` defaults 1.0, validated float in (0, 1.0) |
| 52 | `_deadband_max` defaults 1.2, validated float >= 1.0 |

#### `__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 53 | `dt_max` set to `dt_min * 100` when `_dt_max` is None |
| 54 | Warning + `dt_max = dt_min * 100` when `dt_max < dt_min` |
| 55 | No adjustment when `dt_max >= dt_min` |
| 56 | Deadband min/max swapped when `deadband_min > deadband_max` |
| 57 | No swap when `deadband_min <= deadband_max` |

#### Properties

| # | Functionality |
|---|--------------|
| 58 | `dt_min` returns `precision(self._dt_min)` |
| 59 | `dt_max` returns `precision(self._dt_max)`, fallback `dt_min * 100` if None |
| 60 | `dt0` returns `precision(sqrt(dt_min * dt_max))` |
| 61 | `is_adaptive` returns `True` |
| 62 | `min_gain` returns `precision(self._min_gain)` |
| 63 | `max_gain` returns `precision(self._max_gain)` |
| 64 | `safety` returns `precision(self._safety)` |
| 65 | `deadband_min` returns `precision(self._deadband_min)` |
| 66 | `deadband_max` returns `precision(self._deadband_max)` |
| 67 | `settings_dict` extends super with dt_min, dt_max, atol, rtol, algorithm_order, min_gain, max_gain, safety, deadband_min, deadband_max, dt |

### `BaseAdaptiveStepController`

| # | Functionality |
|---|--------------|
| 68 | `__init__` calls super, `setup_compile_settings`, `register_buffers` |
| 69 | `build` calls `build_controller` with config-derived args including `clamp_factory(precision)` |
| 70 | `build_controller` abstract method |
| 71 | `min_gain` forwards to `compile_settings.min_gain` |
| 72 | `max_gain` forwards to `compile_settings.max_gain` |
| 73 | `safety` forwards to `compile_settings.safety` |
| 74 | `deadband_min` forwards to `compile_settings.deadband_min` |
| 75 | `deadband_max` forwards to `compile_settings.deadband_max` |
| 76 | `algorithm_order` returns `int(compile_settings.algorithm_order)` |
| 77 | `atol` forwards to `compile_settings.atol` |
| 78 | `rtol` forwards to `compile_settings.rtol` |
| 79 | `local_memory_elements` abstract property |

---

## `step_control/adaptive_I_controller.py`

### `AdaptiveIController`

| # | Functionality |
|---|--------------|
| 80 | `__init__` builds `AdaptiveStepControlConfig` via `build_config`, passes to super |
| 81 | `local_memory_elements` returns `0` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 82 | `order_exponent` computed as `1 / (2 * (1 + algorithm_order))` |
| 83 | Norm computed as mean squared scaled error: `sum((error_i / tol)^2) / n` |
| 84 | Step accepted when `nrm2 <= 1.0` |
| 85 | Step rejected when `nrm2 > 1.0` |
| 86 | Gain computed as `safety * nrm2^(-order_exponent)` |
| 87 | Gain clamped to `[min_gain, max_gain]` |
| 88 | Deadband: gain set to 1.0 when within `[deadband_min, deadband_max]` and deadband enabled |
| 89 | Deadband disabled path: no modification when `deadband_min == 1.0 and deadband_max == 1.0` |
| 90 | `dt` updated as `dt * gain`, clamped to `[dt_min, dt_max]` |
| 91 | Returns `0` when `dt_new_raw > dt_min` (normal) |
| 92 | Returns `8` when `dt_new_raw <= dt_min` (minimum step reached) |

---

## `step_control/adaptive_PI_controller.py`

### `PIStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 93 | `_kp` defaults `1/18`, validated as float |
| 94 | `_ki` defaults `1/9`, validated as float |
| 95 | `kp` returns `precision(self._kp)` |
| 96 | `ki` returns `precision(self._ki)` |

### `AdaptivePIController`

| # | Functionality |
|---|--------------|
| 97 | `__init__` builds `PIStepControlConfig` via `build_config`, passes to super |
| 98 | `kp` forwards to `compile_settings.kp` |
| 99 | `ki` forwards to `compile_settings.ki` |
| 100 | `local_memory_elements` returns `1` |
| 101 | `settings_dict` extends super with `kp` and `ki` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 102 | `kp` and `ki` scaled by `1 / (2 * (algorithm_order + 1))` |
| 103 | Allocates timestep_buffer from buffer_registry |
| 104 | Reads `err_prev` from `timestep_buffer[0]` |
| 105 | Norm computed as mean squared scaled error |
| 106 | Step accepted when `nrm2 <= 1.0` |
| 107 | Proportional gain: `nrm2^(-kp)` |
| 108 | Integral gain uses `err_prev` when initialized (> 0), else uses `nrm2` as fallback |
| 109 | Combined gain: `safety * pgain * igain` |
| 110 | Gain clamped to `[min_gain, max_gain]` |
| 111 | Deadband applied when enabled |
| 112 | `dt` updated and clamped |
| 113 | `timestep_buffer[0]` updated with current `nrm2` |
| 114 | Returns `0` (normal) or `8` (min step reached) |

---

## `step_control/adaptive_PID_controller.py`

### `PIDStepControlConfig` (attrs, extends PIStepControlConfig)

| # | Functionality |
|---|--------------|
| 115 | `_kd` defaults 0.0, validated as float |
| 116 | `kd` returns `precision(self._kd)` |

### `AdaptivePIDController`

| # | Functionality |
|---|--------------|
| 117 | `__init__` builds `PIDStepControlConfig` via `build_config`, passes to super |
| 118 | `kp` forwards to `compile_settings.kp` |
| 119 | `ki` forwards to `compile_settings.ki` |
| 120 | `kd` forwards to `compile_settings.kd` |
| 121 | `local_memory_elements` returns `2` |
| 122 | `settings_dict` extends super with `kp`, `ki`, `kd` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 123 | Exponents: `kp / (2*(order+1))`, `ki / (2*(order+1))`, `kd / (2*(order+1))` |
| 124 | Allocates timestep_buffer (2 slots) |
| 125 | Reads `err_prev` from `timestep_buffer[0]`, `err_prev_prev` from `timestep_buffer[1]` |
| 126 | `err_prev_safe` falls back to `nrm2` when uninitialized |
| 127 | `err_prev_prev_safe` falls back to `err_prev_safe` when uninitialized |
| 128 | Gain: `safety * nrm2^(-expo1) * err_prev_safe^(-expo2) * err_prev_prev_safe^(-expo3)` |
| 129 | Gain clamped, deadband applied |
| 130 | `timestep_buffer[1]` = previous `err_prev`, `timestep_buffer[0]` = current `nrm2` |
| 131 | Returns `0` or `8` |

---

## `step_control/gustafsson_controller.py`

### `GustafssonStepControlConfig` (attrs)

| # | Functionality |
|---|--------------|
| 132 | `_gamma` defaults 0.9, validated float in (0, 1) |
| 133 | `_newton_max_iters` defaults 20, validated int >= 0 |
| 134 | `gamma` returns `precision(self._gamma)` |
| 135 | `newton_max_iters` returns `int(self._newton_max_iters)` |
| 136 | `settings_dict` extends super with `gamma` and `newton_max_iters` |

### `GustafssonController`

| # | Functionality |
|---|--------------|
| 137 | `__init__` builds `GustafssonStepControlConfig`, passes to super |
| 138 | `gamma` forwards to `compile_settings.gamma` |
| 139 | `newton_max_iters` forwards to `compile_settings.newton_max_iters` |
| 140 | `local_memory_elements` returns `2` |

#### `build_controller` device function logic

| # | Functionality |
|---|--------------|
| 141 | `expo` = `1 / (2 * (algorithm_order + 1))` |
| 142 | `gain_numerator` = `(1 + 2 * newton_max_iters) * gamma` |
| 143 | Allocates timestep_buffer (2 slots: dt_prev, err_prev) |
| 144 | Reads `dt_prev` and `err_prev` from buffer, floored at 1e-16 |
| 145 | Norm uses 1e-12 floor on error (different from I/PI/PID which use 1e-16) |
| 146 | Step accepted when `nrm2 <= 1.0` |
| 147 | Basic gain: `fac * nrm2^(-expo)` where `fac = min(gamma, gain_numerator / (niters + 2*newton_max_iters))` |
| 148 | Gustafsson gain: `safety * (dt/dt_prev) * (nrm2^2/err_prev)^(-expo) * gamma` |
| 149 | Final gain: min of basic and Gustafsson when accepted and `dt_prev > 1e-16` |
| 150 | Falls back to basic gain when not accepted or `dt_prev` uninitialized |
| 151 | Gain clamped, deadband applied |
| 152 | Buffer updated: `timestep_buffer[0] = current_dt`, `timestep_buffer[1] = nrm2` |
| 153 | Returns `0` or `8` |

---

## `matrix_free_solvers/base_solver.py`

### `MatrixFreeSolverConfig` (attrs)

| # | Functionality |
|---|--------------|
| 154 | `n` defaults 0, validated int >= 1 |
| 155 | `max_iters` defaults 100, validated int in [1, 32767], metadata `prefixed=True` |
| 156 | `norm_device_function` defaults None, `eq=False` |

### `MatrixFreeSolver`

| # | Functionality |
|---|--------------|
| 157 | `__init__` stores `solver_type`, calls super with `instance_label=solver_type`, creates `ScaledNorm` |
| 158 | `atol` forwards to `norm.atol` |
| 159 | `rtol` forwards to `norm.rtol` |
| 160 | `max_iters` forwards to `compile_settings.max_iters` |
| 161 | `n` forwards to `compile_settings.n` |

#### `MatrixFreeSolver.update`

| # | Functionality |
|---|--------------|
| 162 | Merges `updates_dict` and `kwargs` |
| 163 | Empty dict returns empty set |
| 164 | Forwards to `norm.update(silent=True)`, captures recognised |
| 165 | Injects `norm_device_function` into updates |
| 166 | Forwards to `update_compile_settings(silent=True)` |
| 167 | Returns union of recognised sets |

---

## `matrix_free_solvers/linear_solver.py`

### `LinearSolverConfig` (attrs)

| # | Functionality |
|---|--------------|
| 168 | `operator_apply` defaults None, validated optional device, `eq=False` |
| 169 | `preconditioner` defaults None, validated optional device, `eq=False` |
| 170 | `linear_correction_type` defaults `"minimal_residual"`, validated in `["steepest_descent", "minimal_residual"]` |
| 171 | `preconditioned_vec_location` defaults `"local"` |
| 172 | `temp_location` defaults `"local"` |
| 173 | `use_cached_auxiliaries` defaults False |
| 174 | `settings_dict` returns dict with `krylov_max_iters`, `linear_correction_type`, location fields |

### `LinearSolverCache` (attrs)

| # | Functionality |
|---|--------------|
| 175 | `linear_solver` field validated as device function |

### `LinearSolver`

| # | Functionality |
|---|--------------|
| 176 | `__init__` builds `LinearSolverConfig` with `instance_label="krylov"`, calls super with `solver_type="krylov"`, sets up settings and registers buffers |
| 177 | `register_buffers` registers `"preconditioned_vec"` and `"temp"` with configured locations |

#### `build`

| # | Functionality |
|---|--------------|
| 178 | Branch: `use_cached_auxiliaries=True` produces `linear_solver_cached` with `cached_aux` parameter |
| 179 | Branch: `use_cached_auxiliaries=False` produces `linear_solver` without `cached_aux` |
| 180 | Both variants compute initial residual: `rhs = rhs - operator_apply(x)` |
| 181 | Early convergence check: `scaled_norm(rhs, x) <= 1.0` |
| 182 | Warp-synchronous early exit via `all_sync(mask, converged)` |
| 183 | Branch: `preconditioned=True` applies preconditioner to rhs |
| 184 | Branch: `preconditioned=False` copies rhs to preconditioned_vec |
| 185 | Branch: `sd_flag` (steepest descent) computes `numerator = rhs . z`, `denominator = (A*z) . z` |
| 186 | Branch: `mr_flag` (minimal residual) computes `numerator = (A*z) . rhs`, `denominator = (A*z) . (A*z)` |
| 187 | `alpha = numerator / denominator` when denominator nonzero |
| 188 | `alpha = 0` when denominator is zero |
| 189 | Solution update only when thread not already converged |
| 190 | `converged = converged or (norm <= 1.0)` |
| 191 | Returns status `0` on convergence, `4` on max iterations |
| 192 | `krylov_iters_out[0]` set to iteration count |

#### `update`

| # | Functionality |
|---|--------------|
| 193 | Delegates to `super().update(silent=True)` |
| 194 | Updates buffer locations via `buffer_registry.update` |
| 195 | Calls `register_buffers()` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 196 | `device_function` | `get_cached_output("linear_solver")` |
| 197 | `linear_correction_type` | `compile_settings.linear_correction_type` |
| 198 | `krylov_atol` | `self.atol` (-> `norm.atol`) |
| 199 | `krylov_rtol` | `self.rtol` (-> `norm.rtol`) |
| 200 | `krylov_max_iters` | `self.max_iters` |
| 201 | `use_cached_auxiliaries` | `compile_settings.use_cached_auxiliaries` |
| 202 | `settings_dict` | merges `compile_settings.settings_dict` + `krylov_atol` + `krylov_rtol` |

---

## `matrix_free_solvers/newton_krylov.py`

### `NewtonKrylovConfig` (attrs)

| # | Functionality |
|---|--------------|
| 203 | `residual_function` defaults None, `eq=False` |
| 204 | `linear_solver_function` defaults None, `eq=False` |
| 205 | `_newton_damping` defaults 0.5, validated float in (0, 1) |
| 206 | `newton_max_backtracks` defaults 8, validated int in [1, 32767] |
| 207 | `delta_location` defaults `"local"` |
| 208 | `residual_location` defaults `"local"` |
| 209 | `residual_temp_location` defaults `"local"` |
| 210 | `stage_base_bt_location` defaults `"local"` |
| 211 | `krylov_iters_local_location` defaults `"local"` |
| 212 | `newton_damping` returns `precision(self._newton_damping)` |
| 213 | `settings_dict` returns dict with newton_max_iters, newton_damping, newton_max_backtracks, location fields |

### `NewtonKrylovCache` (attrs)

| # | Functionality |
|---|--------------|
| 214 | `newton_krylov_solver` field validated as device function |

### `NewtonKrylov`

| # | Functionality |
|---|--------------|
| 215 | `__init__` builds `NewtonKrylovConfig` with `instance_label="newton"`, stores `linear_solver`, sets up settings and registers buffers |
| 216 | `register_buffers` registers `delta`, `residual`, `residual_temp`, `stage_base_bt` (all config.n), `krylov_iters_local` (size 1, precision `np_int32`) |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 217 | Computes initial residual via `residual_function`, negates it |
| 218 | Initial convergence check: `scaled_norm(residual, stage_increment) <= 1.0` |
| 219 | Warp-synchronous exit via `all_sync(mask, converged)` |
| 220 | Active threads increment `iters_count` via `selp` |
| 221 | Calls `linear_solver_fn` to solve for delta |
| 222 | Accumulates `total_krylov_iters` for active threads |
| 223 | Backtracking loop with `alpha` starting at 1.0, multiplied by damping each iteration |
| 224 | `max_backtracks = config.newton_max_backtracks + 1` (off-by-one correction) |
| 225 | Backtrack inner loop uses `any_sync` for warp-synchronous check |
| 226 | Convergence: `norm2_new <= 1.0` sets both `converged` and `found_step` |
| 227 | Sufficient decrease: `norm2_new < norm2_prev` sets `found_step`, updates residual and norm |
| 228 | Backtrack failure reverts `stage_increment` to `stage_base_bt` |
| 229 | Status bit `2` set when not converged at exit |
| 230 | Status bit `1` set when last backtrack failed |
| 231 | Status ORed with `last_lin_status` when linear solver signaled non-zero |
| 232 | `counters[0] = iters_count`, `counters[1] = total_krylov_iters` |

#### `update`

| # | Functionality |
|---|--------------|
| 233 | Forwards krylov-prefixed params to `linear_solver.update(silent=True)` |
| 234 | Injects `linear_solver_function` from linear solver's device_function |
| 235 | Delegates to `super().update(silent=True)` for norm and compile settings |
| 236 | Updates buffer locations via `buffer_registry.update` |
| 237 | Calls `register_buffers()` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 238 | `device_function` | `get_cached_output("newton_krylov_solver")` |
| 239 | `newton_atol` | `norm.atol` |
| 240 | `newton_rtol` | `norm.rtol` |
| 241 | `newton_max_iters` | `self.max_iters` |
| 242 | `newton_damping` | `compile_settings.newton_damping` |
| 243 | `newton_max_backtracks` | `compile_settings.newton_max_backtracks` |
| 244 | `krylov_atol` | `linear_solver.atol` |
| 245 | `krylov_rtol` | `linear_solver.rtol` |
| 246 | `krylov_max_iters` | `linear_solver.max_iters` |
| 247 | `linear_correction_type` | `linear_solver.linear_correction_type` |
| 248 | `settings_dict` | merges `linear_solver.settings_dict` + `compile_settings.settings_dict` + `newton_atol` + `newton_rtol` |

---

## `norms.py`

### `resize_tolerances` (module-level function)

| # | Functionality |
|---|--------------|
| 249 | Sets `_n_changing = True` on instance during resize |
| 250 | For each of `atol`, `rtol`: skips if length already matches `n` |
| 251 | Expands uniform tolerance arrays (all equal values) to new size `n` |
| 252 | Leaves non-uniform arrays unchanged |
| 253 | Sets `_n_changing = False` after resize |

### `ScaledNormConfig` (attrs)

| # | Functionality |
|---|--------------|
| 254 | `n` defaults 1, validated int >= 1, `on_setattr=resize_tolerances` |
| 255 | `atol` defaults `[1e-6]`, prefixed, converter `tol_converter` |
| 256 | `rtol` defaults `[1e-6]`, prefixed, converter `tol_converter` |
| 257 | `_n_changing` internal field, not in init, not in eq |
| 258 | `inv_n` returns `precision(1.0 / n)` |
| 259 | `tol_floor` returns `precision(1e-16)` |

### `ScaledNormCache` (attrs)

| # | Functionality |
|---|--------------|
| 260 | `scaled_norm` field validated as device function |

### `ScaledNorm`

| # | Functionality |
|---|--------------|
| 261 | `__init__` calls super with `instance_label`, builds `ScaledNormConfig`, sets up settings |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 262 | For each element: `tol_i = atol[i] + rtol[i] * |reference[i]|` |
| 263 | `tol_i` floored at `1e-16` to avoid division by zero |
| 264 | Computes `sum(|values[i]| / tol_i)^2 / n` |
| 265 | Returns mean squared scaled norm |

#### `update`

| # | Functionality |
|---|--------------|
| 266 | Merges updates_dict and kwargs |
| 267 | Empty dict returns empty set |
| 268 | Delegates to `update_compile_settings` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 269 | `device_function` | `get_cached_output("scaled_norm")` |
| 270 | `precision` | `compile_settings.precision` |
| 271 | `n` | `compile_settings.n` |
| 272 | `atol` | `compile_settings.atol` |
| 273 | `rtol` | `compile_settings.rtol` |

---

## `loops/ode_loop_config.py`

### `ODELoopConfig` (attrs)

| # | Functionality |
|---|--------------|
| 274 | `n_states` defaults 0, validated int >= 0 |
| 275 | `n_parameters` defaults 0 |
| 276 | `n_drivers` defaults 0 |
| 277 | `n_observables` defaults 0 |
| 278 | `n_error` defaults 0 |
| 279 | `n_counters` defaults 0 |
| 280 | `state_summaries_buffer_height` defaults 0 |
| 281 | `observable_summaries_buffer_height` defaults 0 |
| 282 | 14 buffer location fields, each defaults `"local"`, validated in `["shared", "local"]` |
| 283 | `compile_flags` factory `OutputCompileFlags` |
| 284 | `_save_every` defaults None, optional float > 0 |
| 285 | `_summarise_every` defaults None, optional float > 0 |
| 286 | `_sample_summaries_every` defaults None, optional float > 0 |
| 287 | `save_last` defaults False |
| 288 | `save_regularly` defaults False |
| 289 | `summarise_regularly` defaults False |
| 290 | 7 device function fields (save_state_fn, update_summaries_fn, save_summaries_fn, step_controller_fn, step_function, evaluate_driver_at_t, evaluate_observables), each optional, `eq=False` |
| 291 | `_dt0` defaults 0.01, optional float > 0 |
| 292 | `is_adaptive` defaults False |

#### `samples_per_summary` property

| # | Functionality |
|---|--------------|
| 293 | Returns 0 when either `summarise_every` or `sample_summaries_every` is None |
| 294 | Computes integer ratio `round(summarise_every / sample_summaries_every)` |
| 295 | Warning emitted when adjusted value differs from raw `_summarise_every` (deviation <= 0.01) |
| 296 | `ValueError` raised when deviation > 0.01 (not integer multiple) |

#### Precision properties

| # | Functionality |
|---|--------------|
| 297 | `save_every` returns `precision(self._save_every)` or None |
| 298 | `summarise_every` returns `precision(self._summarise_every)` or None |
| 299 | `sample_summaries_every` returns `precision(self._sample_summaries_every)` or None |
| 300 | `dt0` returns `precision(self._dt0)` |

---

## `loops/ode_loop.py`

### `IVPLoopCache` (attrs)

| # | Functionality |
|---|--------------|
| 301 | `loop_function` field |

### `IVPLoop`

| # | Functionality |
|---|--------------|
| 302 | `__init__` builds `ODELoopConfig` via `build_config` with all named + kwargs, calls `setup_compile_settings` and `register_buffers` |
| 303 | `register_buffers` registers 15 buffers: state, proposed_state, parameters, drivers, proposed_drivers, observables, proposed_observables, error, counters, state_summary, observable_summary, dt, accept_step, proposed_counters (int32 precision) |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 304 | Initialises `t` from `t0` in float64 |
| 305 | Clears `persistent_local` and `shared_scratch` to zero on entry |
| 306 | Copies initial_states and parameters into local buffers |
| 307 | Evaluates drivers at `t0` when `evaluate_driver_at_t is not None and n_drivers > 0` |
| 308 | Evaluates observables at `t0` when `n_observables > 0` |
| 309 | When `settling_time == 0`: saves initial state, advances next_save and next_update_summary |
| 310 | When `settling_time == 0` and `summarise`: calls `save_summaries` to reset buffers |
| 311 | Main loop: finish condition depends on `save_regularly`/`summarise_regularly` flags |
| 312 | When neither save nor summarise regularly: finishes when `end_of_step > t_end` |
| 313 | `save_last`: `at_end` triggers final save when `t_prec < t_end` and otherwise finished |
| 314 | `irrecoverable` forces finish |
| 315 | Warp-synchronous exit via `all_sync(mask, finished)` |
| 316 | `do_save` computed from `end_of_step >= next_save` when save_regularly |
| 317 | `do_update_summary` computed from `end_of_step >= next_update_summary` when summarise_regularly |
| 318 | `dt_eff` adjusted to hit output boundary exactly when saving or summarising |
| 319 | Step function called with all buffers |
| 320 | `first_step_flag` cleared after first step |
| 321 | Step status ORed into cumulative status |
| 322 | Fixed mode: step failure is irrecoverable |
| 323 | Adaptive mode: step failure forces error to 1e16 (rejection) |
| 324 | Controller called in adaptive mode; acceptance from `accept_step[0]` AND no step failure |
| 325 | Fixed mode: accept = not step_failed |
| 326 | Controller status bit `0x8` triggers irrecoverable |
| 327 | Counter accumulation when `save_counters_bool`: newton iters (i<2), total steps (i==2), rejected steps (i==3 and not accept) |
| 328 | Stagnation detection: 2 consecutive `t_proposal == t` triggers status `0x40` and irrecoverable |
| 329 | State, drivers, observables committed via `selp(accept, new, old)` |
| 330 | Output gated on `accept` |
| 331 | `next_save` incremented by `save_every` after save |
| 332 | Counters reset after save |
| 333 | `next_update_summary` incremented by `sample_summaries_every` |
| 334 | `save_summaries` called when `update_idx % samples_per_summary == 0` |
| 335 | `summary_idx` incremented after saving summaries |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 336 | `save_every` | `compile_settings.save_every` |
| 337 | `summarise_every` | `compile_settings.summarise_every` |
| 338 | `sample_summaries_every` | `compile_settings.sample_summaries_every` |
| 339 | `compile_flags` | `compile_settings.compile_flags` |
| 340 | `device_function` | `get_cached_output("loop_function")` |
| 341 | `save_state_fn` | `compile_settings.save_state_fn` |
| 342 | `update_summaries_fn` | `compile_settings.update_summaries_fn` |
| 343 | `save_summaries_fn` | `compile_settings.save_summaries_fn` |
| 344 | `step_controller_fn` | `compile_settings.step_controller_fn` |
| 345 | `step_function` | `compile_settings.step_function` |
| 346 | `evaluate_driver_at_t` | `compile_settings.evaluate_driver_at_t` |
| 347 | `evaluate_observables` | `compile_settings.evaluate_observables` |
| 348 | `dt0` | `compile_settings.dt0` |
| 349 | `is_adaptive` | `compile_settings.is_adaptive` |

#### `update`

| # | Functionality |
|---|--------------|
| 350 | Defaults `updates_dict` to `{}` when None |
| 351 | Merges kwargs |
| 352 | Empty dict returns empty set |
| 353 | Flattens nested dict values via `unpack_dict_values` |
| 354 | Delegates to `update_compile_settings(silent=True)` |
| 355 | Updates buffer locations via `buffer_registry.update(silent=True)` |
| 356 | Calls `register_buffers()` |
| 357 | `KeyError` raised for unrecognised keys when `silent=False` |
| 358 | No error when `silent=True` |
| 359 | Returns recognised union unpacked_keys |

---

## `array_interpolator.py`

### `InterpolatorCache` (attrs)

| # | Functionality |
|---|--------------|
| 360 | `evaluation_function` defaults None |
| 361 | `driver_del_t` defaults None |

### `ArrayInterpolatorConfig` (attrs)

| # | Functionality |
|---|--------------|
| 362 | `order` defaults 3, validated int > 0 |
| 363 | `wrap` defaults True, validated bool |
| 364 | `boundary_condition` defaults `"not-a-knot"`, validated in `{"natural", "periodic", "not-a-knot", "clamped"}` |
| 365 | `dt` init=False, defaults 1e-16, validated float > 0 |
| 366 | `t0` defaults 0.0, validated float >= 0 |
| 367 | `num_inputs` init=False, defaults 0 |
| 368 | `num_segments` init=False, defaults 0 |

### `ArrayInterpolator`

| # | Functionality |
|---|--------------|
| 369 | `__init__` creates config with precision only, stores `_coefficients=None`, `_input_array=None`, calls `update_from_dict` |

#### `update_from_dict`

| # | Functionality |
|---|--------------|
| 370 | Splits input_dict into config keys, input keys, and time keys |
| 371 | Updates compile settings with config keys |
| 372 | Normalises input array via `_normalise_input_array` |
| 373 | Returns False if input array unchanged |
| 374 | Validates time inputs via `_validate_time_inputs` |
| 375 | When `wrap=True` and no boundary_condition given: defaults to `"periodic"` |
| 376 | When `wrap=False` and no boundary_condition given: defaults to `"clamped"`, `num_segments = base + 2` |
| 377 | When `wrap=False` and boundary_condition `"clamped"`: `num_segments = base + 2` |
| 378 | When `wrap=False` and boundary_condition not clamped: `num_segments = base` |
| 379 | Computes coefficients via `_compute_coefficients` |
| 380 | Returns True when config or input changed |

#### `_normalise_input_array`

| # | Functionality |
|---|--------------|
| 381 | Converts each array to precision dtype |
| 382 | `ValueError` if array cannot be converted |
| 383 | `ValueError` if any array is not 1D |
| 384 | `ValueError` if arrays have different lengths |
| 385 | `ValueError` if fewer than `order + 1` samples |
| 386 | Returns column-stacked array |

#### `_validate_time_inputs`

| # | Functionality |
|---|--------------|
| 387 | `ValueError` if both `dt` and `time` provided |
| 388 | `dt` path: uses dt directly, t0 from dict or defaults to 0.0 |
| 389 | `time` path: `ValueError` if not 1D |
| 390 | `time` path: `ValueError` if length mismatch with num_samples |
| 391 | `time` path: `ValueError` if not strictly increasing |
| 392 | `time` path: `ValueError` if not uniformly spaced |
| 393 | `time` path: extracts dt from differences, t0 from first element |
| 394 | `ValueError` if neither dt nor time provided |

#### `build` device function logic

| # | Functionality |
|---|--------------|
| 395 | `evaluate_all`: wrapping path uses `idx % num_segments` |
| 396 | `evaluate_all`: non-wrapping path clamps segment index, returns zero outside range |
| 397 | `evaluate_all`: Horner's rule evaluation for each input polynomial |
| 398 | `evaluate_time_derivative`: same wrap/no-wrap logic |
| 399 | `evaluate_time_derivative`: derivative Horner's rule, scaled by `inv_resolution` |
| 400 | Clamped non-wrap: `evaluation_start` offset by `-resolution` |

#### `update`

| # | Functionality |
|---|--------------|
| 401 | Empty dict returns empty set |
| 402 | Delegates to `update_compile_settings(silent=True)` |
| 403 | `KeyError` for unrecognised keys when `silent=False` |

#### Properties (forwarding)

| # | Property | Delegates to |
|---|----------|-------------|
| 404 | `evaluation_function` | `get_cached_output("evaluation_function")` |
| 405 | `driver_del_t` | `get_cached_output("driver_del_t")` |
| 406 | `coefficients` | `self._coefficients` |
| 407 | `num_inputs` | `input_array.shape[1]` |
| 408 | `num_samples` | `input_array.shape[0]` |
| 409 | `input_array` | `self._input_array` |
| 410 | `order` | `compile_settings.order` |
| 411 | `wrap` | `compile_settings.wrap` |
| 412 | `boundary_condition` | `compile_settings.boundary_condition` |
| 413 | `num_segments` | `compile_settings.num_segments` |
| 414 | `t0` | `compile_settings.t0` |
| 415 | `dt` | `compile_settings.dt` |

#### `get_input_array`

| # | Functionality |
|---|--------------|
| 416 | Returns `self._input_array` |

#### `get_interpolated`

| # | Functionality |
|---|--------------|
| 417 | `ValueError` if `eval_times` not 1D |
| 418 | Returns empty array if `eval_times` is empty |
| 419 | `RuntimeError` if coefficients are None |
| 420 | Launches CUDA kernel to evaluate all times, returns host array |

#### `plot_interpolated`

| # | Functionality |
|---|--------------|
| 421 | `ImportError` if matplotlib not available |
| 422 | `ValueError` if `eval_times` not 1D |
| 423 | Wrapping mode: tiles sample markers across repeated periods |
| 424 | Non-wrapping mode: uses original sample times |
| 425 | Plots interpolated curves + sample markers for each input |
| 426 | Legend shown when `num_inputs > 1` |

#### `check_against_system_drivers` (static)

| # | Functionality |
|---|--------------|
| 427 | `ValueError` if number of inputs != number of system drivers |
| 428 | `ValueError` if input key set != system driver symbol set |

#### `_compute_coefficients`

| # | Functionality |
|---|--------------|
| 429 | `ValueError` for unsupported boundary condition |
| 430 | Clamped non-wrap: pads inputs with zero rows |
| 431 | Periodic: `ValueError` if `wrap=False` |
| 432 | Periodic: `ValueError` if first and last samples don't match |
| 433 | Builds tridiagonal-like system with function value constraints at both edges of each segment |
| 434 | Interior derivative continuity constraints for orders 1..order-1 |
| 435 | Natural BC: sets highest derivatives to zero at endpoints |
| 436 | Periodic BC: wraps derivative continuity from last to first segment |
| 437 | Clamped BC: sets first derivative to zero at endpoints |
| 438 | Not-a-knot BC: finite difference constraints at start and end |
| 439 | `ValueError` if assembled system is not square (row_index != num_coeffs) |
| 440 | Solves linear system, reshapes to (num_segments, num_inputs, order+1) |
