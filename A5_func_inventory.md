# A5 Functionality Inventory — Output Handling + Summary Metrics

---

## `outputhandling/output_config.py`

### `_indices_validator` — Module-Level Function

| # | Functionality |
|---|--------------|
| 1 | Returns without error when `array` is `None` |
| 2 | Raises `TypeError` when array is not an ndarray |
| 3 | Raises `TypeError` when array dtype is not `np.int_` |
| 4 | Raises `ValueError` when any index is negative |
| 5 | Raises `ValueError` when any index >= `max_index` |
| 6 | Raises `ValueError` listing duplicated indices when duplicates present |
| 7 | Passes without error for valid unique in-range array |

### `OutputCompileFlags` — attrs class

| # | Functionality |
|---|--------------|
| 8 | Construction with all defaults yields all-False flags |
| 9 | Each boolean attribute validated as `instance_of(bool)` |
| 10 | `__attrs_post_init__` calls super (inherits `_CubieConfigBase` hashing) |

### `OutputConfig.__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 11 | Calls `validation_passes()` after construction |

### `OutputConfig.validation_passes`

| # | Functionality |
|---|--------------|
| 12 | Calls `update_from_outputs_list` with current `_output_types` |
| 13 | Calls `_check_saved_indices` |
| 14 | Calls `_check_summarised_indices` |
| 15 | Calls `_validate_index_arrays` |
| 16 | Calls `_check_for_no_outputs` |

### `OutputConfig._validate_index_arrays`

| # | Functionality |
|---|--------------|
| 17 | Validates all four index arrays against their respective maxima |
| 18 | State indices validated against `_max_states` |
| 19 | Observable indices validated against `_max_observables` |

### `OutputConfig._check_for_no_outputs`

| # | Functionality |
|---|--------------|
| 20 | Raises `ValueError` when no output types are enabled |
| 21 | Passes when at least one of save_state, save_observables, save_time, save_counters, save_summaries is True |

### `OutputConfig._check_saved_indices`

| # | Functionality |
|---|--------------|
| 22 | Converts `_saved_state_indices` to numpy int array |
| 23 | Converts `_saved_observable_indices` to numpy int array |

### `OutputConfig._check_summarised_indices`

| # | Functionality |
|---|--------------|
| 24 | Converts `_summarised_state_indices` to numpy int array |
| 25 | Converts `_summarised_observable_indices` to numpy int array |

### `OutputConfig.max_states` — Property + Setter

| # | Functionality |
|---|--------------|
| 26 | Getter returns `_max_states` |
| 27 | Setter: when saved indices span full range, expands to new size |
| 28 | Setter: when saved indices do NOT span full range, leaves them unchanged |
| 29 | Setter: updates `_max_states` and re-runs `__attrs_post_init__` |

### `OutputConfig.max_observables` — Property + Setter

| # | Functionality |
|---|--------------|
| 30 | Getter returns `_max_observables` |
| 31 | Setter: when saved observable indices span full range, expands to new size |
| 32 | Setter: when saved observable indices do NOT span full range, leaves them unchanged |
| 33 | Setter: updates `_max_observables` and re-runs `__attrs_post_init__` |

### `OutputConfig.save_state` — Calculated Property

| # | Functionality |
|---|--------------|
| 34 | Returns True when `_save_state` is True AND saved state indices non-empty |
| 35 | Returns False when `_save_state` is True AND saved state indices empty |
| 36 | Returns False when `_save_state` is False |

### `OutputConfig.save_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 37 | Returns True when `_save_observables` is True AND saved observable indices non-empty |
| 38 | Returns False when `_save_observables` is True AND observable indices empty |
| 39 | Returns False when `_save_observables` is False |

### `OutputConfig.save_time` — Forwarding Property

| # | Functionality |
|---|--------------|
| 40 | Returns `_save_time` |

### `OutputConfig.save_counters` — Forwarding Property

| # | Functionality |
|---|--------------|
| 41 | Returns `_save_counters` |

### `OutputConfig.save_summaries` — Calculated Property

| # | Functionality |
|---|--------------|
| 42 | Returns True when `_summary_types` is non-empty |
| 43 | Returns False when `_summary_types` is empty |

### `OutputConfig.summarise_state` — Calculated Property

| # | Functionality |
|---|--------------|
| 44 | Returns True when save_summaries True AND n_summarised_states > 0 |
| 45 | Returns False when save_summaries False |
| 46 | Returns False when n_summarised_states == 0 |

### `OutputConfig.summarise_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 47 | Returns True when save_summaries True AND n_summarised_observables > 0 |
| 48 | Returns False when save_summaries False |
| 49 | Returns False when n_summarised_observables == 0 |

### `OutputConfig.compile_flags` — Calculated Property

| # | Functionality |
|---|--------------|
| 50 | Returns `OutputCompileFlags` with fields derived from current config |
| 51 | `save_state` flag matches `self.save_state` |
| 52 | `save_observables` flag matches `self.save_observables` |
| 53 | `summarise` flag matches `self.save_summaries` |
| 54 | `summarise_observables` flag matches `self.summarise_observables` |
| 55 | `summarise_state` flag matches `self.summarise_state` |
| 56 | `save_counters` flag matches `self.save_counters` |

### `OutputConfig.saved_state_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 57 | Getter returns empty array when `_save_state` is False |
| 58 | Getter returns `_saved_state_indices` when `_save_state` is True |
| 59 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.saved_observable_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 60 | Getter returns empty array when `_save_observables` is False |
| 61 | Getter returns `_saved_observable_indices` when `_save_observables` is True |
| 62 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.summarised_state_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 63 | Getter returns empty array when `save_summaries` is False |
| 64 | Getter returns `_summarised_state_indices` when `save_summaries` is True |
| 65 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.summarised_observable_indices` — Property + Setter

| # | Functionality |
|---|--------------|
| 66 | Getter returns empty array when `save_summaries` is False |
| 67 | Getter returns `_summarised_observable_indices` when `save_summaries` is True |
| 68 | Setter converts to numpy int array, validates, checks no-outputs |

### `OutputConfig.n_saved_states` — Calculated Property

| # | Functionality |
|---|--------------|
| 69 | Returns length of `_saved_state_indices` when `_save_state` True |
| 70 | Returns 0 when `_save_state` False |

### `OutputConfig.n_saved_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 71 | Returns length of `_saved_observable_indices` when `_save_observables` True |
| 72 | Returns 0 when `_save_observables` False |

### `OutputConfig.n_summarised_states` — Calculated Property

| # | Functionality |
|---|--------------|
| 73 | Returns length of `_summarised_state_indices` when `save_summaries` True |
| 74 | Returns 0 when `save_summaries` False |

### `OutputConfig.n_summarised_observables` — Calculated Property

| # | Functionality |
|---|--------------|
| 75 | Returns length of `_summarised_observable_indices` when `save_summaries` True |
| 76 | Returns 0 when `save_summaries` False |

### `OutputConfig.summary_types` — Forwarding Property

| # | Functionality |
|---|--------------|
| 77 | Returns `_summary_types` tuple |

### `OutputConfig.summary_legend_per_variable` — Calculated Property

| # | Functionality |
|---|--------------|
| 78 | Returns empty dict when `_summary_types` is empty |
| 79 | Returns dict mapping indices to metric legend strings when types present |

### `OutputConfig.summary_unit_modifications` — Calculated Property

| # | Functionality |
|---|--------------|
| 80 | Returns empty dict when `_summary_types` is empty |
| 81 | Returns dict mapping indices to unit modification strings when types present |

### `OutputConfig.sample_summaries_every` — Forwarding Property

| # | Functionality |
|---|--------------|
| 82 | Returns `_sample_summaries_every` |

### `OutputConfig.summaries_buffer_height_per_var` — Calculated Property

| # | Functionality |
|---|--------------|
| 83 | Returns 0 when `summary_types` is empty |
| 84 | Delegates to `summary_metrics.summaries_buffer_height` when types present |

### `OutputConfig.summaries_output_height_per_var` — Calculated Property

| # | Functionality |
|---|--------------|
| 85 | Returns 0 when `_summary_types` is empty |
| 86 | Delegates to `summary_metrics.summaries_output_height` when types present |

### `OutputConfig.state_summaries_buffer_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 87 | Returns `summaries_buffer_height_per_var * n_summarised_states` |

### `OutputConfig.observable_summaries_buffer_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 88 | Returns `summaries_buffer_height_per_var * n_summarised_observables` |

### `OutputConfig.total_summary_buffer_size` — Calculated Property

| # | Functionality |
|---|--------------|
| 89 | Returns sum of state and observable summary buffer heights |

### `OutputConfig.state_summaries_output_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 90 | Returns `summaries_output_height_per_var * n_summarised_states` |

### `OutputConfig.observable_summaries_output_height` — Calculated Property

| # | Functionality |
|---|--------------|
| 91 | Returns `summaries_output_height_per_var * n_summarised_observables` |

### `OutputConfig.buffer_sizes_dict` — Calculated Property

| # | Functionality |
|---|--------------|
| 92 | Returns dict with keys: n_saved_states, n_saved_observables, n_summarised_states, n_summarised_observables, state/observable buffer heights, total_summary_buffer_size, state/observable output heights, compile_flags |

### `OutputConfig.output_types` — Property + Setter

| # | Functionality |
|---|--------------|
| 93 | Getter returns `_output_types` list |
| 94 | Setter accepts tuple (converts to list) |
| 95 | Setter accepts string (wraps in list) |
| 96 | Setter raises `TypeError` for unsupported type |
| 97 | Setter calls `update_from_outputs_list` then `_check_for_no_outputs` |

### `OutputConfig.update_from_outputs_list`

| # | Functionality |
|---|--------------|
| 98 | Empty list: clears all flags and summary types |
| 99 | Non-empty: sets `_save_state` True when "state" in list |
| 100 | Non-empty: sets `_save_observables` True when "observables" in list |
| 101 | Non-empty: sets `_save_time` True when "time" in list |
| 102 | Non-empty: sets `_save_counters` True when "iteration_counters" in list |
| 103 | Non-empty: collects summary types matching `implemented_metrics` prefixes |
| 104 | Non-empty: warns for unrecognised output types |
| 105 | Non-empty: calls `_check_for_no_outputs` |

### `OutputConfig.from_loop_settings` — Classmethod

| # | Functionality |
|---|--------------|
| 106 | Converts None indices to empty numpy arrays |
| 107 | Copies output_types list before modifying |
| 108 | Passes all parameters through to constructor |

---

## `outputhandling/output_sizes.py`

### `ArraySizingClass.nonzero` — Property

| # | Functionality |
|---|--------------|
| 109 | Returns new object with all int/tuple fields coerced to at least 1 |
| 110 | Does not mutate the original object |

### `OutputArrayHeights` — attrs class

| # | Functionality |
|---|--------------|
| 111 | Construction with defaults yields all-1 heights |

### `OutputArrayHeights.from_output_fns` — Classmethod

| # | Functionality |
|---|--------------|
| 112 | state height = n_saved_states + 1 when save_time True |
| 113 | state height = n_saved_states when save_time False |
| 114 | observables height = n_saved_observables |
| 115 | state_summaries = state_summaries_output_height |
| 116 | observable_summaries = observable_summaries_output_height |
| 117 | per_variable = summaries_output_height_per_var |

### `SingleRunOutputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 118 | state shape = (output_samples, heights.state) |
| 119 | observables shape = (output_samples, heights.observables) |
| 120 | state_summaries shape = (summarise_samples, heights.state_summaries) |
| 121 | observable_summaries shape = (summarise_samples, heights.observable_summaries) |

### `BatchInputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 122 | initial_values = (states, num_runs) |
| 123 | parameters = (parameters, num_runs) |
| 124 | driver_coefficients = (None, drivers, None) |

### `BatchOutputSizes.from_solver` — Classmethod

| # | Functionality |
|---|--------------|
| 125 | Adds num_runs as third dimension to all single-run shapes |
| 126 | status_codes = (num_runs,) |
| 127 | iteration_counters = (n_saves, 4, num_runs) |

---

## `outputhandling/output_functions.py`

### `OutputFunctionCache` — attrs class

| # | Functionality |
|---|--------------|
| 128 | Stores save_state_function, update_summaries_function, save_summaries_function |

### `OutputFunctions.__init__`

| # | Functionality |
|---|--------------|
| 129 | Defaults `output_types` to `["state"]` when None |
| 130 | Creates `OutputConfig.from_loop_settings` with all parameters |
| 131 | Calls `setup_compile_settings` with the config |

### `OutputFunctions.update`

| # | Functionality |
|---|--------------|
| 132 | Merges `updates_dict` and `kwargs` |
| 133 | Returns empty set for empty dict |
| 134 | Delegates to `update_compile_settings(silent=True)` |
| 135 | Calls `validation_passes()` after update |
| 136 | Raises `KeyError` for unrecognised params when `silent=False` |
| 137 | Suppresses KeyError when `silent=True` |

### `OutputFunctions.build`

| # | Functionality |
|---|--------------|
| 138 | Updates `summary_metrics` with sample_summaries_every and precision |
| 139 | Calls `save_state_factory` with config indices and flags |
| 140 | Calls `update_summary_factory` with config summary parameters |
| 141 | Calls `save_summary_factory` with config summary parameters |
| 142 | Returns `OutputFunctionCache` with all three functions |

### `OutputFunctions` — Forwarding Properties

| # | Property | Delegates to |
|---|----------|-------------|
| 143 | `save_state_func` | `get_cached_output("save_state_function")` |
| 144 | `update_summaries_func` | `get_cached_output("update_summaries_function")` |
| 145 | `save_summary_metrics_func` | `get_cached_output("save_summaries_function")` |
| 146 | `output_types` | `compile_settings.output_types` |
| 147 | `compile_flags` | `compile_settings.compile_flags` |
| 148 | `save_time` | `compile_settings.save_time` |
| 149 | `saved_state_indices` | `compile_settings.saved_state_indices` |
| 150 | `saved_observable_indices` | `compile_settings.saved_observable_indices` |
| 151 | `summarised_state_indices` | `compile_settings.summarised_state_indices` |
| 152 | `summarised_observable_indices` | `compile_settings.summarised_observable_indices` |
| 153 | `n_saved_states` | `compile_settings.n_saved_states` |
| 154 | `n_saved_observables` | `compile_settings.n_saved_observables` |
| 155 | `state_summaries_output_height` | `compile_settings.state_summaries_output_height` |
| 156 | `observable_summaries_output_height` | `compile_settings.observable_summaries_output_height` |
| 157 | `summaries_buffer_height_per_var` | `compile_settings.summaries_buffer_height_per_var` |
| 158 | `state_summaries_buffer_height` | `compile_settings.state_summaries_buffer_height` |
| 159 | `observable_summaries_buffer_height` | `compile_settings.observable_summaries_buffer_height` |
| 160 | `summaries_output_height_per_var` | `compile_settings.summaries_output_height_per_var` |
| 161 | `summary_legend_per_variable` | `compile_settings.summary_legend_per_variable` |
| 162 | `summary_unit_modifications` | `compile_settings.summary_unit_modifications` |
| 163 | `buffer_sizes_dict` | `compile_settings.buffer_sizes_dict` |

### `OutputFunctions.has_time_domain_outputs` — Calculated Property

| # | Functionality |
|---|--------------|
| 164 | Returns True when save_time True |
| 165 | Returns True when save_state True |
| 166 | Returns True when save_observables True |
| 167 | Returns False when all three False |

### `OutputFunctions.has_summary_outputs` — Calculated Property

| # | Functionality |
|---|--------------|
| 168 | Returns True when summarise_state True |
| 169 | Returns True when summarise_observables True |
| 170 | Returns False when neither is True |

### `OutputFunctions.output_array_heights` — Calculated Property

| # | Functionality |
|---|--------------|
| 171 | Returns `OutputArrayHeights.from_output_fns(self)` |

---

## `outputhandling/save_state.py`

### `save_state_factory`

| # | Functionality |
|---|--------------|
| 172 | Generated function copies selected states when `save_state` True |
| 173 | Generated function skips state copy when `save_state` False |
| 174 | Generated function appends time at position nstates when `save_time` True |
| 175 | Generated function skips time when `save_time` False |
| 176 | Generated function copies selected observables when `save_observables` True |
| 177 | Generated function skips observables when `save_observables` False |
| 178 | Generated function copies 4 counters when `save_counters` True |
| 179 | Generated function skips counters when `save_counters` False |
| 180 | State indices map correctly (uses `saved_state_indices[k]`) |
| 181 | Observable indices map correctly (uses `saved_observable_indices[m]`) |

---

## `outputhandling/save_summaries.py`

### `do_nothing` — Module-level CUDA function

| # | Functionality |
|---|--------------|
| 182 | No-op base case for empty metric chains |

### `chain_metrics`

| # | Functionality |
|---|--------------|
| 183 | Returns `do_nothing` when metric_functions is empty |
| 184 | Single metric: returns wrapper calling inner_chain then metric |
| 185 | Multiple metrics: recursively chains remaining metrics |
| 186 | Each wrapper slices buffer using offset and size |
| 187 | Each wrapper slices output using offset and size |
| 188 | Passes metric-specific params through to each function |

### `save_summary_factory`

| # | Functionality |
|---|--------------|
| 189 | Iterates over state variables when `summarise_states` True |
| 190 | Skips state variables when `summarise_states` False |
| 191 | Iterates over observable variables when `summarise_observables` True |
| 192 | Skips observable variables when `summarise_observables` False |
| 193 | Each variable gets buffer slice at `idx * total_buffer_size` |
| 194 | Each variable gets output slice at `idx * total_output_size` |
| 195 | Passes `summarise_every` through to chain |

---

## `outputhandling/update_summaries.py`

### `do_nothing` — Module-level CUDA function

| # | Functionality |
|---|--------------|
| 196 | No-op base case for empty metric chains |

### `chain_metrics`

| # | Functionality |
|---|--------------|
| 197 | Returns `do_nothing` when metric_functions is empty |
| 198 | Single metric: returns wrapper calling inner_chain then metric |
| 199 | Multiple metrics: recursively chains remaining metrics |
| 200 | Each wrapper slices buffer using offset and size |
| 201 | Passes metric-specific params through |

### `update_summary_factory`

| # | Functionality |
|---|--------------|
| 202 | Iterates over state variables when `summarise_states` True |
| 203 | Skips state variables when `summarise_states` False |
| 204 | Iterates over observable variables when `summarise_observables` True |
| 205 | Skips observable variables when `summarise_observables` False |
| 206 | Each variable gets buffer slice at `idx * total_buffer_size` |
| 207 | Passes current_state value for state vars via `summarised_state_indices[idx]` |
| 208 | Passes current_observables value for obs vars via `summarised_observable_indices[idx]` |

---

## `summarymetrics/metrics.py`

### `MetricFuncCache` — attrs class

| # | Functionality |
|---|--------------|
| 209 | Stores `update` and `save` callables with None defaults |

### `MetricConfig` — attrs class

| # | Functionality |
|---|--------------|
| 210 | `sample_summaries_every` defaults to 0.01, validated > 0 or None |
| 211 | Inherits precision from `CUDAFactoryConfig` |

### `register_metric` — Decorator factory

| # | Functionality |
|---|--------------|
| 212 | Instantiates the decorated class with `registry.precision` |
| 213 | Calls `registry.register_metric(instance)` |
| 214 | Returns the original class (not the instance) |

### `SummaryMetric.__init__`

| # | Functionality |
|---|--------------|
| 215 | Stores buffer_size, output_size, name, unit_modification |
| 216 | Creates MetricConfig with sample_summaries_every and precision |
| 217 | Sets up compile settings via `setup_compile_settings` |

### `SummaryMetric.update_device_func` — Forwarding Property

| # | Functionality |
|---|--------------|
| 218 | Returns `get_cached_output("update")` |

### `SummaryMetric.save_device_func` — Forwarding Property

| # | Functionality |
|---|--------------|
| 219 | Returns `get_cached_output("save")` |

### `SummaryMetric.update`

| # | Functionality |
|---|--------------|
| 220 | Delegates to `update_compile_settings(kwargs, silent=True)` |

### `SummaryMetric.build` — Abstract

| # | Functionality |
|---|--------------|
| 221 | Abstract method, must be overridden by subclasses |

### `SummaryMetrics.__attrs_post_init__`

| # | Functionality |
|---|--------------|
| 222 | Resets `_params` to empty dict |
| 223 | Defines `_combined_metrics` mapping frozensets to combined names |

### `SummaryMetrics.update`

| # | Functionality |
|---|--------------|
| 224 | Updates `self.precision` if "precision" in kwargs |
| 225 | Propagates kwargs to all registered metric objects |

### `SummaryMetrics.register_metric`

| # | Functionality |
|---|--------------|
| 226 | Raises `ValueError` for duplicate metric name |
| 227 | Appends name to `_names` list |
| 228 | Stores buffer_size, output_size, metric object, and default param (0) |

### `SummaryMetrics._apply_combined_metrics`

| # | Functionality |
|---|--------------|
| 229 | Substitutes {mean, std, rms} with mean_std_rms when all three requested |
| 230 | Substitutes {mean, std} with mean_std when both requested |
| 231 | Substitutes {std, rms} with std_rms when both requested |
| 232 | Substitutes {max, min} with extrema when both requested |
| 233 | Substitutes {dxdt_max, dxdt_min} with dxdt_extrema when both requested |
| 234 | Substitutes {d2xdt2_max, d2xdt2_min} with d2xdt2_extrema when both requested |
| 235 | Prefers larger combinations (sorted by size descending) |
| 236 | Preserves original order of metrics |
| 237 | No substitution when only one of a pair is requested |
| 238 | Skips combined metric if not registered |

### `SummaryMetrics.preprocess_request`

| # | Functionality |
|---|--------------|
| 239 | Calls `parse_string_for_params` to extract parameters |
| 240 | Calls `_apply_combined_metrics` for substitution |
| 241 | Warns and removes unregistered metrics |
| 242 | Returns validated list of metric names |

### `SummaryMetrics.implemented_metrics` — Property

| # | Functionality |
|---|--------------|
| 243 | Returns `_names` list |

### `SummaryMetrics.summaries_buffer_height`

| # | Functionality |
|---|--------------|
| 244 | Sums buffer sizes for all preprocessed metrics |

### `SummaryMetrics.buffer_offsets`

| # | Functionality |
|---|--------------|
| 245 | Returns cumulative buffer offsets for each metric |

### `SummaryMetrics.buffer_sizes`

| # | Functionality |
|---|--------------|
| 246 | Returns tuple of buffer sizes for each metric |

### `SummaryMetrics.output_offsets`

| # | Functionality |
|---|--------------|
| 247 | Returns cumulative output offsets for each metric |

### `SummaryMetrics.summaries_output_height`

| # | Functionality |
|---|--------------|
| 248 | Sums output sizes for all preprocessed metrics |

### `SummaryMetrics._get_size`

| # | Functionality |
|---|--------------|
| 249 | Returns int directly when size is not callable |
| 250 | Calls size(param) when size is callable |
| 251 | Warns when callable size has param == 0 |

### `SummaryMetrics.legend`

| # | Functionality |
|---|--------------|
| 252 | Single-element metrics get name as heading |
| 253 | Multi-element metrics get `{name}_1`, `{name}_2`, etc. |

### `SummaryMetrics.unit_modifications`

| # | Functionality |
|---|--------------|
| 254 | Returns one unit modification per output element |
| 255 | Multi-element outputs repeat the same modification |

### `SummaryMetrics.output_sizes`

| # | Functionality |
|---|--------------|
| 256 | Returns tuple of output sizes for each metric |

### `SummaryMetrics.save_functions`

| # | Functionality |
|---|--------------|
| 257 | Returns tuple of save_device_func for each preprocessed metric |

### `SummaryMetrics.update_functions`

| # | Functionality |
|---|--------------|
| 258 | Returns tuple of update_device_func for each preprocessed metric |

### `SummaryMetrics.params`

| # | Functionality |
|---|--------------|
| 259 | Returns tuple of parameter values for each preprocessed metric |

### `SummaryMetrics.parse_string_for_params`

| # | Functionality |
|---|--------------|
| 260 | Extracts `[N]` parameter from metric string |
| 261 | Raises `ValueError` for non-integer parameter |
| 262 | Stores parsed param in `_params` dict |
| 263 | Metrics without `[N]` get default param 0 |
| 264 | Resets `_params` on each call |

---

## Individual Metric Files

All metrics follow the same pattern: `__init__` sets name/buffer_size/output_size/unit_modification, `build()` returns `MetricFuncCache(update, save)`.

### `mean.py` — Mean

| # | Functionality |
|---|--------------|
| 265 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 266 | update: accumulates sum in buffer[0] |
| 267 | save: divides by summarise_every, resets to 0.0 |

### `max.py` — Max

| # | Functionality |
|---|--------------|
| 268 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 269 | update: replaces buffer[0] when value > buffer[0] |
| 270 | update: no-op when value <= buffer[0] |
| 271 | save: copies buffer[0] to output, resets to -1e30 |

### `min.py` — Min

| # | Functionality |
|---|--------------|
| 272 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 273 | update: replaces buffer[0] when value < buffer[0] |
| 274 | update: no-op when value >= buffer[0] |
| 275 | save: copies buffer[0] to output, resets to 1e30 |

### `std.py` — Std

| # | Functionality |
|---|--------------|
| 276 | buffer_size=3, output_size=1, unit_modification="[unit]" |
| 277 | update: stores shift value on first sample (current_index==0), resets accumulators |
| 278 | update: accumulates shifted sum and shifted sum-of-squares |
| 279 | save: computes variance = mean_sq_shifted - mean_shifted^2, outputs sqrt(variance) |
| 280 | save: updates shift to mean for next period, resets accumulators |

### `rms.py` — RMS

| # | Functionality |
|---|--------------|
| 281 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 282 | update: resets sum_of_squares on first step (current_index==0) |
| 283 | update: adds value^2 to sum_of_squares |
| 284 | save: outputs sqrt(sum_of_squares / summarise_every), resets to 0.0 |

### `peaks.py` — Peaks

| # | Functionality |
|---|--------------|
| 285 | buffer_size=lambda n: 3+n, output_size=lambda n: n, unit_modification="s" |
| 286 | update: detects peaks when prev > value AND prev_prev < prev, with guards (index>=2, counter<npeaks, prev_prev!=0) |
| 287 | update: stores peak index at buffer[3+counter], increments counter |
| 288 | update: shifts prev/prev_prev history |
| 289 | save: copies n_peaks indices from buffer[3:] to output, resets |

### `negative_peaks.py` — NegativePeaks

| # | Functionality |
|---|--------------|
| 290 | buffer_size=lambda n: 3+n, output_size=lambda n: n, unit_modification="s" |
| 291 | update: detects negative peaks when prev < value AND prev_prev > prev, with guards |
| 292 | update: stores peak index, increments counter |
| 293 | save: copies indices and resets |

### `extrema.py` — Extrema

| # | Functionality |
|---|--------------|
| 294 | buffer_size=2, output_size=2, unit_modification="[unit]" |
| 295 | update: tracks max in buffer[0] and min in buffer[1] |
| 296 | save: outputs [max, min], resets to [-1e30, 1e30] |

### `max_magnitude.py` — MaxMagnitude

| # | Functionality |
|---|--------------|
| 297 | buffer_size=1, output_size=1, unit_modification="[unit]" |
| 298 | update: replaces buffer[0] when abs(value) > buffer[0] |
| 299 | save: copies buffer[0] to output, resets to 0.0 |

### `mean_std.py` — MeanStd

| # | Functionality |
|---|--------------|
| 300 | buffer_size=3, output_size=2, unit_modification="[unit]" |
| 301 | update: shifted data accumulation (same as std) |
| 302 | save: outputs [mean, std], resets with shift=mean |

### `mean_std_rms.py` — MeanStdRms

| # | Functionality |
|---|--------------|
| 303 | buffer_size=3, output_size=3, unit_modification="[unit]" |
| 304 | update: shifted data accumulation |
| 305 | save: outputs [mean, std, rms], resets with shift=mean |
| 306 | save: RMS uses E[X^2] = E[(X-shift)^2] + 2*shift*E[X-shift] + shift^2 |

### `std_rms.py` — StdRms

| # | Functionality |
|---|--------------|
| 307 | buffer_size=3, output_size=2, unit_modification="[unit]" |
| 308 | update: shifted data accumulation |
| 309 | save: outputs [std, rms], resets with shift=mean |

### `dxdt_max.py` — DxdtMax

| # | Functionality |
|---|--------------|
| 310 | buffer_size=2, output_size=1, unit_modification="[unit]*s^-1" |
| 311 | update: computes unscaled derivative (value - prev), updates max with predicated commit, guard on prev!=0 |
| 312 | save: scales by 1/sample_summaries_every, resets to -1e30 |

### `dxdt_min.py` — DxdtMin

| # | Functionality |
|---|--------------|
| 313 | buffer_size=2, output_size=1, unit_modification="[unit]*s^-1" |
| 314 | update: computes unscaled derivative, updates min with predicated commit, guard on prev!=0 |
| 315 | save: scales by 1/sample_summaries_every, resets to 1e30 |

### `dxdt_extrema.py` — DxdtExtrema

| # | Functionality |
|---|--------------|
| 316 | buffer_size=3, output_size=2, unit_modification="[unit]*s^-1" |
| 317 | update: computes unscaled derivative, updates max (buffer[1]) and min (buffer[2]) with predicated commit |
| 318 | save: scales both by 1/sample_summaries_every, outputs [max, min], resets sentinels |

### `d2xdt2_max.py` — D2xdt2Max

| # | Functionality |
|---|--------------|
| 319 | buffer_size=3, output_size=1, unit_modification="[unit]*s^-2" |
| 320 | update: computes central difference (value - 2*prev + prev_prev), updates max, guard on prev_prev!=0 |
| 321 | save: scales by 1/sample_summaries_every^2, resets to -1e30 |

### `d2xdt2_min.py` — D2xdt2Min

| # | Functionality |
|---|--------------|
| 322 | buffer_size=3, output_size=1, unit_modification="[unit]*s^-2" |
| 323 | update: computes central difference, updates min, guard on prev_prev!=0 |
| 324 | save: scales by 1/sample_summaries_every^2, resets to 1e30 |

### `d2xdt2_extrema.py` — D2xdt2Extrema

| # | Functionality |
|---|--------------|
| 325 | buffer_size=4, output_size=2, unit_modification="[unit]*s^-2" |
| 326 | update: computes central difference, updates max (buffer[2]) and min (buffer[3]), guard on prev_prev!=0 |
| 327 | save: scales both by 1/sample_summaries_every^2, outputs [max, min], resets sentinels |

---

**Total inventory items: 327**
