# CuBIE Name Tracking

## File: src/cubie/integrators/loops/__init__.py

No methods, properties, functions, or attributes to track (only imports).

## File: src/cubie/integrators/loops/ode_loop.py

### IVPLoop.__init__
**Location**: src/cubie/integrators/loops/ode_loop.py:80
**Type**: method
**Current Name**: __init__
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop.build
**Location**: src/cubie/integrators/loops/ode_loop.py:140
**Type**: method
**Current Name**: build
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop.update
**Location**: src/cubie/integrators/loops/ode_loop.py:645
**Type**: method
**Current Name**: update
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop precision properties group
**Location**: src/cubie/integrators/loops/ode_loop.py:124, 129, 135
**Type**: property
**Current Name**: precision, numba_precision, simsafe_precision
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop timestep properties group
**Location**: src/cubie/integrators/loops/ode_loop.py:534, 540, 622, 628, 634
**Type**: property
**Current Name**: dt_save, dt_summarise, dt0, dt_min, dt_max
**Recommended Rename**: [save_interval, summary_interval, initial_timestep, minimum_timestep, maximum_timestep]
**Rename Executed**: []

### IVPLoop.is_adaptive
**Location**: src/cubie/integrators/loops/ode_loop.py:640
**Type**: property
**Current Name**: is_adaptive
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop buffer indices properties group
**Location**: src/cubie/integrators/loops/ode_loop.py:546, 552, 558
**Type**: property
**Current Name**: shared_buffer_indices, buffer_indices, local_indices
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop memory requirement properties group
**Location**: src/cubie/integrators/loops/ode_loop.py:564, 569
**Type**: property
**Current Name**: shared_memory_elements, local_memory_elements
**Recommended Rename**: [shared_memory_element_count, local_memory_element_count]
**Rename Executed**: []

### IVPLoop.compile_flags
**Location**: src/cubie/integrators/loops/ode_loop.py:574
**Type**: property
**Current Name**: compile_flags
**Recommended Rename**: [NA]
**Rename Executed**: []

### IVPLoop device function properties group
**Location**: src/cubie/integrators/loops/ode_loop.py:580, 586, 592, 598, 604, 610, 616
**Type**: property
**Current Name**: save_state_fn, update_summaries_fn, save_summaries_fn, step_controller_fn, step_function, driver_function, observables_fn
**Recommended Rename**: [state_saving_function, summary_update_function, summary_saving_function, step_controller_function, step_function, driver_evaluation_function, observables_function]
**Rename Executed**: [2025-11-12: completed (partial - save_state_fn, update_summaries_fn, save_summaries_fn, observables_fn renamed; step_controller_fn, step_function, driver_function remain)]

## File: src/cubie/integrators/loops/ode_loop_config.py

### LoopLocalIndices.empty
**Location**: src/cubie/integrators/loops/ode_loop_config.py:66
**Type**: method
**Current Name**: empty
**Recommended Rename**: [NA]
**Rename Executed**: []

### LoopLocalIndices.from_sizes
**Location**: src/cubie/integrators/loops/ode_loop_config.py:87
**Type**: method
**Current Name**: from_sizes
**Recommended Rename**: [NA]
**Rename Executed**: []

### LoopLocalIndices buffer slice attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:51, 52, 53, 56
**Type**: attribute
**Current Name**: dt, accept, controller, algorithm
**Recommended Rename**: [timestep_slice, acceptance_flag_slice, controller_state_slice, algorithm_state_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopLocalIndices buffer end attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:57, 60
**Type**: attribute
**Current Name**: loop_end, total_end
**Recommended Rename**: [loop_buffer_end_offset, total_buffer_end_offset]
**Rename Executed**: [2025-11-12: completed]

### LoopLocalIndices.loop_elements
**Location**: src/cubie/integrators/loops/ode_loop_config.py:129
**Type**: property
**Current Name**: loop_elements
**Recommended Rename**: [loop_element_count]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices.from_dict
**Location**: src/cubie/integrators/loops/ode_loop_config.py:232
**Type**: method
**Current Name**: from_dict
**Recommended Rename**: [load_indices_from_mapping]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices.from_sizes
**Location**: src/cubie/integrators/loops/ode_loop_config.py:253
**Type**: method
**Current Name**: from_sizes
**Recommended Rename**: [NA]
**Rename Executed**: []

### LoopSharedIndices state buffer attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:171, 175
**Type**: attribute
**Current Name**: state, proposed_state
**Recommended Rename**: [state_buffer_slice, proposed_state_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices observable buffer attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:179, 183
**Type**: attribute
**Current Name**: observables, proposed_observables
**Recommended Rename**: [observables_buffer_slice, proposed_observables_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices.parameters
**Location**: src/cubie/integrators/loops/ode_loop_config.py:187
**Type**: attribute
**Current Name**: parameters
**Recommended Rename**: [parameters_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices driver buffer attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:191, 195
**Type**: attribute
**Current Name**: drivers, proposed_drivers
**Recommended Rename**: [driver_buffer_slice, proposed_driver_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices summary buffer attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:199, 203
**Type**: attribute
**Current Name**: state_summaries, observable_summaries
**Recommended Rename**: [state_summary_buffer_slice, observable_summary_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices.error
**Location**: src/cubie/integrators/loops/ode_loop_config.py:207
**Type**: attribute
**Current Name**: error
**Recommended Rename**: [error_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices counter buffer attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:211, 215
**Type**: attribute
**Current Name**: counters, proposed_counters
**Recommended Rename**: [counter_buffer_slice, proposed_counter_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices buffer management attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:219, 223, 227
**Type**: attribute
**Current Name**: local_end, scratch, all
**Recommended Rename**: [loop_buffer_end_offset, scratch_buffer_slice, full_buffer_slice]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices.loop_shared_elements
**Location**: src/cubie/integrators/loops/ode_loop_config.py:346
**Type**: property
**Current Name**: loop_shared_elements
**Recommended Rename**: [loop_shared_element_count]
**Rename Executed**: [2025-11-12: completed]

### LoopSharedIndices dimension properties group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:351, 356, 361, 366, 371
**Type**: property
**Current Name**: n_states, n_parameters, n_drivers, n_observables, n_counters
**Recommended Rename**: [state_count, parameter_count, driver_count, observable_count, counter_count]
**Rename Executed**: [2025-11-12: completed]

### ODELoopConfig buffer indices attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:418, 421
**Type**: attribute
**Current Name**: shared_buffer_indices, local_indices
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig.precision
**Location**: src/cubie/integrators/loops/ode_loop_config.py:425
**Type**: attribute
**Current Name**: precision
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig.compile_flags
**Location**: src/cubie/integrators/loops/ode_loop_config.py:430
**Type**: attribute
**Current Name**: compile_flags
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig timestep attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:434, 438, 470, 474, 478
**Type**: attribute
**Current Name**: _dt_save, _dt_summarise, _dt0, _dt_min, _dt_max
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig device function attributes group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:442, 446, 450, 454, 458, 462, 466
**Type**: attribute
**Current Name**: save_state_fn, update_summaries_fn, save_summaries_fn, step_controller_fn, step_function, driver_function, observables_fn
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig.is_adaptive
**Location**: src/cubie/integrators/loops/ode_loop_config.py:482
**Type**: attribute
**Current Name**: is_adaptive
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig.saves_per_summary
**Location**: src/cubie/integrators/loops/ode_loop_config.py:487
**Type**: property
**Current Name**: saves_per_summary
**Recommended Rename**: [save_count_per_summary]
**Rename Executed**: [2025-11-12: completed]

### ODELoopConfig precision properties group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:492, 497
**Type**: property
**Current Name**: numba_precision, simsafe_precision
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig timestep properties group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:502, 507, 512, 517, 522
**Type**: property
**Current Name**: dt_save, dt_summarise, dt0, dt_min, dt_max
**Recommended Rename**: [NA]
**Rename Executed**: []

### ODELoopConfig memory requirement properties group
**Location**: src/cubie/integrators/loops/ode_loop_config.py:527, 534
**Type**: property
**Current Name**: loop_shared_element_count, loop_local_elements
**Recommended Rename**: [loop_shared_element_count, loop_local_element_count]
**Rename Executed**: [2025-11-12: completed]
