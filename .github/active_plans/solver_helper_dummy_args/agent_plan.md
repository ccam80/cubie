# Solver Helper Dummy Arguments - Agent Plan

## Overview

This plan details the implementation of complete dummy argument coverage for all solver helper functions in `SymbolicODE._generate_dummy_args()`. The method currently only covers `dxdt` and `observables`; this change adds entries for all remaining helpers defined in `ODECache`.

## Target File

**File**: `src/cubie/odesystems/symbolic/symbolicODE.py`  
**Method**: `_generate_dummy_args(self) -> Dict[str, Tuple]`

## Solver Helper Signatures from Codegen Templates

Each helper's signature is extracted from the corresponding template in `codegen/`:

### 1. dxdt (already implemented)
```python
# Template: DXDT_TEMPLATE in dxdt.py
def dxdt(state, parameters, drivers, observables, out, t)
# Args: (n_states,), (n_params,), (n_drivers,), (n_obs,), (n_states,), scalar
```

### 2. observables (already implemented)
```python
# Template: OBSERVABLES_TEMPLATE in dxdt.py
def get_observables(state, parameters, drivers, observables, t)
# Args: (n_states,), (n_params,), (n_drivers,), (n_obs,), scalar
```

### 3. linear_operator
```python
# Template: OPERATOR_APPLY_TEMPLATE in linear_operators.py
def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)
# Args: (n_states,), (n_params,), (n_drivers,), (n_states,), scalar, scalar, scalar, (n_states,), (n_states,)
```

### 4. linear_operator_cached
```python
# Template: CACHED_OPERATOR_APPLY_TEMPLATE in linear_operators.py
def operator_apply(state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out)
# Args: (n_states,), (n_params,), (n_drivers,), (n_aux,), (n_states,), scalar, scalar, scalar, (n_states,), (n_states,)
```

### 5. prepare_jac
```python
# Template: PREPARE_JAC_TEMPLATE in linear_operators.py
def prepare_jac(state, parameters, drivers, t, cached_aux)
# Args: (n_states,), (n_params,), (n_drivers,), scalar, (n_aux,)
```

### 6. calculate_cached_jvp
```python
# Template: CACHED_JVP_TEMPLATE in linear_operators.py
def calculate_cached_jvp(state, parameters, drivers, cached_aux, t, v, out)
# Args: (n_states,), (n_params,), (n_drivers,), (n_aux,), scalar, (n_states,), (n_states,)
```

### 7. neumann_preconditioner
```python
# Template: NEUMANN_TEMPLATE in preconditioners.py
def preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp)
# Args: (n_states,), (n_params,), (n_drivers,), (n_states,), scalar, scalar, scalar, (n_states,), (n_states,), (n_states,)
```

### 8. neumann_preconditioner_cached
```python
# Template: NEUMANN_CACHED_TEMPLATE in preconditioners.py
def preconditioner(state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp)
# Args: (n_states,), (n_params,), (n_drivers,), (n_aux,), (n_states,), scalar, scalar, scalar, (n_states,), (n_states,), (n_states,)
```

### 9. stage_residual
```python
# Template: RESIDUAL_TEMPLATE in nonlinear_residuals.py
def residual(u, parameters, drivers, t, h, a_ij, base_state, out)
# Args: (n_states,), (n_params,), (n_drivers,), scalar, scalar, scalar, (n_states,), (n_states,)
```

### 10. n_stage_residual
```python
# Template: N_STAGE_RESIDUAL_TEMPLATE in nonlinear_residuals.py
def residual(u, parameters, drivers, t, h, a_ij, base_state, out)
# Args: (n_stages * n_states,), (n_params,), (n_stages * n_drivers,), scalar, scalar, scalar, (n_states,), (n_stages * n_states,)
```

### 11. n_stage_linear_operator
```python
# Template: N_STAGE_OPERATOR_TEMPLATE in linear_operators.py
def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)
# Args: (n_stages * n_states,), (n_params,), (n_stages * n_drivers,), (n_states,), scalar, scalar, scalar, (n_stages * n_states,), (n_stages * n_states,)
```

### 12. n_stage_neumann_preconditioner
```python
# Template: N_STAGE_NEUMANN_TEMPLATE in preconditioners.py
def preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp)
# Args: (n_stages * n_states,), (n_params,), (n_stages * n_drivers,), (n_states,), scalar, scalar, scalar, (n_stages * n_states,), (n_stages * n_states,), (n_stages * n_states,)
```

### 13. time_derivative_rhs
```python
# Template: TIME_DERIVATIVE_TEMPLATE in time_derivative.py
def time_derivative_rhs(state, parameters, drivers, driver_dt, observables, out, t)
# Args: (n_states,), (n_params,), (n_drivers,), (n_drivers,), (n_obs,), (n_states,), scalar
```

## Implementation Details

### Constants and Sizing

```python
precision = self.precision
sizes = self.sizes
n_states = int(sizes.states)
n_params = int(sizes.parameters)
n_drivers = int(sizes.drivers)
n_obs = int(sizes.observables)

# For n-stage helpers, use default 2 stages
n_stages = 2
n_flat_states = n_stages * n_states
n_flat_drivers = n_stages * n_drivers

# For cached aux buffer, use reasonable default
n_aux = max(1, n_states * 2)
```

### Helper Arrays

Create reusable arrays to minimize duplication:
- `state_arr = np.ones((n_states,), dtype=precision)`
- `params_arr = np.ones((n_params,), dtype=precision)`
- `drivers_arr = np.ones((n_drivers,), dtype=precision)`
- `obs_arr = np.ones((n_obs,), dtype=precision)`
- `out_arr = np.ones((n_states,), dtype=precision)`
- `cached_aux_arr = np.ones((n_aux,), dtype=precision)`
- Scalars: `t = precision(0.0)`, `h = precision(0.01)`, `a_ij = precision(1.0)`

### n-Stage Arrays

- `flat_state_arr = np.ones((n_flat_states,), dtype=precision)`
- `flat_drivers_arr = np.ones((n_flat_drivers,), dtype=precision)`
- `flat_out_arr = np.ones((n_flat_states,), dtype=precision)`

## Dictionary Entries to Add

```python
return {
    'dxdt': dxdt_args,
    'observables': obs_args,
    'linear_operator': linear_operator_args,
    'linear_operator_cached': linear_operator_cached_args,
    'prepare_jac': prepare_jac_args,
    'calculate_cached_jvp': calculate_cached_jvp_args,
    'neumann_preconditioner': neumann_preconditioner_args,
    'neumann_preconditioner_cached': neumann_preconditioner_cached_args,
    'stage_residual': stage_residual_args,
    'n_stage_residual': n_stage_residual_args,
    'n_stage_linear_operator': n_stage_linear_operator_args,
    'n_stage_neumann_preconditioner': n_stage_neumann_preconditioner_args,
    'time_derivative_rhs': time_derivative_rhs_args,
}
```

## Edge Cases

1. **Zero drivers**: When `n_drivers=0`, still use `max(1, n_drivers)` to avoid empty arrays
2. **Zero observables**: When `n_obs=0`, still use `max(1, n_obs)` to avoid empty arrays
3. **Precision casting**: All scalars must use `precision(value)` to ensure correct dtype

## Integration Points

- **ODECache**: The dictionary keys must match the field names in `ODECache` attrs class in `baseODE.py`
- **get_solver_helper**: The keys correspond to valid `func_type` values accepted by `get_solver_helper()`

## Dependencies

- `numpy` (already imported)
- `self.precision` (inherited from BaseODE)
- `self.sizes` (inherited from BaseODE via compile_settings)

## Testing Considerations

- Tests should verify that `_generate_dummy_args()` returns entries for all expected keys
- Tests should verify argument tuple lengths match expected arity
- Tests should verify array shapes are consistent with system sizes
