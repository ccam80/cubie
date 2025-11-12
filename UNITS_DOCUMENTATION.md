# Units Support in CuBIE

## Overview

This document explains the units tracking feature added to CuBIE, which enables unit information to flow from CellML model definitions through to output legends in `SolveResult` objects.

## Key Concept

Units are **optional metadata** attached to variables (states, parameters, constants, observables, drivers). When not specified, all variables default to `"dimensionless"`. This ensures **full backwards compatibility** with existing code.

---

## Architecture: Units Flow Through CuBIE

### High-Level Data Flow

```
CellML Model File                SymbolicODE Creation           Solver Execution        Result Legends
┌──────────────┐                 ┌──────────────┐              ┌─────────────┐         ┌──────────────┐
│ .cellml file │                 │ SymbolicODE  │              │   Solver    │         │ SolveResult  │
│              │                 │              │              │             │         │              │
│ <variable    │   extract       │ .state_units │   access     │   .system   │  read   │ .time_domain_│
│  name="V"    │─────────────────>│ .param_units │─────────────>│             │────────>│   legend     │
│  units="mV"  │   cellmlmanip   │ .obs_units   │              │             │         │              │
│ />           │                 │              │              │             │         │ {0: "V [mV]"}│
└──────────────┘                 └──────────────┘              └─────────────┘         └──────────────┘
```

### Detailed Component Interaction

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  1. INPUT: CellML Model or Manual Specification                                    │
│                                                                                     │
│     Option A: CellML File                    Option B: Manual Creation             │
│     ┌─────────────────────┐                  ┌────────────────────┐                │
│     │ load_cellml_model() │                  │ SymbolicODE.create()│                │
│     │                     │                  │                    │                │
│     │ Extracts:           │                  │ Accepts:           │                │
│     │ - variable.units    │                  │ - state_units={}   │                │
│     │ - initial values    │                  │ - parameter_units={}│               │
│     └──────────┬──────────┘                  └─────────┬──────────┘                │
│                │                                       │                            │
│                └───────────────────┬───────────────────┘                            │
│                                    │                                                │
└────────────────────────────────────┼────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  2. STORAGE: IndexedBaseMap (Core Data Structure)                                  │
│                                                                                     │
│     ┌───────────────────────────────────────────────────┐                          │
│     │ IndexedBaseMap                                    │                          │
│     │                                                   │                          │
│     │  Symbol Names:  ['x', 'y', 'z']                  │                          │
│     │  Default Values: {'x': 1.0, 'y': 2.0, 'z': 0.0}  │                          │
│     │  Units:         {'x': 'meters',                  │                          │
│     │                  'y': 'meters/second',            │                          │
│     │                  'z': 'dimensionless'}   ◄────── NEW!                        │
│     └───────────────────────────────────────────────────┘                          │
│                                                                                     │
│     Each IndexedBaseMap instance represents one category:                          │
│     - states                                                                       │
│     - parameters                                                                   │
│     - constants                                                                    │
│     - observables                                                                  │
│     - drivers                                                                      │
│                                                                                     │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  3. AGGREGATION: IndexedBases (Container)                                          │
│                                                                                     │
│     ┌──────────────────────────────────────────┐                                   │
│     │ IndexedBases                             │                                   │
│     │                                          │                                   │
│     │  .states      → IndexedBaseMap (states)  │                                   │
│     │  .parameters  → IndexedBaseMap (params)  │                                   │
│     │  .constants   → IndexedBaseMap (consts)  │                                   │
│     │  .observables → IndexedBaseMap (obs)     │                                   │
│     │  .drivers     → IndexedBaseMap (drivers) │                                   │
│     └──────────────────────────────────────────┘                                   │
│                                                                                     │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  4. ACCESS: SymbolicODE (User-Facing API)                                          │
│                                                                                     │
│     ┌──────────────────────────────────────────┐                                   │
│     │ SymbolicODE                              │                                   │
│     │                                          │                                   │
│     │  .indices → IndexedBases                 │                                   │
│     │                                          │                                   │
│     │  Properties (NEW!):                      │                                   │
│     │  .state_units      → indices.states.units│                                   │
│     │  .parameter_units  → indices.parameters.units                                │
│     │  .constant_units   → indices.constants.units                                 │
│     │  .observable_units → indices.observables.units                               │
│     │  .driver_units     → indices.drivers.units                                   │
│     └──────────────────────────────────────────┘                                   │
│                                                                                     │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  5. EXECUTION: Solver                                                              │
│                                                                                     │
│     ┌──────────────────────────────────────────┐                                   │
│     │ Solver                                   │                                   │
│     │                                          │                                   │
│     │  .system → SymbolicODE                   │                                   │
│     │                                          │                                   │
│     │  Units are NOT used during computation   │                                   │
│     │  (purely metadata for output labeling)   │                                   │
│     └──────────────────────────────────────────┘                                   │
│                                                                                     │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  6. OUTPUT: SolveResult Legends                                                    │
│                                                                                     │
│     ┌──────────────────────────────────────────────────────────────┐               │
│     │ SolveResult                                                  │               │
│     │                                                              │               │
│     │  time_domain_legend_from_solver(solver)                     │               │
│     │  ├─ Reads: solver.system.state_units                        │               │
│     │  ├─ Reads: solver.system.observable_units                   │               │
│     │  └─ Formats: "variable_name [unit]"                         │               │
│     │                                                              │               │
│     │  summary_legend_from_solver(solver)                         │               │
│     │  ├─ Reads: solver.system.state_units                        │               │
│     │  ├─ Reads: solver.system.observable_units                   │               │
│     │  └─ Formats: "variable_name [unit] summary_type"            │               │
│     │                                                              │               │
│     │  Example Output:                                            │               │
│     │  .time_domain_legend = {                                    │               │
│     │      0: "voltage [millivolt]",                              │               │
│     │      1: "calcium [micromolar]"                              │               │
│     │  }                                                           │               │
│     │                                                              │               │
│     │  .summaries_legend = {                                      │               │
│     │      0: "voltage [millivolt] mean",                         │               │
│     │      1: "voltage [millivolt] max",                          │               │
│     │      2: "voltage [millivolt] min",                          │               │
│     │      3: "calcium [micromolar] mean",                        │               │
│     │      ...                                                    │               │
│     │  }                                                           │               │
│     └──────────────────────────────────────────────────────────────┘               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## What Happens When No Units Are Provided?

### Default Behavior

When units are **not specified**, all variables automatically default to `"dimensionless"`:

```python
from cubie import SymbolicODE

# No units specified
ode = SymbolicODE.create(
    dxdt="dx = -a * x",
    states={'x': 1.0},
    parameters={'a': 0.5}
)

print(ode.state_units)      # {'x': 'dimensionless'}
print(ode.parameter_units)  # {'a': 'dimensionless'}
```

### Legend Output

```python
result = solve_ivp(ode, [0, 1], [1.0])
print(result.time_domain_legend)
# {0: 'x [dimensionless]'}
```

### Backwards Compatibility

**All existing code continues to work without modification.** The units feature is purely additive:

```python
# Old code (still works exactly as before)
ode = SymbolicODE.create(
    dxdt="dx = -a * x",
    states={'x': 1.0},
    parameters={'a': 0.5}
)
# Units default to 'dimensionless', legends show 'x [dimensionless]'
```

---

## How to Provide Units

### Format 1: Dictionary Mapping

Map each variable name to its unit string:

```python
ode = SymbolicODE.create(
    dxdt="dx = -a * x",
    states={'x': 1.0},
    parameters={'a': 0.5},
    state_units={'x': 'meters'},              # dict format
    parameter_units={'a': 'per_second'}       # dict format
)
```

### Format 2: List/Tuple (Aligned with Variable Order)

Provide units in the same order as variables:

```python
ode = SymbolicODE.create(
    dxdt=["dx = -a * x", "dy = b * y"],
    states={'x': 1.0, 'y': 2.0},
    parameters={'a': 0.5, 'b': 0.3},
    state_units=['meters', 'meters'],         # list format
    parameter_units=['per_second', 'per_second']  # list format
)
```

**Note:** If a variable is missing from the dictionary, it defaults to `'dimensionless'`.

### Format 3: CellML Automatic Extraction

CellML models include unit definitions in the XML:

```xml
<!-- Example CellML file -->
<variable name="V" units="millivolt" initial_value="-84.0"/>
<variable name="Ca" units="micromolar" initial_value="0.0001"/>
```

Units are **automatically extracted** when loading:

```python
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model

ode = load_cellml_model("cardiac_model.cellml")

# Units automatically populated from CellML
print(ode.state_units)
# {'V': 'millivolt', 'Ca': 'micromolar', ...}
```

---

## Unit String Format

### Accepted Forms

CuBIE stores units as **plain strings**. Any string is valid:

```python
# Standard SI units
'meters', 'seconds', 'kilograms', 'ampere'

# Derived units
'meters_per_second', 'joules', 'watts'

# Biological units
'millivolt', 'micromolar', 'millimolar'

# Custom units
'dimensionless', 'arbitrary_units', 'AU'

# Complex expressions (stored as-is)
'meters/second', 'm/s', 'kg*m/s^2'
```

### No Validation or Conversion

CuBIE does **not**:
- Validate unit strings
- Convert between unit systems
- Check dimensional consistency
- Perform unit arithmetic

Units are **purely descriptive metadata** for labeling outputs.

---

## Effects on Existing (Unitless) Models

### No Breaking Changes

The units feature is designed for **100% backwards compatibility**:

| Scenario | Before Units Feature | After Units Feature |
|----------|---------------------|---------------------|
| Create SymbolicODE without units | Works | Works identically, units default to "dimensionless" |
| Load CellML model | Works | Works identically, now also extracts units |
| Access solve results | Works | Works identically, legends now include "[unit]" suffix |
| All existing tests | Pass | Pass (verified) |

### Legend Changes

The **only visible difference** for existing code is in legend strings:

**Before:**
```python
result.time_domain_legend = {0: 'x', 1: 'y'}
```

**After:**
```python
result.time_domain_legend = {0: 'x [dimensionless]', 1: 'y [dimensionless]'}
```

If your code **parses legend strings**, you may need to update regex patterns to handle the `[unit]` suffix.

---

## Complete Usage Examples

### Example 1: Simple ODE with Custom Units

```python
from cubie import SymbolicODE, solve_ivp
import numpy as np

# Define model with units
ode = SymbolicODE.create(
    dxdt="dx = -a * x",
    states={'x': 1.0},
    parameters={'a': 0.5},
    state_units={'x': 'meters'},
    parameter_units={'a': 'per_second'},
    precision=np.float32
)

# Units accessible via properties
print(f"State units: {ode.state_units}")
# State units: {'x': 'meters'}

print(f"Parameter units: {ode.parameter_units}")
# Parameter units: {'a': 'per_second'}

# Solve
result = solve_ivp(ode, [0.0, 1.0], [1.0])

# Units appear in legends
print(result.time_domain_legend)
# {0: 'x [meters]'}
```

### Example 2: Multi-Variable System

```python
ode = SymbolicODE.create(
    dxdt=[
        "dx = velocity",
        "dvelocity = -k * x / mass"
    ],
    states={
        'x': 0.0,
        'velocity': 1.0
    },
    parameters={
        'k': 10.0,
        'mass': 1.0
    },
    state_units={
        'x': 'meters',
        'velocity': 'meters_per_second'
    },
    parameter_units={
        'k': 'newtons_per_meter',
        'mass': 'kilograms'
    }
)

print(ode.state_units)
# {'x': 'meters', 'velocity': 'meters_per_second'}

result = solve_ivp(ode, [0, 10], [0.0, 1.0])
print(result.time_domain_legend)
# {0: 'x [meters]', 1: 'velocity [meters_per_second]'}
```

### Example 3: CellML Model

```python
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model

# Load CellML file with units
ode = load_cellml_model("beeler_reuter_1977.cellml")

# Units extracted automatically
print(ode.state_units)
# {'membrane_V': 'millivolt', 
#  'sodium_channel_m_gate_m': 'dimensionless',
#  'sodium_channel_h_gate_h': 'dimensionless',
#  ...}

# Solve and get results
result = solve_ivp(ode, [0, 100], ode.states.values_array)

# Legends include units from CellML
print(result.time_domain_legend)
# {0: 'membrane_V [millivolt]',
#  1: 'sodium_channel_m_gate_m [dimensionless]',
#  ...}
```

### Example 4: Observables with Units

```python
ode = SymbolicODE.create(
    dxdt=[
        "dx = -a * x",
        "kinetic_energy = 0.5 * mass * x**2"
    ],
    states={'x': 1.0},
    parameters={'a': 0.5, 'mass': 2.0},
    observables=['kinetic_energy'],
    state_units={'x': 'meters_per_second'},
    parameter_units={'a': 'per_second', 'mass': 'kilograms'},
    observable_units={'kinetic_energy': 'joules'}
)

result = solve_ivp(ode, [0, 5], [1.0])
print(result.time_domain_legend)
# {0: 'x [meters_per_second]', 1: 'kinetic_energy [joules]'}
```

### Example 5: Summary Metrics with Units

```python
from cubie import Solver

ode = SymbolicODE.create(
    dxdt="dx = -a * x",
    states={'x': 1.0},
    parameters={'a': 0.5},
    state_units={'x': 'millivolt'}
)

solver = Solver(
    ode,
    algorithm='euler',
    output_settings={
        'saved_states': ['x'],
        'summarised_states': ['x'],
        'dt_summarise': 0.1
    }
)

result = solver.solve([0, 1], [1.0])

# Summary legends include units
print(result.summaries_legend)
# {0: 'x [millivolt] mean',
#  1: 'x [millivolt] max',
#  2: 'x [millivolt] min'}
```

---

## Modified Files Summary

### Core Changes

1. **`src/cubie/odesystems/symbolic/indexedbasemaps.py`**
   - `IndexedBaseMap.__init__()`: Added `units` parameter (dict or list)
   - `IndexedBaseMap.pop()`: Removes units when symbol removed
   - `IndexedBaseMap.push()`: Adds unit when symbol added
   - `IndexedBases.from_user_inputs()`: Added `*_units` parameters

2. **`src/cubie/odesystems/symbolic/symbolicODE.py`**
   - `SymbolicODE.create()`: Added `state_units`, `parameter_units`, etc.
   - Added properties: `.state_units`, `.parameter_units`, `.constant_units`, `.observable_units`, `.driver_units`

3. **`src/cubie/odesystems/symbolic/parsing/parser.py`**
   - `parse_input()`: Added `*_units` parameters
   - `_process_parameters()`: Passes units to `IndexedBases.from_user_inputs()`

4. **`src/cubie/odesystems/symbolic/parsing/cellml.py`**
   - `load_cellml_model()`: Extracts units from CellML variables
   - Passes units to `SymbolicODE.create()`

5. **`src/cubie/batchsolving/solveresult.py`**
   - `time_domain_legend_from_solver()`: Formats as `"label [unit]"`
   - `summary_legend_from_solver()`: Formats as `"label [unit] summary_type"`

### Test Coverage

**`tests/odesystems/symbolic/test_cellml.py`** - Added 3 tests:
- `test_units_extracted_from_cellml`: Verifies CellML units extraction
- `test_default_units_for_symbolic_ode`: Verifies default "dimensionless"
- `test_custom_units_for_symbolic_ode`: Verifies custom units specification

---

## API Reference

### SymbolicODE.create()

```python
SymbolicODE.create(
    dxdt: str | Iterable[str],
    states: dict[str, float] | Iterable[str] | None = None,
    parameters: dict[str, float] | Iterable[str] | None = None,
    constants: dict[str, float] | Iterable[str] | None = None,
    observables: Iterable[str] | None = None,
    drivers: Iterable[str] | dict[str, Any] | None = None,
    
    # NEW: Units parameters (all optional)
    state_units: dict[str, str] | Iterable[str] | None = None,
    parameter_units: dict[str, str] | Iterable[str] | None = None,
    constant_units: dict[str, str] | Iterable[str] | None = None,
    observable_units: dict[str, str] | Iterable[str] | None = None,
    driver_units: dict[str, str] | Iterable[str] | None = None,
    
    user_functions: dict[str, Callable] | None = None,
    name: str | None = None,
    precision: PrecisionDType = np.float32,
    strict: bool = False,
) -> SymbolicODE
```

### SymbolicODE Properties

```python
ode = SymbolicODE.create(...)

# NEW properties (all return dict[str, str])
ode.state_units       # Units for state variables
ode.parameter_units   # Units for parameters
ode.constant_units    # Units for constants
ode.observable_units  # Units for observables
ode.driver_units      # Units for drivers
```

### load_cellml_model()

```python
from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model

ode = load_cellml_model(
    path: str,
    precision: PrecisionDType = np.float32,
    name: str | None = None,
    parameters: list[str] | None = None,
    observables: list[str] | None = None,
) -> SymbolicODE

# Units automatically extracted from CellML file
# Access via ode.state_units, ode.parameter_units, etc.
```

---

## Frequently Asked Questions

### Q: Do I need to update my existing code?

**A:** No. All existing code works without modification. Units default to "dimensionless" when not specified.

### Q: Will this break my tests?

**A:** Only if your tests parse legend strings. Legends now include `[unit]` suffix. Otherwise, no breaking changes.

### Q: Can I disable units?

**A:** Units are always present but default to "dimensionless". There's no way to completely disable them, but they have zero performance impact.

### Q: Does CuBIE check unit consistency?

**A:** No. Units are purely descriptive metadata. CuBIE does not validate, convert, or check dimensional consistency.

### Q: Can I use mathematical expressions in units?

**A:** Yes, but they're stored as plain strings. For example, `'kg*m/s^2'` is valid but not evaluated.

### Q: What if I load a CellML file without units?

**A:** Variables without units in CellML default to "dimensionless".

### Q: Do units affect simulation performance?

**A:** No. Units are only used for labeling outputs, not during computation.

---

## Migration Guide

### If You Parse Legend Strings

**Before:**
```python
# Old code expecting plain names
legend = result.time_domain_legend
for idx, name in legend.items():
    if name == 'voltage':
        # do something
```

**After (recommended):**
```python
# Updated to handle units
legend = result.time_domain_legend
for idx, label in legend.items():
    # Extract name without unit
    name = label.split(' [')[0]  # 'voltage [millivolt]' → 'voltage'
    if name == 'voltage':
        # do something
```

### If You Display Legends

**Before:**
```python
print(f"Variable: {legend[0]}")
# Output: Variable: voltage
```

**After (no change needed):**
```python
print(f"Variable: {legend[0]}")
# Output: Variable: voltage [millivolt]
```

The new format is more informative and requires no code changes for display purposes.

---

## Summary

The units feature:
- ✅ **Optional** - Defaults to "dimensionless"
- ✅ **Backwards compatible** - No breaking changes
- ✅ **Automatic** - CellML units extracted automatically
- ✅ **Flexible** - Accepts dict or list format
- ✅ **Metadata only** - No runtime performance impact
- ✅ **Informative** - Enriches output legends

Units flow from model definition → storage → ODE system → solver → result legends, providing clear labeling for analysis and visualization.
