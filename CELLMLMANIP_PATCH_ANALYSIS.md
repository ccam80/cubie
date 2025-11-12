# CellML Fabbri_Linder Model - cellmlmanip Parser Analysis

## Error Summary
**File:** `tests/fixtures/cellml/Fabbri_Linder.cellml`
**Error:** `ValueError: Cannot determine the source & target for connection (i_f, cAMP) - (i_f_y_gate, cAMP)`
**Location:** `cellmlmanip/parser.py:552` in `_determine_connection_direction()`

## Root Cause

### The Problem
The Fabbri_Linder CellML model defines a parent-child component relationship:
- **Parent:** `i_f` component
- **Child:** `i_f_y_gate` component (encapsulated within `i_f`)

Both components have a `cAMP` variable that needs to be connected:
- `i_f.cAMP`: `public_interface="in"` (receives from external source)
- `i_f_y_gate.cAMP`: `public_interface="in"` (expects to receive from parent)

There is a connection element mapping these two variables.

### CellML Specification Requirements
According to the CellML 1.0/1.1 specification:
- **public_interface**: Defines interface exposed to parent component and sibling components
- **private_interface**: Defines interface exposed to encapsulated (child) components
- Values: "in", "out", or "none" (default if not specified)

For a parent component to pass a variable value to a child component:
- Parent variable must have `private_interface="out"`
- Child variable must have `public_interface="in"`

### The Issue in Fabbri_Linder Model
The `i_f` component's `cAMP` variable is defined as:
```xml
<variable name="cAMP" public_interface="in" units="millimolar"/>
```

**Missing:** `private_interface="out"` attribute

This is in contrast to other variables in the same component that ARE properly passed to children:
```xml
<variable name="ACh" private_interface="out" public_interface="in" units="nanomolar"/>
<variable name="time" private_interface="out" public_interface="in" units="second"/>
<variable name="V" private_interface="out" public_interface="in" units="millivolt"/>
```

### cellmlmanip Parser Logic
The parser code (lines 547-550) checks:
```python
# parent/child components are connected using private/public interface, respectively
if child_var.public_interface == 'in' and parent_var.private_interface == 'out':
    return parent_var, child_var
elif child_var.public_interface == 'out' and parent_var.private_interface == 'in':
    return child_var, parent_var
raise ValueError(...)  # Falls through to error
```

Since `parent_var.private_interface` is `None` (not `'out'`), neither condition is satisfied.

## Is This a Model Error or Parser Limitation?

### Arguments for Model Error:
1. The CellML specification explicitly requires `private_interface` to be set for parent→child connections
2. Other variables in the same model correctly use both interface attributes
3. The model appears to have an oversight/inconsistency

### Arguments for Parser Limitation:
1. The connection semantics are unambiguous from context:
   - Both variables have `public_interface="in"`
   - There's an explicit connection element
   - Component hierarchy is defined (parent→child)
   - The data flow direction can be inferred from the source component providing `cAMP` via `public_interface="out"`

2. The parser could be more tolerant of missing interface declarations when the intent is clear

3. Other CellML parsers may handle this more gracefully

## Proposed Minimal Patch for cellmlmanip

### Option 1: Infer Missing private_interface (More Tolerant)

Add a fallback mechanism that infers the interface direction when not explicitly set:

```python
# In _determine_connection_direction method, around line 547
def _determine_connection_direction(self, comp_1, var_1, comp_2, var_2):
    # ... existing code ...
    
    # parent/child components are connected using private/public interface, respectively
    parent_private = parent_var.private_interface or 'none'
    child_public = child_var.public_interface or 'none'
    
    if child_public == 'in' and parent_private == 'out':
        return parent_var, child_var
    elif child_public == 'out' and parent_private == 'in':
        return child_var, parent_var
    
    # NEW: Attempt to infer direction from context
    # If parent's private_interface is not set but child expects input,
    # check if there's a source that provides this value to the parent
    if child_public == 'in' and parent_private == 'none':
        # Parent receives via public_interface and should pass to child
        if parent_var.public_interface == 'in':
            # Infer that parent acts as passthrough
            import warnings
            warnings.warn(
                f"Inferring private_interface='out' for {comp_1}.{var_1} "
                f"in connection to child {comp_2}.{var_2}. "
                f"Consider adding private_interface='out' to the model.",
                UserWarning
            )
            return parent_var, child_var
    
    raise ValueError('Cannot determine the source & target for connection (%s, %s) - (%s, %s)' %
                     (comp_1, var_1, comp_2, var_2))
```

### Option 2: Better Error Message (Minimal Change)

If strict spec compliance is desired, at least provide a helpful error message:

```python
# At line 552, replace the generic ValueError with:
raise ValueError(
    f'Cannot determine the source & target for connection ({comp_1}, {var_1}) - ({comp_2}, {var_2}). '
    f'For parent-child connections, the parent variable must have private_interface set. '
    f'Variable {comp_1}.{var_1} has public_interface="{parent_var.public_interface}" '
    f'but private_interface="{parent_var.private_interface}". '
    f'Expected private_interface="out" to pass value to child component {comp_2}.'
)
```

### Option 3: Validation Pass with Auto-Fix (Most Robust)

Add a pre-processing validation step that detects and fixes common interface issues:

```python
def _validate_and_fix_interfaces(self):
    """Validate and optionally fix common interface specification issues."""
    for connection in self.connections:
        comp_1, var_1 = connection['comp_1'], connection['var_1']
        comp_2, var_2 = connection['comp_2'], connection['var_2']
        
        # Check if parent-child relationship
        if self.components[comp_1].parent == comp_2 or self.components[comp_2].parent == comp_1:
            # Determine parent/child
            if self.components[comp_1].parent == comp_2:
                parent_comp, parent_var = comp_2, var_2
                child_comp, child_var = comp_1, var_1
            else:
                parent_comp, parent_var = comp_1, var_1
                child_comp, child_var = comp_2, var_2
            
            # Get variable objects
            p_var = self.model.get_variable_by_name(self._get_variable_name(parent_comp, parent_var))
            c_var = self.model.get_variable_by_name(self._get_variable_name(child_comp, child_var))
            
            # Check for missing private_interface on parent
            if c_var.public_interface == 'in' and p_var.private_interface is None:
                if p_var.public_interface == 'in':
                    # Parent receives and passes to child - auto-set private_interface
                    warnings.warn(
                        f"Auto-setting private_interface='out' for {parent_comp}.{parent_var}",
                        UserWarning
                    )
                    p_var.private_interface = 'out'
```

## Recommendation

**Option 1** is recommended as it:
1. Maintains backward compatibility with correctly-specified models
2. Adds tolerance for common modeling patterns
3. Warns users about the issue so they can fix their models
4. Allows models from the Physiome repository to load successfully

The patch should be contributed upstream to the cellmlmanip project with appropriate tests demonstrating the issue and fix.

## Testing the Patch

After applying the patch, test with:
```python
import cellmlmanip
model = cellmlmanip.load_model('tests/fixtures/cellml/Fabbri_Linder.cellml')
print(f"Successfully loaded model with {len(list(model.get_state_variables()))} states")
```

Expected output: Model loads successfully with a warning about inferred interfaces.

## Implementation Details

### Files to Modify
- `cellmlmanip/parser.py`: Modify `_determine_connection_direction()` method (around lines 547-552)

### Affected Variable Interfaces in Fabbri_Linder.cellml
The following parent component variables are missing `private_interface="out"` but have connections to child components:

1. `Membrane` component:
   - `i_f`, `i_NaK`, `i_NaCa`, `i_Na`, `i_Kr`, `i_Ks`, `i_to`, `i_CaL`, `i_CaT`, `i_KACh`, `i_Kur`

2. `i_f` component:
   - `cAMP`

3. `i_CaL` component:
   - `PKA`

4. `i_Ks` component:
   - `PKA`

5. `i_KACh` component:
   - `ACh_cas`

Total: 15 variables across 5 components would benefit from the more tolerant parser behavior.
