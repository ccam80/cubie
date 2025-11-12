# CellML Variable Flow Diagrams - Fabbri_Linder Model

This document provides visual diagrams showing how variables flow through the component hierarchy in the Fabbri_Linder CellML model, comparing a **correctly specified variable (ACh)** with the **problematic variable (cAMP)**.

## Legend

```
[Component]           - Component in the model
variable_name         - Variable with interface attributes
├─                    - Component hierarchy (parent-child)
→                     - Data flow direction
public_in             - public_interface="in"
public_out            - public_interface="out"
private_out           - private_interface="out"
(missing)             - Attribute not specified (defaults to "none")
```

---

## Diagram 1: ACh Variable Flow (CORRECTLY SPECIFIED) ✓

The ACh (acetylcholine) variable demonstrates the **correct** way to specify variable interfaces in a parent-child component hierarchy.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Component: ACh                                                      │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: ACh                                   │                │
│ │   public_interface="out"                        │                │
│ │   initial_value="0"                             │                │
│ └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Connection: ACh.ACh → i_f.ACh
                              ↓ (public_out → public_in)
┌─────────────────────────────────────────────────────────────────────┐
│ Component: i_f (Parent)                                             │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: ACh                                   │                │
│ │   public_interface="in"    ← receives from ACh component         │
│ │   private_interface="out"  ← passes to child  ✓                  │
│ └─────────────────────────────────────────────────┘                │
│                                                                     │
│   ├─ [i_f_y_gate] (Child component)                                │
└───┼─────────────────────────────────────────────────────────────────┘
    │
    │ Connection: i_f.ACh → i_f_y_gate.ACh
    ↓ (private_out → public_in)
┌─────────────────────────────────────────────────────────────────────┐
│ Component: i_f_y_gate (Child of i_f)                                │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: ACh                                   │                │
│ │   public_interface="in"    ← receives from parent i_f            │
│ │                                                 │                │
│ │ Used in equations to compute ACh_shift          │                │
│ └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Flow Summary (ACh):
1. **Source:** `ACh` component provides value via `public_interface="out"`
2. **Intermediate:** `i_f` component receives via `public_interface="in"` and passes via `private_interface="out"` ✓
3. **Destination:** `i_f_y_gate` component receives via `public_interface="in"`

### Key Point:
The `i_f` component has **BOTH** `public_interface="in"` AND `private_interface="out"` set, allowing it to:
- Receive the value from its parent/sibling (ACh component)
- Pass the value to its child (i_f_y_gate component)

---

## Diagram 2: cAMP Variable Flow (INCORRECTLY SPECIFIED) ✗

The cAMP variable demonstrates the **problem** that causes cellmlmanip parsing to fail.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Component: cAMP                                                     │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: cAMP                                  │                │
│ │   public_interface="out"                        │                │
│ │   initial_value="0.032883333"                   │                │
│ └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Connection: cAMP.cAMP → i_f.cAMP
                              ↓ (public_out → public_in)
┌─────────────────────────────────────────────────────────────────────┐
│ Component: i_f (Parent)                                             │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: cAMP                                  │                │
│ │   public_interface="in"    ← receives from cAMP component        │
│ │   private_interface=(missing) ← NOT SET!  ✗                      │
│ │                                  Defaults to "none"               │
│ └─────────────────────────────────────────────────┘                │
│                                                                     │
│   ├─ [i_f_y_gate] (Child component)                                │
└───┼─────────────────────────────────────────────────────────────────┘
    │
    │ Connection: i_f.cAMP → i_f_y_gate.cAMP
    ↓ PARSING FAILS HERE! ✗
    │ Parser cannot determine direction because:
    │   - Child has public_interface="in" (expects input)
    │   - Parent has private_interface=(none) (not configured to output)
    │
┌─────────────────────────────────────────────────────────────────────┐
│ Component: i_f_y_gate (Child of i_f)                                │
│ ┌─────────────────────────────────────────────────┐                │
│ │ Variable: cAMP                                  │                │
│ │   public_interface="in"    ← expects from parent i_f             │
│ │                                                 │                │
│ │ ERROR: Cannot receive because parent doesn't output!             │
│ └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Flow Summary (cAMP):
1. **Source:** `cAMP` component provides value via `public_interface="out"`
2. **Intermediate:** `i_f` component receives via `public_interface="in"` but **MISSING** `private_interface="out"` ✗
3. **Destination:** `i_f_y_gate` component expects value via `public_interface="in"`
4. **Result:** Parser fails at step 2→3 because `i_f` doesn't declare `private_interface="out"`

### The Problem:
The `i_f` component is **missing** the `private_interface="out"` attribute, so:
- It successfully receives the value from the cAMP component
- But it cannot pass the value to its child component according to CellML specification
- The parser strictly enforces this requirement

---

## Side-by-Side Comparison

### ACh Variable (Working):
```xml
<!-- In i_f component -->
<variable name="ACh" 
          private_interface="out"    ← PRESENT ✓
          public_interface="in" 
          units="nanomolar"/>
```

### cAMP Variable (Broken):
```xml
<!-- In i_f component -->
<variable name="cAMP" 
          public_interface="in"      ← MISSING private_interface ✗
          units="millimolar"/>
```

### Fix Required:
```xml
<!-- Should be: -->
<variable name="cAMP" 
          private_interface="out"    ← ADD THIS
          public_interface="in" 
          units="millimolar"/>
```

---

## Complete Connection Chain

### ACh Complete Chain (Working):
```
ACh Component                    i_f Component                    i_f_y_gate Component
┌──────────────┐                ┌──────────────┐                ┌──────────────────┐
│ ACh          │                │ ACh          │                │ ACh              │
│ public: out  │───────────────>│ public: in   │───────────────>│ public: in       │
└──────────────┘   (sibling/    │ private: out │   (parent/     └──────────────────┘
                    parent       └──────────────┘    child
                   connection)                      connection)
```

### cAMP Complete Chain (Broken):
```
cAMP Component                   i_f Component                    i_f_y_gate Component
┌──────────────┐                ┌──────────────┐                ┌──────────────────┐
│ cAMP         │                │ cAMP         │                │ cAMP             │
│ public: out  │───────────────>│ public: in   │─────X────────>│ public: in       │
└──────────────┘   (sibling/    │ private: ✗   │   FAILS!       └──────────────────┘
                    parent       └──────────────┘   (parser      
                   connection)      (none)         cannot        
                    ✓ Works                        determine      
                                                   direction)     
```

---

## All 15 Affected Variables

The following variables in the Fabbri_Linder model have the same issue as cAMP (missing `private_interface="out"`):

### In Membrane Component (11 variables):
1. `i_f` - funny current
2. `i_NaK` - sodium-potassium pump current
3. `i_NaCa` - sodium-calcium exchanger current
4. `i_Na` - fast sodium current
5. `i_Kr` - rapid delayed rectifier potassium current
6. `i_Ks` - slow delayed rectifier potassium current
7. `i_to` - transient outward current
8. `i_CaL` - L-type calcium current
9. `i_CaT` - T-type calcium current
10. `i_KACh` - acetylcholine-activated potassium current
11. `i_Kur` - ultra-rapid delayed rectifier potassium current

### In Other Components (4 variables):
12. `cAMP` in `i_f` component
13. `PKA` in `i_CaL` component
14. `PKA` in `i_Ks` component
15. `ACh_cas` in `i_KACh` component

All follow the same pattern:
- Parent component receives via `public_interface="in"`
- Parent component has connection to child component
- Parent component **missing** `private_interface="out"`
- Child component expects value via `public_interface="in"`

---

## Why cellmlmanip Fails

The cellmlmanip parser implements the CellML specification strictly. In `parser.py:547-552`:

```python
# parent/child components are connected using private/public interface, respectively
if child_var.public_interface == 'in' and parent_var.private_interface == 'out':
    return parent_var, child_var  # Valid connection
elif child_var.public_interface == 'out' and parent_var.private_interface == 'in':
    return child_var, parent_var  # Valid connection
    
# If neither condition is true:
raise ValueError('Cannot determine the source & target for connection...')
```

Since `parent_var.private_interface` is `None` (not `'out'`), neither condition matches, causing the error.

---

## CellML Specification Reference

From the CellML 1.0/1.1 specification:

> **public_interface**: Defines the interface exposed to the parent component and components in the sibling set.
> 
> **private_interface**: Defines the interface exposed to components in the encapsulated set (child components).
> 
> Each interface has three possible values: "in", "out", and "none", where "none" indicates the absence of an interface.

For a variable to flow from parent to child:
- Parent must have `private_interface="out"` (expose to children)
- Child must have `public_interface="in"` (accept from parent)

The Fabbri_Linder model violates this requirement in 15 places.

---

## Conclusion

**ACh variable:** Correctly implements the CellML specification with both `public_interface="in"` and `private_interface="out"`, allowing the value to flow from source → parent → child.

**cAMP variable:** Missing `private_interface="out"`, breaking the parent→child connection according to the CellML specification. However, the intent is clear from the model structure and connections.

The recommended cellmlmanip patch would add tolerance for this common pattern by inferring the missing `private_interface="out"` when context makes the data flow unambiguous.
