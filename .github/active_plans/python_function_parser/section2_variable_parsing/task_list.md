# Implementation Task List
# Feature: Python Function Parser - Section 2 (Variable Parsing)
# Plan Reference: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md

## Task Group 1: NameGenerator Module
**Status**: [ ]
**Dependencies**: None (Section 1 provides AccessPattern TypedDict)

**Required Context**:
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (lines 1-147)
- File: .github/active_plans/python_function_parser/section1_source_code/task_list.md (lines 340-380 for AccessPattern definition)
- File: .github/context/cubie_internal_structure.md (entire file)
- File: .github/copilot-instructions.md (entire file)

**Input Validation Required**:
- state_param_name in `__init__`: Check `isinstance(state_param_name, str)` and `len(state_param_name) > 0` - raise ValueError if invalid
- state_accesses in `generate_state_names`: Check `isinstance(state_accesses, list)` - raise TypeError if invalid
- constant_accesses in `generate_constant_names`: Check `isinstance(constant_accesses, list)` - raise TypeError if invalid
- direct_args in `generate_constant_names`: Check `isinstance(direct_args, list)` - raise TypeError if invalid

**Tasks**:
1. **Create NameGenerator module**
   - File: src/cubie/odesystems/symbolic/parsing/name_generator.py
   - Action: Create
   - Details:
     Full implementation converting access patterns to variable names following CuBIE conventions.
     Key methods:
     - `__init__(state_param_name)`: Initialize with validation
     - `generate_state_names(state_accesses)`: Convert state accesses to ordered names
     - `generate_constant_names(constant_accesses, direct_args)`: Extract constant names
     - `validate_consistency(accesses)`: Check for mixed access patterns
     - `_generate_int_subscript_names()`: Handle y[0] pattern
     - `_generate_str_subscript_names()`: Handle y["name"] pattern
     - `_generate_attribute_names()`: Handle y.attr pattern
     
     Error messages must clearly describe what was found vs what was expected.
     Support all three access patterns: int subscript, str subscript, attribute.
     Preserve order of first appearance, deduplicate.
   - Edge cases:
     - Empty state_accesses: Raise ValueError with suggestion to access state parameter
     - Mixed int/str subscripts: Raise ValueError showing examples of both
     - Mixed attribute and subscript: Raise ValueError explaining inconsistency
     - Duplicate accesses: Silently deduplicate while preserving order
   - Integration:
     - Used by VariableIdentifier (Task Group 4)
     - Receives AccessPattern dicts from Section 1 OdeAstVisitor
     - Produces variable names for VariableClassifier

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_name_generator.py
- Test function: test_int_subscript_names - y[0], y[1] generates ['y_0', 'y_1']
- Test function: test_str_subscript_names - y["velocity"] generates ['velocity']
- Test function: test_attribute_names - y.position generates ['position']
- Test function: test_mixed_int_str_error - Mixed y[0] and y["name"] raises ValueError
- Test function: test_mixed_attribute_subscript_error - Mixed y.attr and y[0] raises ValueError
- Test function: test_preserve_order - Order matches first appearance in accesses
- Test function: test_deduplicate_accesses - y[0] accessed twice generates single 'y_0'
- Test function: test_constant_from_attribute - constants.k generates 'k'
- Test function: test_constant_from_str_subscript - constants["k"] generates 'k'
- Test function: test_direct_arg_constants - Direct args added to constant names
- Test function: test_empty_state_accesses_error - Empty access list raises ValueError
- Test function: test_deduplicate_utility - _deduplicate_preserving_order removes duplicates correctly

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: VariableClassifier Module
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (lines 148-297)
- File: src/cubie/odesystems/symbolic/parsing/name_generator.py (entire file)

**Input Validation Required**:
- inferred_states in `__init__`: Check `isinstance(inferred_states, list)` and all items are strings
- inferred_constants in `__init__`: Check `isinstance(inferred_constants, list)` and all items are strings
- user_specs in `__init__`: Check is dict if not None
- No validation on keys within user_specs (handled by classification logic)

**Tasks**:
1. **Create VariableClassifier module**
   - File: src/cubie/odesystems/symbolic/parsing/variable_classifier.py
   - Action: Create
   - Details:
     Categorize variables into states, parameters, constants, observables with user overrides.
     Define TypedDicts and attrs classes:
     - `UserSpecifications` TypedDict: states, parameters, constants, observables fields
     - `ClassifiedVariables` attrs class: categorized results with defaults dicts
     
     Key methods:
     - `__init__(inferred_states, inferred_constants, user_specs, assignments)`: Initialize
     - `classify()`: Main workflow returning ClassifiedVariables
     - `_apply_state_override()`: Use user states if provided
     - `_promote_parameters(constants)`: Move constants to parameters per user spec
     - `_extract_observables()`: Validate observables are assigned
     - `_collect_defaults()`: Extract default values from dict specs
     - `validate(classified)`: Check for overlaps and inconsistencies
     
     Error messages must list available options and suggest corrections.
   - Edge cases:
     - Parameter not in constants: Raise ValueError listing available constants
     - Observable not assigned: Raise ValueError showing required assignment format
     - State count mismatch: Warn but use user specification
     - Name in multiple categories: Detected in validate(), raise ValueError
     - User specs with wrong types (e.g., states as int): Raise TypeError
   - Integration:
     - Used by VariableIdentifier (Task Group 4)
     - Receives inferred names from NameGenerator
     - Produces ClassifiedVariables for SymbolManager

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_variable_classifier.py
- Test function: test_classify_default_all_constants - Without user specs, all inferred become constants
- Test function: test_classify_with_state_override - User states override inferred
- Test function: test_promote_to_parameters - User parameter spec moves from constants
- Test function: test_parameter_not_found_error - Invalid parameter name raises ValueError
- Test function: test_observable_validation - Observable must be in assignments
- Test function: test_collect_defaults_dicts - Extract defaults from dict user specs
- Test function: test_collect_defaults_lists - Lists have no defaults extracted
- Test function: test_state_count_mismatch_warning - Warning when user state count differs
- Test function: test_no_category_overlap - Validate catches name in multiple categories
- Test function: test_states_dict_format - States as dict extracts keys as names
- Test function: test_parameters_dict_format - Parameters as dict extracts keys as names

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: SymbolManager Module
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (lines 298-476)
- File: src/cubie/odesystems/symbolic/parsing/variable_classifier.py (entire file)
- File: src/cubie/odesystems/symbolic/indexedbasemaps.py (lines 193-280)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-40 for TIME_SYMBOL)

**Input Validation Required**:
- classified_vars in `__init__`: Check `isinstance(classified_vars, ClassifiedVariables)` - raise TypeError if invalid
- time_symbol in `__init__`: Check `isinstance(time_symbol, sp.Symbol)` - raise TypeError if invalid
- assignments in `build_symbol_map`: Check `isinstance(assignments, dict)` - raise TypeError if invalid
- No additional validation needed for optional parameters

**Tasks**:
1. **Create SymbolManager module**
   - File: src/cubie/odesystems/symbolic/parsing/symbol_manager.py
   - Action: Create
   - Details:
     Create IndexedBases and symbol mappings from classified variables.
     
     Key methods:
     - `__init__(classified_vars, time_symbol=TIME_SYMBOL)`: Initialize
     - `create_indexed_bases(drivers, unit_specs)`: Call IndexedBases.from_user_inputs
     - `build_symbol_map(assignments)`: Map function vars to SymPy symbols
     - `get_derivative_name(state_name)`: Return 'd{state}' format
     - `get_symbol_for_name(name)`: Lookup symbol by name
     
     Handle defaults from classified_vars: if defaults dict provided, pass to IndexedBases.
     Create auxiliary symbols for assignments not in any category.
     Include time symbol, state symbols, parameter symbols, constant symbols, derivative symbols, observable symbols.
     
     Also define convenience function:
     - `create_symbols_from_function()`: Combines create + build in one call
   - Edge cases:
     - create_indexed_bases called multiple times: Replaces existing
     - build_symbol_map called before create_indexed_bases: Raise ValueError
     - Variable in assignments not in any category: Create as auxiliary symbol with sp.Symbol(name, real=True)
     - Derivative name generation: Simple prefix 'd' without complex rules
   - Integration:
     - Used by VariableIdentifier (Task Group 4)
     - Receives ClassifiedVariables from VariableClassifier
     - Uses IndexedBases.from_user_inputs from existing CuBIE infrastructure
     - Produces IndexedBases and symbol_map for Section 3

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_symbol_manager.py
- Test function: test_create_indexed_bases_basic - Creates IndexedBases with correct categories
- Test function: test_create_indexed_bases_with_defaults - Defaults passed to IndexedBases
- Test function: test_build_symbol_map_states - State symbols mapped correctly
- Test function: test_build_symbol_map_parameters - Parameter symbols mapped correctly
- Test function: test_build_symbol_map_constants - Constant symbols mapped correctly
- Test function: test_build_symbol_map_observables - Observable symbols mapped correctly
- Test function: test_derivative_symbols - d{state} symbols for all states
- Test function: test_auxiliary_symbols - Assignments not in categories create auxiliaries
- Test function: test_time_symbol_included - TIME_SYMBOL mapped to 't'
- Test function: test_get_derivative_name - Returns 'd{name}' format
- Test function: test_get_symbol_for_name - Lookup works for all names
- Test function: test_build_before_create_error - Raises ValueError if build called first
- Test function: test_create_symbols_convenience - Convenience function returns both structures

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: VariableIdentifier Orchestrator
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (lines 477-596)
- File: src/cubie/odesystems/symbolic/parsing/name_generator.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/variable_classifier.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/symbol_manager.py (entire file)
- File: .github/active_plans/python_function_parser/section1_source_code/task_list.md (lines 340-380 for VisitorResults)

**Input Validation Required**:
- visitor_results in `__init__`: Check is dict with required keys ('state_accesses', 'constant_accesses', 'assignments', 'return_node')
- func_params in `__init__`: Check `isinstance(func_params, list)` and all items are strings
- user_specs in `__init__`: No validation (passed to VariableClassifier which validates)

**Tasks**:
1. **Create VariableIdentifier orchestrator module**
   - File: src/cubie/odesystems/symbolic/parsing/variable_identifier.py
   - Action: Create
   - Details:
     Orchestrate complete variable identification workflow.
     
     Key methods:
     - `__init__(visitor_results, func_params, user_specs)`: Initialize with validation
     - `identify(drivers, unit_specs)`: Main workflow returning (IndexedBases, symbol_map, ClassifiedVariables)
     - `_identify_state_parameter()`: Extract state param name (second param)
     - `_identify_constant_parameters()`: Extract constant param names (after state)
     - `_extract_direct_constant_args()`: Get direct args (not state/time)
     - `_validate_return_statement(state_count)`: Check return value count
     - `validate_output(indexed_bases, symbol_map)`: Final structure validation
     
     Workflow:
     1. Identify state parameter from func_params
     2. Generate state names via NameGenerator
     3. Identify constant parameters and direct args
     4. Generate constant names via NameGenerator
     5. Classify variables via VariableClassifier
     6. Validate return statement count
     7. Create symbols via SymbolManager
     8. Final validation
     9. Return IndexedBases, symbol_map, classified_vars
     
     Handle different return formats: list, tuple, dict, single expression.
   - Edge cases:
     - Less than 2 function params: Raise ValueError in _identify_state_parameter
     - No return statement: Detected in _validate_return_statement
     - Return count mismatch: Raise ValueError showing expected vs actual
     - Missing visitor_results keys: Detected in __init__
     - Dict return format: Count keys in _validate_return_statement
   - Integration:
     - Receives VisitorResults from Section 1 OdeAstVisitor
     - Coordinates Groups 1, 2, 3 components
     - Produces IndexedBases and symbol_map for Section 3 FunctionParser
     - Final output format matches string parser expectations

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_variable_identifier.py
- Test function: test_full_workflow_int_subscripts - End-to-end with y[0], y[1]
- Test function: test_full_workflow_str_subscripts - End-to-end with y["name"]
- Test function: test_full_workflow_attributes - End-to-end with y.attr
- Test function: test_with_user_state_override - User states override inferred
- Test function: test_parameter_promotion - User params move from constants
- Test function: test_observable_extraction - Observables validated and extracted
- Test function: test_direct_constant_args - Direct function args become constants
- Test function: test_return_count_validation - Return value count checked
- Test function: test_dict_return_format - Dict return handled correctly
- Test function: test_insufficient_params_error - Less than 2 params raises ValueError
- Test function: test_missing_return_error - No return statement raises ValueError
- Test function: test_return_count_mismatch_error - Wrong return count raises ValueError
- Test function: test_validate_output_completeness - Final validation checks all structures
- Test function: test_integration_with_all_categories - States, params, consts, obs together

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

### Task Group Overview
- **Group 1**: NameGenerator - Convert access patterns to variable names (12 tests)
- **Group 2**: VariableClassifier - Categorize with user overrides (11 tests)
- **Group 3**: SymbolManager - Create IndexedBases and symbol mappings (13 tests)
- **Group 4**: VariableIdentifier - Orchestrate complete workflow (14 tests)

### Dependency Chain
```
Group 1 (NameGenerator)
    ↓
Group 2 (VariableClassifier) - depends on Group 1
    ↓
Group 3 (SymbolManager) - depends on Groups 1, 2
    ↓
Group 4 (VariableIdentifier) - depends on Groups 1, 2, 3
```

### Test Files Created
1. `tests/odesystems/symbolic/test_name_generator.py` - 12 tests
2. `tests/odesystems/symbolic/test_variable_classifier.py` - 11 tests
3. `tests/odesystems/symbolic/test_symbol_manager.py` - 13 tests
4. `tests/odesystems/symbolic/test_variable_identifier.py` - 14 tests

**Total**: 50 tests covering all four modules

### Integration Points
- **From Section 1**: AccessPattern and VisitorResults TypedDicts from OdeAstVisitor
- **To Section 3**: IndexedBases and symbol_map for FunctionParser
- **With Existing CuBIE**: 
  - Uses IndexedBases.from_user_inputs() from indexedbasemaps.py
  - Uses TIME_SYMBOL from parser.py
  - Follows ParsedEquations structure conventions
  - Integrates with existing SymbolicODE infrastructure
- **Internal**: NameGenerator → VariableClassifier → SymbolManager → VariableIdentifier

### Estimated Complexity
- **Group 1**: Low-Medium - Name generation with pattern validation
- **Group 2**: Medium - User specification handling, categorization logic, validation
- **Group 3**: Medium - IndexedBases integration, symbol mapping, auxiliary handling
- **Group 4**: Medium - Orchestration, multi-step validation, format handling

**Overall**: Medium complexity - bridges AST analysis with SymPy symbolic representation, handling multiple access patterns and user specifications
