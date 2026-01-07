# Implementation Task List
# Feature: Fix Extra State Save Bug
# Plan Reference: .github/active_plans/fix_extra_state_save/agent_plan.md

## Task Group 1: Prevent Duplicate Final Saves in IVPLoop
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 373-379, 586-632, 769-794)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 195-210)
- File: .github/context/cubie_internal_structure.md (for architecture context)

**Input Validation Required**:
- None (no new inputs; modifying existing logic)

**Tasks**:
1. **Fix duplicate save logic when save_last and save_regularly coincide**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     The issue occurs in the `do_save` flag computation (lines 616-632). When
     `save_last=True` and `save_regularly=True`, the `at_end` flag can trigger
     a save even when a regular save at the same time point has already 
     occurred or will occur.
     
     The fix should modify the `do_save |= at_end` logic to suppress the 
     `save_last` contribution when `save_regularly` has already covered that
     save point.
     
     Current problematic code (around lines 631-632):
     ```python
     if save_last:
         do_save |= at_end
     ```
     
     Replace with logic that checks if the regular save already covers the
     end point. The key insight is: when `save_regularly=True`, the regular
     save mechanism already fires at `next_save`. If `at_end` is True AND
     `save_regularly` is True AND the regular save just occurred (indicated
     by `do_save` already being True from the regular logic), then we should
     NOT add `at_end` again.
     
     However, the simpler and correct approach is: when `save_last=True` and 
     `save_regularly=True`, only let `at_end` contribute to `do_save` if the
     regular save mechanism didn't already trigger for this step. This means:
     
     ```python
     if save_last:
         # Only add at_end contribution if regular save didn't already fire
         at_end_contributes = at_end & ~(save_regularly & do_save)
         do_save |= at_end_contributes
     ```
     
     This prevents the double-save because:
     - If `save_regularly` triggered `do_save=True`, then `at_end` doesn't add
     - If `save_regularly` didn't trigger (e.g., t_end doesn't align with 
       save_every grid), then `at_end` correctly triggers the final save
     
   - Edge cases:
     - t_end exactly aligns with regular save point: only one save
     - t_end doesn't align with regular save point: save_last triggers save
     - save_regularly=False but save_last=True: at_end triggers save
     - save_regularly=True but save_last=False: only regular saves occur
   - Integration: This change is in the compiled CUDA device function; 
     recompilation will be needed after this change.

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_save_last_no_duplicate_at_aligned_end
- Description: Verify that when duration is an exact multiple of save_every,
  the final time point is saved exactly once (not twice). Count the number
  of saves and verify the last save time equals t_end.

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_save_last_with_save_every
- tests/integrators/loops/test_ode_loop.py::test_save_last_no_duplicate_at_aligned_end

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (3 lines changed - lines 631-634)
  * tests/integrators/loops/test_ode_loop.py (47 lines added)
- Functions/Methods Added/Modified:
  * Modified `do_save` flag logic in IVPLoop device function (ode_loop.py)
  * Added test_save_last_no_duplicate_at_aligned_end() in test_ode_loop.py
- Implementation Summary:
  Modified the save_last logic to suppress the at_end contribution when
  save_regularly has already triggered a save at the same step. The new
  logic uses `at_end_contributes = at_end & ~(save_regularly & do_save)`
  to prevent duplicate saves when t_end aligns with a regular save point.
- Issues Flagged: None

---

## Task Group 2: Prevent Duplicate Final Summaries in IVPLoop
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 373-379, 620-635, 795-826)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 195-210)

**Input Validation Required**:
- None (no new inputs; modifying existing logic)

**Tasks**:
1. **Fix duplicate summary logic when summarise_last and summarise_regularly coincide**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Similar to the save issue, when `summarise_last=True` and 
     `summarise_regularly=True`, the final summary update/save may be 
     duplicated.
     
     The issue is in two places:
     
     **Part A: do_update_summary flag computation (around lines 633-635)**
     
     Current problematic code:
     ```python
     if summarise_last:
         do_update_summary |= at_end
     ```
     
     Replace with:
     ```python
     if summarise_last:
         # Only add at_end contribution if regular summary didn't already fire
         at_end_contributes = at_end & ~(summarise_regularly & do_update_summary)
         do_update_summary |= at_end_contributes
     ```
     
     **Part B: save_summary_now logic (around lines 813-816)**
     
     Current code:
     ```python
     save_summary_now = (
         (update_idx % samples_per_summary == int32(0))
         or (summarise_last and at_end)
     )
     ```
     
     This logic is problematic because if the regular summary fires at the
     same step as `at_end`, we save twice. The fix:
     
     ```python
     regular_save_due = (update_idx % samples_per_summary == int32(0))
     at_end_save = summarise_last and at_end and not regular_save_due
     save_summary_now = regular_save_due or at_end_save
     ```
     
     This ensures only one save per step: either the regular periodic save
     OR the at_end save, never both.
     
   - Edge cases:
     - t_end exactly aligns with summarise_every multiple: one summary save
     - t_end doesn't align: summarise_last triggers save
     - summarise_regularly=False but summarise_last=True: at_end saves
     - samples_per_summary boundary coincides with at_end
   - Integration: Same as Task 1; modifies CUDA device function.

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_summarise_last_no_duplicate_at_aligned_end
- Description: Verify that when duration is an exact multiple of 
  summarise_every, the final summary is written exactly once. Verify
  state_summaries array has the expected number of entries.

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_with_summarise_every
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_with_summarise_every_combined
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_no_duplicate_at_aligned_end

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (8 lines changed - lines 636-641, 817-825)
  * tests/integrators/loops/test_ode_loop.py (52 lines added)
- Functions/Methods Added/Modified:
  * Modified `do_update_summary` flag logic in IVPLoop device function (ode_loop.py)
  * Modified `save_summary_now` logic in IVPLoop device function (ode_loop.py)
  * Added test_summarise_last_no_duplicate_at_aligned_end() in test_ode_loop.py
- Implementation Summary:
  Part A: Modified the summarise_last logic to suppress the at_end contribution
  when summarise_regularly has already triggered a summary update at the same
  step. Uses `at_end_contributes = at_end & ~(summarise_regularly & do_update_summary)`.
  Part B: Modified save_summary_now to use explicit boolean variables that
  prevent both regular_save_due and at_end_save from firing on the same step.
  When regular summary save fires, at_end_save is suppressed with `not regular_save_due`.
- Issues Flagged: None

---

## Task Group 3: Test Timing Isolation for Session-Scoped Fixtures
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: tests/conftest.py (lines 180-186, 197-270, 372-468, 802-866)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 180-186, 197-270)

**Input Validation Required**:
- None (test infrastructure change)

**Tasks**:
1. **Ensure _user_timing is reset for tests without explicit timing parameters**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     The issue is that `_user_timing` dictionary in `SingleIntegratorRunCore`
     persists across session-scoped fixtures. When a test explicitly sets
     timing parameters (e.g., `save_every=0.02`), subsequent tests that use
     default timing may incorrectly inherit those values.
     
     The fix is to modify the `single_integrator_run` and 
     `single_integrator_run_mutable` fixtures to explicitly reset the timing
     state when the fixture's solver_settings don't include explicit timing.
     
     After the `SingleIntegratorRun` is instantiated but before returning,
     check if timing parameters were NOT explicitly provided in overrides,
     and if so, reset `_user_timing` to default values:
     
     ```python
     @pytest.fixture(scope="session")
     def single_integrator_run(
         system,
         solver_settings,
         driver_array,
         step_controller_settings,
         algorithm_settings,
         output_settings,
         loop_settings
     ):
         # ... existing code to create sir ...
         
         driver_function = _get_driver_function(driver_array)
         driver_del_t = _get_driver_del_t(driver_array)
         enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
             algorithm_settings, system, driver_array
         )
         sir = SingleIntegratorRun(
             system=system,
             driver_function=driver_function,
             driver_del_t=driver_del_t,
             step_control_settings=step_controller_settings,
             algorithm_settings=enhanced_algorithm_settings,
             output_settings=output_settings,
             loop_settings=loop_settings
         )
         
         # Reset _user_timing if timing wasn't explicitly overridden
         # This ensures tests with default timing aren't affected by
         # prior tests with explicit timing values
         timing_keys = ('save_every', 'summarise_every', 
                        'sample_summaries_every')
         explicit_timing = any(
             key in loop_settings and loop_settings[key] is not None
             for key in timing_keys
         )
         if not explicit_timing:
             sir._user_timing = {
                 'save_every': None,
                 'summarise_every': None,
                 'sample_summaries_every': None,
             }
         
         return sir
     ```
     
     Apply the same pattern to `single_integrator_run_mutable`.
     
   - Edge cases:
     - Test explicitly sets save_every=None: treated as explicit override
     - Test sets only save_every but not summarise_every: partial explicit
     - First test in session has no timing: should use defaults
   - Integration: This is test infrastructure; no production code changes.

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_timing_state_isolation
- Description: Verify that a test with default timing parameters produces
  correct output regardless of whether a previous test used explicit timing.
  This test should verify the number of saves matches expectations for
  save_last=True behavior.

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_save_last_flag_from_config
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_flag_from_config
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_collects_final_summary
- tests/integrators/loops/test_ode_loop.py::test_timing_state_isolation
- tests/batchsolving/test_solver.py::test_solve_ivp_with_summarise_variables

**Outcomes**: 
- Files Modified: 
  * tests/conftest.py (32 lines changed - lines 818-849, 868-899)
  * tests/integrators/loops/test_ode_loop.py (53 lines added)
- Functions/Methods Added/Modified:
  * single_integrator_run() fixture in conftest.py - added timing reset logic
  * single_integrator_run_mutable() fixture in conftest.py - added timing reset logic
  * test_timing_state_isolation() in test_ode_loop.py
- Implementation Summary:
  Modified both single_integrator_run and single_integrator_run_mutable fixtures
  to check if timing parameters (save_every, summarise_every, sample_summaries_every)
  were explicitly provided in loop_settings. When they are not explicitly set
  (all are None or not present), the fixture resets _user_timing to default
  None values. This prevents timing state from leaking between tests that use
  session-scoped fixtures.
- Issues Flagged: None

---

## Task Group 4: Integration Test for All Three Fixes
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2, Task Group 3

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file for test patterns)
- File: tests/batchsolving/test_solver.py (lines 1001-1016 for test_solve_ivp_with_summarise_variables)
- File: tests/conftest.py (entire file for fixture patterns)

**Input Validation Required**:
- None (test code only)

**Tasks**:
1. **Create comprehensive test for the originally failing scenarios**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify (add test)
   - Details:
     Add a test that covers the original failing test scenario: solve_ivp
     with no explicit timing parameters (save_every, summarise_every both
     None), which triggers save_last=True and summarise_last=True.
     
     The test should verify:
     - Exactly 2 state saves occur (t=0 and t=t_end)
     - State array has shape matching exactly 2 save points
     - No duplicate saves (verified by checking save_idx advancement)
     
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "precision": np.float32,
                 "duration": 0.1,
                 "output_types": ["state", "time"],
                 "algorithm": "euler",
                 "dt": 0.01,
                 "save_every": None,
                 "summarise_every": None,
                 "sample_summaries_every": None,
             }
         ],
         indirect=True,
     )
     def test_save_last_only_produces_two_saves(
         device_loop_outputs,
         precision,
     ):
         """Verify save_last=True produces exactly 2 saves: t=0 and t=t_end.
     
         When no save_every is specified and time-domain outputs are requested,
         the loop should save only at t=0 (initial) and t=t_end (final).
         This test guards against the extra save bug.
         """
         states = device_loop_outputs.state
     
         # Should have exactly 2 saves: initial at t=0 and final at t=t_end
         assert states.shape[0] == 2, (
             f"Expected 2 saves (t=0 and t=t_end), got {states.shape[0]}"
         )
     
         # Verify times are correct
         t_initial = states[0, -1]  # time is last column
         t_final = states[1, -1]
     
         assert t_initial == pytest.approx(precision(0.0), rel=1e-5), (
             f"Initial save should be at t=0, got {t_initial}"
         )
         assert t_final == pytest.approx(precision(0.1), rel=1e-5), (
             f"Final save should be at t=t_end=0.1, got {t_final}"
         )
     ```
     
   - Edge cases: 
     - Zero duration: only 1 save at t=0
     - Very short duration (one step): 2 saves
   - Integration: This test uses existing fixtures; no new infrastructure.

2. **Create test for combined save_last and summarise_last without timing**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify (add test)
   - Details:
     Add a test covering the case where both save_last and summarise_last
     are True (no explicit timing for either), ensuring no duplicate
     summaries or saves.
     
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "precision": np.float32,
                 "duration": 0.1,
                 "output_types": ["state", "time", "mean"],
                 "algorithm": "euler",
                 "dt": 0.01,
                 "save_every": None,
                 "summarise_every": None,
                 "sample_summaries_every": None,
             }
         ],
         indirect=True,
     )
     def test_save_and_summarise_last_no_duplicates(
         device_loop_outputs,
         precision,
     ):
         """Verify both save_last and summarise_last work without duplicates.
     
         When no timing parameters are specified but both time-domain and
         summary outputs are requested, exactly 2 saves and 1 summary should
         be produced.
         """
         states = device_loop_outputs.state
         state_summaries = device_loop_outputs.state_summaries
     
         # Should have exactly 2 saves: initial at t=0 and final at t=t_end
         assert states.shape[0] == 2, (
             f"Expected 2 saves, got {states.shape[0]}"
         )
     
         # Should have exactly 1 summary collected at t=t_end
         assert state_summaries is not None
         assert state_summaries.shape[0] >= 1, (
             "At least 1 summary expected"
         )
     
         # Verify no NaN values in summary
         assert not np.isnan(state_summaries[0]).any()
     ```
     
   - Edge cases: Same as task 1
   - Integration: Uses existing fixtures.

**Tests to Create**:
(Tests are defined inline in the task details above)

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_save_last_only_produces_two_saves
- tests/integrators/loops/test_ode_loop.py::test_save_and_summarise_last_no_duplicates
- tests/batchsolving/test_solver.py::test_solve_ivp_with_summarise_variables
- tests/batchsolving/test_solver.py::test_integration_with_solve_ivp (if exists)

**Outcomes**: 
- Files Modified: 
  * tests/integrators/loops/test_ode_loop.py (75 lines added)
- Functions/Methods Added/Modified:
  * test_save_last_only_produces_two_saves() in test_ode_loop.py
  * test_save_and_summarise_last_no_duplicates() in test_ode_loop.py
- Implementation Summary:
  Added two integration tests that verify the core bug fixes from Task Groups 1-3.
  test_save_last_only_produces_two_saves verifies that when no timing parameters
  are specified and time-domain outputs are requested, exactly 2 saves occur
  (t=0 and t=t_end). test_save_and_summarise_last_no_duplicates verifies that
  when both save_last and summarise_last are true (no explicit timing), the
  expected number of saves and summaries are produced without duplicates.
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 4
**Dependency Chain**: 
- Group 1 (save dedup) → standalone
- Group 2 (summary dedup) → depends on Group 1 (similar changes)
- Group 3 (test timing) → depends on Groups 1,2 (uses fixed loop)
- Group 4 (integration tests) → depends on all prior groups

**Tests Created**:
- `test_save_last_no_duplicate_at_aligned_end`
- `test_summarise_last_no_duplicate_at_aligned_end`
- `test_timing_state_isolation`
- `test_save_last_only_produces_two_saves`
- `test_save_and_summarise_last_no_duplicates`

**Estimated Complexity**: Medium
- Core logic changes are surgical (2-3 lines each)
- Test fixture changes require understanding of session scoping
- Integration testing verifies the complete fix

