---
name: pipeline-execution
description: Guide for orchestrating custom agents through the feature development workflow. Use this when asked to run the pipeline on an issue or execute the agent pipeline.
---

# Pipeline Execution Skill

This skill orchestrates the custom agent pipeline for feature development. The pipeline automates feature development through specialized agents.

**Important**: Pipeline coordination is handled by the **default Copilot agent** (you). Custom agents do NOT have the ability to invoke other custom agents. You are responsible for invoking each agent in the proper sequence based on the `return_after` parameter.

**CRITICAL: During pipeline execution, you (the default agent) should NOT read or edit source files directly**. Your role is purely coordination - invoke the appropriate agents and let them do the work. The exception is reading task_list.md to identify task groups for coordination. You do not have any control over which agents to run. The user will specify entry and return-after agents explicitly if they are different to the defaults. Never deviate from the explicitly ordered or default agents to call.

## Pipeline Execution Steps

When executing a pipeline command:

1. **Fetch the issue details** using GitHub tools to understand the request
2. **Invoke plan_new_feature agent** with the issue content
3. **Invoke detailed_implementer** with plan_new_feature outputs
4. **For each task group in task_list.md**:
   a. Invoke **taskmaster** for that specific task group
   b. Wait for taskmaster to complete
5. **Invoke run_tests** for complete test verification (once after all taskmasters complete)
6. **Invoke reviewer** to validate implementation
7. **If reviewer suggests edits**: invoke taskmaster for review edits (taskmaster_2)
8. **Invoke run_tests** at pipeline exit for final verification
9. **Report results and terminate** - summarize changes and any remaining test failures

## Pipeline Termination Rules

**CRITICAL**: The pipeline has a **maximum of 2 taskmaster invocations** after initial task group execution:
1. One optional taskmaster after first run_tests (step 5) if tests fail
2. One taskmaster_2 after reviewer (step 7) if reviewer suggests edits

**Do NOT loop** between run_tests and taskmaster. If tests still fail after taskmaster_2, report the failures to the user and terminate.

After step 8 (final run_tests), the pipeline MUST terminate regardless of test results.

Include any remaining test failures in the final summary for user review.

## Default Configuration

- **Default return_after level**: Use `taskmaster_2` for complete implementation
- **Default starting agent**: `plan_new_feature` unless specified otherwise

## Pipeline Levels

The pipeline executes through these levels (in order):

1. **plan_new_feature** - Creates user stories, overview, and architectural plan
2. **detailed_implementer** - Creates detailed task list with task groups
3. **taskmaster** - Executes one task group at a time (called per group)
4. **run_tests** - Runs tests once after all taskmaster invocations complete
5. **reviewer** - Reviews implementation against user stories
6. **taskmaster_2** - Applies review edits (final taskmaster invocation)
7. **run_tests_final** - Final test verification (no further fixes attempted)

## Example Execution

User says: "run pipeline on issue #123"

Your action:
```
1. Use github/issue_read to get issue #123 details
2. Invoke plan_new_feature agent with:
   - Prompt: "Issue #123 content and context"
3. Wait for plan_new_feature to complete
4. Invoke detailed_implementer with plan_new_feature outputs
5. Wait for detailed_implementer to complete
6. Read task_list.md to identify task groups (N groups)
7. For each task group 1 to N:
   a. Invoke taskmaster with: "Execute Task Group [i] from task_list.md"
   b. Wait for taskmaster to complete
8. Invoke run_tests for full test suite verification
9. Invoke reviewer with all outputs
10. Wait for reviewer to complete
11. If reviewer suggests edits, invoke taskmaster with review edits
12. Invoke run_tests for final verification
13. Summarize changes and any remaining test failures, then return to user
    - Do NOT invoke additional taskmaster or run_tests calls
    - Report failures for user to examine
```

## Important Notes

- Reading task_list.md to identify task groups is the only file reading you should do during pipeline execution
- Do NOT read source files or make edits yourself
- Trust the custom agents to perform their specialized work
- Your role is coordination only
