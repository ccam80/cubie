# Test Case for Custom Agent Invocation

This test case verifies that custom agents can now invoke downstream custom agents after the `custom-agent` tool fix.

## Test Scenario

Invoke the `detailed_implementer` agent with a simple task that requires calling `taskmaster`.

## Test Command

```
@detailed_implementer Create a simple test task in .github/active_plans/agent_invocation_test/

return_after: taskmaster
```

## Expected Behavior

**Before fix**:
- detailed_implementer would report: "I don't have a 'taskmaster' tool available"
- Pipeline would break at the first subagent invocation

**After fix** (with `custom-agent` tool added):
- detailed_implementer creates task_list.md
- detailed_implementer successfully invokes taskmaster
- taskmaster receives the tool request
- taskmaster attempts to use `custom-agent` to invoke `do_task` agents
- Pipeline completes through to taskmaster level

## Verification Steps

1. Check logs for "Requested tools for custom agent" - should now include `custom-agent`
2. Check logs for "Custom agent tools after filtering" - should include custom agent names
3. Verify no errors about missing tools
4. Verify taskmaster was successfully invoked
5. Verify task_list.md was created and updated

## What to Look For in Logs

**Key log lines**:
```
Candidate tools for custom agent: [...should include custom-agent...]
Requested tools for custom agent: custom-agent, taskmaster, read, view, edit
Custom agent tools after filtering: [...should now include taskmaster and custom-agent if applicable...]
```

If the fix works, you should NOT see:
```
"I don't have a 'taskmaster' tool available"
```

Instead, you should see successful invocation of the downstream agent.

## Alternative Quick Test

Invoke taskmaster directly to see if it can now access do_task:

```
@taskmaster Execute a simple test at .github/active_plans/test/task_list.md

return_after: taskmaster
```

Check if taskmaster reports having access to `do_task` or if it successfully invokes it.
