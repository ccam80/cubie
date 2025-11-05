# Custom Agent Invocation Investigation

## Problem Statement

Custom agents (subagents) are reporting they don't have access to other custom agents listed in their `tools` configuration, preventing the designed agent pipeline from functioning.

**Example from logs**:
- `detailed_implementer` lists `taskmaster` in its tools
- When invoked, `detailed_implementer` reports: "I don't have a 'taskmaster' tool available in my tool list"

## Investigation Results

### Test Performed

Created a test task list and invoked the `taskmaster` agent directly from the main Copilot agent.

**Result**: ✅ **Successful invocation**
- The taskmaster agent was successfully called
- The agent executed and completed its work
- **However**: The taskmaster reported it did NOT have access to the `do_task` tool

### Root Cause

**The issue is a configuration mismatch between how custom agents are defined and how they are made available at runtime.**

1. **Agent Configuration (YAML frontmatter)**:
   ```yaml
   ---
   name: detailed_implementer
   tools:
     - taskmaster  # Listed as a tool
     - do_task
     - other_tools
   ---
   ```

2. **Runtime Behavior**:
   - When a custom agent is invoked as a subagent, it does NOT receive other custom agents in its tool list
   - Only standard tools (view, edit, create, read, search) are available
   - Custom agent tools are NOT passed through to subagents

3. **Current Workaround**:
   - The main Copilot agent CAN invoke custom agents (as demonstrated)
   - Custom agents CANNOT invoke other custom agents (tools not available)

## Architecture Impact

The designed agent pipeline relies on a hierarchical structure:

```
plan_new_feature → detailed_implementer → taskmaster → do_task (multiple)
                                              ↓
                                          reviewer → docstring_guru
```

**Current Status**:
- ✅ Main agent → plan_new_feature: Works
- ✅ Main agent → detailed_implementer: Works  
- ✅ Main agent → taskmaster: Works
- ❌ detailed_implementer → taskmaster: **FAILS** (tool not available)
- ❌ taskmaster → do_task: **FAILS** (tool not available)
- ❌ taskmaster → reviewer: **FAILS** (tool not available)

## Instructions Review

### Main Agent Instructions (this agent)

From the prompt:
```
<custom_agents>
* Custom agents are implemented as tools that you can use. You can tell if a tool is a custom agent because the description will start with "Custom agent:".
* **ALWAYS** try to delegate tasks to custom agents when one is available
* Custom agents have a user-defined prompt and their own private context window
```

### Subagent Instructions

Each agent file (e.g., `detailed_implementer.md`) contains:
```markdown
## Downstream Agents

You have access to invoke the following downstream agent:

- **taskmaster**: Call when `return_after` is set to `taskmaster` or later.
```

**The instructions TELL the agent it has access, but the runtime environment does NOT provide it.**

## Additional Log Analysis

From the problem statement logs:

```
2025-11-05T09:34:00.8230484Z   COPILOT_FEATURE_FLAGS: 
  copilot_swe_agent_use_subagents,
  copilot_swe_agent_use_subagents_as_tools
```

These feature flags suggest the capability exists but may require additional configuration.

```
tools: github/get_file_contents, github/search_code, github/list_commits, 
       github/get_commit, taskmaster, read, edit, create, view, search

Additional custom agents available: detailed_implementer, do_task, 
       docstring_guru, narrative_documenter, plan_new_feature, reviewer, taskmaster
```

This shows:
- `taskmaster` appears in BOTH the tools list AND the custom agents list
- The main agent sees it as available
- But subagents don't receive these tools

## Recommended Solution

Based on this investigation, I recommend **ONE** of the following approaches:

### Option 1: Remove Subagent Invocation (Flatten Hierarchy)

**Change**: Update all agent profiles to remove downstream agent invocations.

**Implementation**:
1. Remove custom agent tools from `tools` lists in YAML frontmatter
2. Update agent instructions to NOT invoke downstream agents
3. Update README to show manual invocation only
4. Users invoke each agent in sequence manually

**Pros**:
- Works with current runtime environment
- Simple, predictable behavior
- No configuration issues

**Cons**:
- Loses automatic pipeline execution
- More manual steps for users
- Designed workflow doesn't function as intended

### Option 2: Fix Runtime Configuration (Recommended)

**Change**: Investigate and fix the runtime environment to pass custom agent tools to subagents.

**Investigation needed**:
1. Check if `copilot_swe_agent_use_subagents_as_tools` feature flag requires activation
2. Review GitHub's custom agent configuration documentation
3. Determine if YAML frontmatter format is correct
4. Check if there's additional configuration needed in repository settings

**Pros**:
- Maintains designed workflow
- Automatic pipeline execution works as intended
- Better user experience

**Cons**:
- Requires environment/configuration changes
- May be outside repository control
- Needs GitHub support or documentation

### Option 3: Hybrid Approach

**Change**: Keep automatic invocation for the main agent only.

**Implementation**:
1. Main agent invokes custom agents based on `return_after`
2. Main agent handles the pipeline coordination
3. Remove downstream invocation from subagent profiles
4. Keep subagents focused on their specific tasks

**Pros**:
- Automatic pipeline still works (via main agent)
- Simpler subagent implementation
- Works with current runtime

**Cons**:
- Main agent needs more complex pipeline logic
- Doesn't match the designed distributed architecture

## Immediate Next Steps

1. **User Decision Required**: Which approach should we take?

2. **For Option 1**: I can update all agent profiles immediately

3. **For Option 2**: Need to:
   - Check GitHub documentation for custom agents
   - Review feature flag requirements
   - Potentially contact GitHub support
   - May require repository settings changes

4. **For Option 3**: I can implement the main agent pipeline coordinator

## Questions for User

1. Do you have access to GitHub repository settings or GitHub support to investigate Option 2?

2. Are the feature flags (`copilot_swe_agent_use_subagents_as_tools`) something you control or are they set by GitHub?

3. What is your preference: maintain the designed hierarchical architecture (Option 2/3) or simplify to manual invocation (Option 1)?

4. Should I examine any additional log files or configuration that might reveal how to enable subagent-to-subagent invocation?
