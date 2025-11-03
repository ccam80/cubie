# Quick Start Guide: Using CuBIE's Custom Agents

This guide helps you get started with CuBIE's custom GitHub Copilot agents.

## Prerequisites

1. **GitHub Copilot**: Active subscription with agent support
2. **Environment Variables** (optional, for full MCP functionality):
   - `PERPLEXITY_API_KEY`: For external research (plan_new_feature)
   - `GITHUB_TOKEN`: Usually auto-configured in GitHub Copilot

## The Six Agents

### Planning & Architecture
- **plan_new_feature**: Research and create implementation plans
- **detailed_implementer**: Convert plans into detailed task lists

### Implementation & Review
- **do_task**: Execute specific implementation tasks
- **reviewer**: Critical analysis of completed work

### Documentation
- **docstring_guru**: API documentation enforcement
- **narrative_documenter**: User-facing documentation

## Common Workflows

### 1. New Feature Development (Full Workflow)

```
Step 1: Planning
@plan_new_feature I need to add [feature description]. 
Please research the best approach and create a plan.

Step 2: Review Plan
Review the files in .github/active_plans/[feature_name]/
- human_overview.md: Architecture and decisions
- agent_plan.md: Technical specification

Step 3: Detailed Implementation Planning
@detailed_implementer Review the plan in 
.github/active_plans/[feature_name]/ and create detailed tasks.

Step 4: Review Task List
Review task_list.md to understand the implementation sequence.

Step 5: Execute Tasks
@do_task Execute task group 1 from 
.github/active_plans/[feature_name]/task_list.md

Repeat for each task group (can parallelize where marked).

Step 6: Code Review
@reviewer Analyze the implementation in 
.github/active_plans/[feature_name]/ and suggest improvements.

Step 7: Address Review Feedback (if needed)
@do_task Apply the edits from review_report.md section [X].

Step 8: Documentation
@docstring_guru Enforce numpydoc standards for all modified files.

@narrative_documenter Create a how-to guide for using [feature_name].
```

### 2. Bug Fix (Simplified Workflow)

```
Step 1: Plan
@plan_new_feature Analyze this bug: [description]. 
Create a minimal fix plan.

Step 2: Implement
@do_task Implement the fix described in the plan.

Step 3: Document (if API changed)
@docstring_guru Update docstrings for changed functions.
```

### 3. Documentation Update Only

```
For API Documentation:
@docstring_guru Review and update docstrings in 
src/cubie/[module]/[file].py

For User Documentation:
@narrative_documenter Create a how-to guide for [topic].
```

### 4. Code Review Existing Work

```
@reviewer Analyze the implementation in [directory/files].
Compare against goals in [description].
```

## Agent Communication Patterns

### When Agents Ask Questions
Most agents (except do_task) will ask for clarification:
- **plan_new_feature**: Design choices, scope clarification
- **detailed_implementer**: Implementation approach, dependencies
- **reviewer**: Standard interpretations, severity of issues
- **docstring_guru**: Function purpose, unclear parameters
- **narrative_documenter**: Target audience, documentation scope

**do_task never asks** - it executes exactly as specified.

### Providing Context
Each agent needs different context:

**plan_new_feature**:
- Feature description or issue number
- Goals and constraints
- Relevant background

**detailed_implementer**:
- Path to agent_plan.md
- Confirmation to proceed

**do_task**:
- Path to task_list.md
- Task group number
- Nothing else (all context in task list)

**reviewer**:
- Path to completed work
- Original goals/plan

**docstring_guru**:
- Files or modules to review
- Specific functions (optional)

**narrative_documenter**:
- Topic to document
- Target audience level
- Document type (how-to, manual, readme)

## Tips for Success

### Planning Phase
1. **Be specific** in your feature request
2. **Review plans carefully** before proceeding
3. **Ask questions** during planning (cheaper than rework)
4. **Save research** (plan_new_feature has 1 Perplexity query limit)

### Implementation Phase
1. **Trust the task list** from detailed_implementer
2. **Execute task groups in order** (dependencies matter)
3. **Parallelize** where task list indicates PARALLEL
4. **Review outcomes** in task_list.md after each group

### Review Phase
1. **Take criticism seriously** (reviewer is harsh but right)
2. **Prioritize critical issues** first
3. **Don't skip review** (catches problems early)

### Documentation Phase
1. **Do API docs first** (docstring_guru)
2. **Then narrative docs** (narrative_documenter)
3. **Test examples** in documentation
4. **Cross-reference** related docs

## Example: Adding a New Integration Algorithm

```bash
# Step 1: Research and plan
@plan_new_feature I need to add the Radau IIA-5 implicit 
integration method to CuBIE. This is for stiff ODEs. Research 
the algorithm and create a plan for implementation following 
our existing integrator patterns.

# Wait for plan, review .github/active_plans/radau_iia5/

# Step 2: Create detailed tasks
@detailed_implementer Review the plan in 
.github/active_plans/radau_iia5/ and create detailed 
implementation tasks.

# Review task_list.md

# Step 3: Execute task groups sequentially
@do_task Execute task group 1 from 
.github/active_plans/radau_iia5/task_list.md

@do_task Execute task group 2 from 
.github/active_plans/radau_iia5/task_list.md

# ... continue for all task groups

# Step 4: Review the implementation
@reviewer Analyze the Radau IIA-5 implementation in 
.github/active_plans/radau_iia5/. Compare against the 
original goals and our existing integrator quality standards.

# If review suggests edits, apply them with do_task

# Step 5: Documentation
@docstring_guru Enforce numpydoc standards for all files 
in src/cubie/integrators/algorithms/ related to Radau IIA-5.

@narrative_documenter Create a how-to guide explaining when 
and how to use the Radau IIA-5 method for solving stiff ODEs 
in CuBIE. Include examples comparing it to other methods.
```

## Troubleshooting

### Agent Not Responding
- Check agent name is spelled correctly
- Ensure you're using @ mention syntax
- Verify Copilot is active

### MCP Server Errors
- Check environment variables set
- See `.github/agents/MCP_SETUP.md` for details
- Try disabling problematic server in `mcp.json`

### Agent Produces Unexpected Output
- Review agent instructions in `.github/agents/[agent].agent`
- Provide more specific context
- Ask the agent to clarify what it understood

### Task Dependencies Not Clear
- Ask detailed_implementer to explain dependency order
- Review task_list.md dependency annotations
- Execute task groups in specified order

## Advanced Usage

### Custom Task Groups
You can ask detailed_implementer to organize tasks differently:
```
@detailed_implementer Create tasks with emphasis on parallel 
execution opportunities.
```

### Partial Workflows
You can skip agents if appropriate:
```
# Quick fix without full planning
@do_task Make these specific changes: [detailed description]

# Documentation only
@docstring_guru [files]
@narrative_documenter [topic]
```

### Multiple Features
Use separate directories in `.github/active_plans/`:
```
.github/active_plans/
├── radau_iia5/
├── adaptive_timestep/
└── cupy_integration/
```

## Best Practices Checklist

Planning:
- [ ] Feature description is clear and specific
- [ ] Reviewed human_overview.md before proceeding
- [ ] Understand architectural impact
- [ ] Verified agent_plan.md is detailed enough

Implementation:
- [ ] Task list dependencies understood
- [ ] Executing in correct order
- [ ] Reviewing outcomes after each group
- [ ] Testing as you go

Review:
- [ ] All critical issues addressed
- [ ] Medium priority issues evaluated
- [ ] Review report kept for reference

Documentation:
- [ ] All modified functions have complete docstrings
- [ ] API reference updated
- [ ] User guide created/updated
- [ ] Examples tested and work

## Getting Help

- **Agent Behavior**: See `.github/agents/README.md`
- **MCP Configuration**: See `.github/agents/MCP_SETUP.md`
- **Implementation Details**: See `.github/agents/IMPLEMENTATION_SUMMARY.md`
- **Repository Conventions**: See `AGENTS.md`
- **General Copilot**: See `.github/copilot-instructions.md`

## Next Steps

1. Try a simple request with plan_new_feature
2. Follow through a complete workflow once
3. Experiment with different agent combinations
4. Provide feedback on what works well or needs improvement

Happy building with CuBIE's custom agents! 🚀
