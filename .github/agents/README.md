# CuBIE GitHub Copilot Agents

This directory contains custom GitHub Copilot agents for the CuBIE project. These agents work together to plan, implement, and document new features through a structured workflow.

## Agent Overview

### 1. plan_new_feature
**Role**: Expert project manager and technical architect  
**Purpose**: Initial research and implementation planning  
**MCP Tools**: Perplexity (research), Playwright (web browsing), GitHub  
**Output**: `human_overview.md` and `agent_plan.md` in `.github/active_plans/<feature_name>/`

Creates comprehensive plans with:
- Architecture diagrams and data flow
- Component descriptions
- Integration points
- Technical decisions and trade-offs

### 2. detailed_implementer
**Role**: Operations manager and implementation planner  
**Purpose**: Convert architectural plans into detailed, executable tasks  
**MCP Tools**: GitHub  
**Output**: `task_list.md` with function-level implementation tasks

Creates:
- Complete function signatures
- Detailed implementation logic
- Dependency-ordered task groups
- Context requirements for each task

### 3. do_task
**Role**: Senior developer and implementer  
**Purpose**: Execute specific implementation tasks precisely  
**MCP Tools**: None (all context in task_list.md)  
**Output**: Git patches and updated `task_list.md` with outcomes

Executes:
- Code changes per specifications
- Following CuBIE conventions strictly
- Minimal, focused changes only
- Proper type hints and docstrings

### 4. reviewer
**Role**: Critical code reviewer  
**Purpose**: Analyze implementations for quality and simplification  
**MCP Tools**: None  
**Output**: `review_report.md` with analysis and suggested edits

Reviews for:
- Goal alignment
- Code duplication
- Unnecessary complexity
- Convention compliance
- Performance issues

### 5. docstring_guru
**Role**: Technical writing specialist  
**Purpose**: Enforce numpydoc standards and maintain API documentation  
**MCP Tools**: None  
**Output**: Updated docstrings and documentation files

Enforces:
- Numpydoc format for all functions/classes
- Proper type hint placement
- Complete module docstrings
- Updated .rst API reference files
- Consistency across narrative docs

### 6. narrative_documenter
**Role**: Technical storyteller  
**Purpose**: Create user-friendly how-to guides and manual sections  
**MCP Tools**: None  
**Output**: Narrative documentation in markdown

Creates:
- How-to guides (task-oriented)
- User manual sections (conceptual)
- README updates
- Accessible explanations with examples

## Workflow

### Standard Feature Development Flow

```
User Request
    ↓
┌───────────────────────┐
│ plan_new_feature      │ → Research with Perplexity, Playwright, GitHub
│                       │ → Creates human_overview.md & agent_plan.md
└───────────┬───────────┘
            ↓
    User approval
            ↓
┌───────────────────────┐
│ detailed_implementer  │ → Reviews source code
│                       │ → Creates task_list.md with ordered tasks
└───────────┬───────────┘
            ↓
    User approval
            ↓
┌───────────────────────┐
│ do_task               │ → Executes task groups
│ (multiple invocations)│ → Creates git patches
│                       │ → Updates task_list.md outcomes
└───────────┬───────────┘
            ↓
    All tasks complete
            ↓
┌───────────────────────┐
│ reviewer              │ → Critical analysis
│                       │ → Creates review_report.md
└───────────┬───────────┘
            ↓
    If edits needed
            ↓
┌───────────────────────┐
│ do_task               │ → Apply suggested edits
│ (additional rounds)   │
└───────────┬───────────┘
            ↓
    Implementation complete
            ↓
┌───────────────────────┐
│ docstring_guru        │ → Enforce documentation standards
│                       │ → Update API references
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ narrative_documenter  │ → Create user-friendly guides
│                       │ → Update how-to documentation
└───────────────────────┘
```

### Documentation-Only Flow

For documentation work without code changes:

```
User Request
    ↓
┌───────────────────────┐
│ docstring_guru        │ → For API doc enforcement
│          OR           │
│ narrative_documenter  │ → For user-facing docs
└───────────────────────┘
```

## Usage Examples

### Starting a New Feature

```
@plan_new_feature I need to add support for Rosenbrock-W integration methods
to CuBIE. Research the algorithm, review how our current integrators work,
and create a plan for implementation.
```

### Converting Plan to Tasks

```
@detailed_implementer Review the agent_plan.md in .github/active_plans/rosenbrock_w/
and create a detailed task list with function signatures and implementation steps.
```

### Executing Tasks

```
@do_task Execute task group 3 from task_list.md in .github/active_plans/rosenbrock_w/
```

### Reviewing Implementation

```
@reviewer Analyze the completed implementation in .github/active_plans/rosenbrock_w/
against the original goals and suggest improvements.
```

### Updating Documentation

```
@docstring_guru Review and enforce numpydoc standards for all files in
src/cubie/integrators/algorithms/

@narrative_documenter Create a how-to guide for using Rosenbrock-W methods
in CuBIE based on the new implementation.
```

## Active Plans Directory Structure

Each feature gets a directory in `.github/active_plans/`:

```
.github/active_plans/
├── feature_name/
│   ├── human_overview.md      # High-level plan (from plan_new_feature)
│   ├── agent_plan.md          # Technical spec (from plan_new_feature)
│   ├── task_list.md           # Implementation tasks (from detailed_implementer)
│   └── review_report.md       # Review analysis (from reviewer)
└── another_feature/
    └── ...
```

## MCP Configuration

MCP servers are configured in `.github/mcp.json`:

- **perplexity**: External research (requires `PERPLEXITY_API_KEY` env var)
- **playwright**: Web automation and browsing
- **github**: Repository operations (uses `GITHUB_TOKEN` env var)

## Best Practices

### For Users

1. **Start with plan_new_feature**: Always begin with research and planning
2. **Review before proceeding**: Check plans before moving to implementation
3. **Ask agents for clarification**: All agents (except do_task) will ask when uncertain
4. **Use task groups efficiently**: Run multiple do_task invocations in parallel where marked
5. **Don't skip review**: The reviewer catches issues before they become technical debt

### For Agents

These are enforced in agent instructions:

- **plan_new_feature**: Use Perplexity ONCE per feature (quota limit)
- **detailed_implementer**: Provide complete context (file paths, line numbers)
- **do_task**: Never deviate from specifications; execute exactly as planned
- **reviewer**: Be harsh but fair; every criticism must be actionable
- **docstring_guru**: Strictly enforce numpydoc; distinguish device functions
- **narrative_documenter**: Avoid jargon; explain all symbols in equations

## Repository Conventions

All agents are trained on CuBIE-specific conventions:

- **Code Style**: PEP8, 79 char lines, numpydoc docstrings
- **Type Hints**: In signatures only (except CUDA device functions: docstring only)
- **Attrs Classes**: Float attributes with underscore + property pattern
- **CUDAFactory**: Never call `build()` directly; use properties
- **Platform**: PowerShell compatible (no `&&` in commands)
- **Testing**: Pytest fixtures; no mocks/patches
- **Breaking Changes**: Acceptable (no backwards compatibility)

See `AGENTS.md` for complete architecture and style guidelines.

## Troubleshooting

### MCP Server Issues

If MCP servers fail to connect:
1. Check environment variables are set (`PERPLEXITY_API_KEY`, `GITHUB_TOKEN`)
2. Verify network connectivity
3. Check `.github/mcp.json` configuration
4. Try disabling problematic server by setting `"disabled": true`

### Agent Behavior Issues

If an agent doesn't follow instructions:
1. Check the `.agent` file for that agent
2. Verify you're providing required context
3. Try being more specific in your request
4. Reference the agent's role and purpose from this README

### Workflow Issues

If the workflow stalls:
1. Each agent should ask for approval before proceeding
2. do_task is the only agent that doesn't ask questions
3. Agents can be invoked out of order for special cases
4. Documentation agents can work independently of implementation flow

## Contributing

When modifying agent configurations:

1. **Update instructions carefully**: Agents rely on precise, explicit instructions
2. **Test with simple requests**: Verify behavior before complex workflows
3. **Maintain handoff clarity**: Ensure agents know when to pass work forward
4. **Document MCP changes**: Update this README if adding/changing MCP servers
5. **Follow conventions**: Agent files should match the established pattern

## Additional Resources

- **AGENTS.md**: Complete CuBIE architecture reference
- **.github/copilot-instructions.md**: Global Copilot instructions
- **docs/**: User-facing documentation
- **Active Plans**: Check `.github/active_plans/` for ongoing work
