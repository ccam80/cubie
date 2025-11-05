# CuBIE GitHub Copilot Agents

This directory contains custom GitHub Copilot agents for the CuBIE project. These agents work together to plan, implement, and document new features through a structured workflow.

All agents are in **Markdown format with YAML front matter** per [GitHub's Custom Agents Configuration](https://docs.github.com/en/copilot/reference/custom-agents-configuration).

## Important: How Agents Interpret User Prompts

**All agents have been configured to decode user prompts correctly:**

- The user prompt describes the **problem, issue, feature, or user story** to work on
- Language like "fix this", "implement Y", "address this" identifies **WHAT** to work on
- The **actions each agent takes** are defined in the agent's profile, NOT in the user prompt
- Each agent extracts the problem context from the prompt, then follows its own role definition

This ensures agents adhere strictly to their defined roles and responsibilities.

## Agent Overview

### File Permissions Summary

Each agent has restricted file creation/edit permissions:

- **plan_new_feature**: `.github/active_plans/<feature>/user_stories.md`, `human_overview.md`, `agent_plan.md`
- **detailed_implementer**: `.github/active_plans/<feature>/task_list.md`
- **taskmaster**: `.github/active_plans/<feature>/task_list.md` (updates only)
- **do_task**: Files listed in assigned task group only
- **reviewer**: `.github/active_plans/<feature>/review_report.md`
- **docstring_guru**: Any `.py` files, `docs/` files, `.github/context/cubie_internal_structure.md`
- **narrative_documenter**: `docs/` directory only (plus `readme.md`)

All agents can **read** all files in the repository.

### Return After Levels

The pipeline supports automatic agent chaining via the `return_after` argument:

1. **plan_new_feature**: Returns `user_stories.md`, `human_overview.md`, `agent_plan.md`
2. **detailed_implementer**: Returns above + `task_list.md`
3. **taskmaster**: Returns above + `task_list.md` updated with implementation outcomes
4. **reviewer**: Returns above + `review_report.md`
5. **taskmaster_2**: Returns above + edits applied from review + updated `review_report.md`
6. **docstring_guru**: Returns above + source files with complete docstrings

**narrative_documenter** exists outside this pipeline and is called separately.

### 1. plan_new_feature
**Role**: Expert project manager and technical architect  
**Purpose**: User story creation, research, and implementation planning  
**MCP Tools**: GitHub (always), Perplexity deep_research (if requested), Playwright (web browsing), bash (for directory creation)
**Custom Agent Tools**: `detailed_implementer`
**Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`  
**Output**: `user_stories.md`, `human_overview.md`, `agent_plan.md` in `.github/active_plans/<feature_name>/`
**File Permissions**: Can create/edit directory and files in `.github/active_plans/<feature>/` only (creates directory first using bash)
**Downstream Agents**: Can call `detailed_implementer` when `return_after` > `plan_new_feature`

Creates comprehensive plans with:
- User stories and acceptance criteria (first step)
- Architecture diagrams and data flow
- Behavior and integration descriptions (not implementation details)
- Technical decisions and trade-offs
- Shortcut: minimal bug fix plans ready for do_task

### 2. detailed_implementer
**Role**: Operations manager and implementation planner  
**Purpose**: Convert architectural plans into detailed, executable tasks  
**MCP Tools**: GitHub, tree-sitter (optional), code-search (optional)
**Custom Agent Tools**: `taskmaster`
**Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`  
**Output**: `task_list.md` with function-level implementation tasks
**File Permissions**: Can create/edit `.github/active_plans/<feature>/task_list.md` only
**Downstream Agents**: Can call `taskmaster` when `return_after` > `detailed_implementer`

Creates:
- Complete function signatures
- Detailed implementation logic
- **Input Validation Required** section (exact validation to perform)
- Dependency-ordered task groups
- Context requirements for each task

### 3. taskmaster
**Role**: Project manager and task orchestrator  
**Purpose**: Manage execution of implementation plans through do_task agents  
**MCP Tools**: None
**Custom Agent Tools**: `do_task`, `reviewer`, `docstring_guru`
**Context**: `task_list.md` from detailed_implementer  
**Output**: Execution summaries, collated changes, ready-for-review implementation
**File Permissions**: Can update `.github/active_plans/<feature>/task_list.md` only (no source code edits)
**Downstream Agents**: Can call `do_task` (repeatedly), `reviewer` when `return_after` > `taskmaster`, `docstring_guru` when `return_after` = `docstring_guru`

Manages:
- Parallel and sequential task execution
- Dependency-ordered task group coordination
- Multiple do_task agent invocations
- Change collation and integration
- Progress tracking and issue flagging
- Handoff preparation for reviewer

### 4. do_task
**Role**: Senior developer and implementer  
**Purpose**: Execute tasks exactly as specified with educational comments  
**MCP Tools**: pytest (optional, only for added tests), linter (optional)
**Custom Agent Tools**: None (leaf node)
**Context**: `AGENTS.md`, `.github/copilot-instructions.md`  
**Output**: Git patches and updated `task_list.md` with outcomes
**File Permissions**: Can create/edit only files listed in assigned task group
**Downstream Agents**: None (leaf node in agent tree)

Executes:
- Code changes per specifications exactly
- Adds educational comments (not docstrings)
- Performs ONLY validation from "Input Validation Required"
- Runs tests ONLY when explicitly added with CUDASIM enabled
- Flags bugs/risks in outcomes (executes without deviation)

### 5. reviewer
**Role**: Critical code reviewer  
**Purpose**: Validate against user stories and analyze for quality  
**MCP Tools**: code-metrics (optional), coverage (optional)
**Custom Agent Tools**: `taskmaster` (for second invocation to apply edits)
**Context**: `AGENTS.md`, requires `agent_plan.md`, `human_overview.md`, `user_stories.md`  
**Output**: `review_report.md` with analysis and suggested edits
**File Permissions**: Can create/edit `.github/active_plans/<feature>/review_report.md` only
**Downstream Agents**: Can call `taskmaster` (second invocation for edits) when `return_after` > `reviewer`

Reviews for:
- User story validation (acceptance criteria met)
- Goal alignment
- Code duplication
- Unnecessary complexity
- Convention compliance
- Performance issues (buffer reuse, math vs memory)

### 6. docstring_guru
**Role**: API documentation specialist  
**Purpose**: Enforce numpydoc standards and maintain API reference docs  
**MCP Tools**: sphinx (optional), doctests (optional)
**Custom Agent Tools**: None (final step in pipeline)
**Context**: `AGENTS.md`  
**Output**: Updated docstrings, API reference files, internal structure updates
**File Permissions**: Can edit any `.py` files, `docs/` files, `.github/context/cubie_internal_structure.md`
**Downstream Agents**: None (typically last in pipeline; narrative_documenter is separate)

Enforces:
- Numpydoc format for all functions/classes
- Proper type hint placement (no types in Parameters if in signature)
- Processes inline comments (keeps helpful, summarizes general ones in Notes)
- Updates .rst API reference files (touched files only)
- Searches narrative docs, reports function usage (does not update narrative)
- Updates `.github/context/cubie_internal_structure.md` with insights
- Escapes all backslashes in docstrings

### 7. narrative_documenter
**Role**: Technical storyteller for user-facing documentation  
**Purpose**: Create concept-based user guides and how-to docs in RST  
**MCP Tools**: mermaid (optional), markdown-lint (optional)
**Custom Agent Tools**: None (independent of main pipeline)
**Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`  
**Output**: Documentation in **reStructuredText (.rst)** for Sphinx (Markdown only for readmes/summaries)
**File Permissions**: Can create/edit files in `docs/` directory only (plus `readme.md`)
**Downstream Agents**: None (exists outside main pipeline)
**Pipeline Position**: **INDEPENDENT** - called separately, not part of the main implementation pipeline

Creates:
- How-to guides (task-oriented, .rst)
- User manual sections (concept-based, .rst - see docs/source/user_guide/)
- README updates (markdown)
- Accepts function updates from docstring_guru
- Updates narrative docs when API changes affect them

## Workflow

### Pipeline Architecture with return_after

The agent pipeline supports automatic chaining via the `return_after` argument. Each agent can execute subsequent agents automatically based on this parameter.

**Pipeline Flow**:
```
plan_new_feature → detailed_implementer → taskmaster → reviewer → taskmaster (2nd) → docstring_guru
                                                                                          ↓
                                                                            narrative_documenter
                                                                            (called separately)
```

**Return After Levels**:

1. **return_after=plan_new_feature**: 
   - plan_new_feature creates outputs and stops
   - Returns: user_stories.md, human_overview.md, agent_plan.md

2. **return_after=detailed_implementer**: 
   - plan_new_feature → detailed_implementer (stops)
   - Returns: above + task_list.md

3. **return_after=taskmaster**: 
   - plan_new_feature → detailed_implementer → taskmaster (stops)
   - Returns: above + task_list.md with implementation outcomes

4. **return_after=reviewer**: 
   - plan_new_feature → detailed_implementer → taskmaster → reviewer (stops)
   - Returns: above + review_report.md

5. **return_after=taskmaster_2**: 
   - Full pipeline + taskmaster applies review edits (stops)
   - Returns: above + applied review edits + updated review_report.md

6. **return_after=docstring_guru**: 
   - Complete pipeline including review edits + docstring_guru (stops)
   - Returns: above + source files with complete docstrings

**narrative_documenter** is called separately, outside this pipeline.

### Standard Feature Development Flow

```
User Request with return_after parameter
    ↓
┌───────────────────────┐
│ plan_new_feature      │ → Creates user_stories.md (first step)
│                       │ → Research with GitHub (always)
│                       │ → Perplexity deep_research (if requested)
│                       │ → Playwright (web browsing)
│                       │ → Creates human_overview.md & agent_plan.md
│                       │ → If return_after > plan_new_feature:
│                       │   calls detailed_implementer via custom-agent
└───────────────────────┘
    ↓ (if return_after > plan_new_feature)
┌───────────────────────┐
│ detailed_implementer  │ → Reviews source code
│                       │ → Creates task_list.md with:
│                       │   - Input Validation Required
│                       │   - Dependency-ordered tasks
│                       │   - PARALLEL/SEQUENTIAL groups
│                       │ → If return_after > detailed_implementer:
│                       │   calls taskmaster via custom-agent
└───────────────────────┘
    ↓ (if return_after > detailed_implementer)
┌───────────────────────┐
│ taskmaster            │ → Manages entire implementation execution
│                       │ → Launches do_task agents in parallel/sequential
│                       │ → Tracks progress and dependencies
│                       │ → Collates all changes
│                       │ → If return_after > taskmaster:
│                       │   calls reviewer via custom-agent
│                       │
│   Orchestrates ↓      │
│                       │
│ ┌─────────────────┐   │
│ │ do_task agents  │   │ → Execute tasks EXACTLY as specified
│ │ (parallel &     │   │ → Add educational comments (not docstrings)
│ │  sequential)    │   │ → Perform ONLY specified validation
│ └─────────────────┘   │ → Flag bugs/risks in outcomes
└───────────────────────┘
    ↓ (if return_after > taskmaster)
┌───────────────────────┐
│ reviewer              │ → Validates against user_stories.md
│                       │ → Checks agent_plan.md & human_overview.md
│                       │ → Analyzes buffer reuse, math vs memory
│                       │ → Creates review_report.md
│                       │ → If return_after > reviewer AND has edits:
│                       │   calls taskmaster (2nd time) via custom-agent
└───────────────────────┘
    ↓ (if return_after > reviewer AND has edits)
┌───────────────────────┐
│ taskmaster (2nd run)  │ → Applies review edits via do_task agents
│                       │ → Updates review_report.md
│                       │ → If return_after = docstring_guru:
│                       │   calls docstring_guru via custom-agent
└───────────────────────┘
    ↓ (if return_after = docstring_guru)
┌───────────────────────┐
│ docstring_guru        │ → Processes inline comments
│                       │ → Summarizes general comments in Notes
│                       │ → Keeps helpful comments
│                       │ → Updates API reference (touched files)
│                       │ → Searches narrative docs for usage
│                       │ → Updates cubie_internal_structure.md
│                       │ → FINAL STEP - no downstream calls
└───────────────────────┘

Separate workflow (called independently):
┌───────────────────────┐
│ narrative_documenter  │ → Accepts function updates from docstring_guru
│                       │ → Creates RST docs (how-to, user guide)
│                       │ → Updates narrative docs if API changed
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
│ narrative_documenter  │ → For user-facing docs (independent)
└───────────────────────┘
```

## Usage Examples

### Using return_after for Automatic Pipeline Execution

The most powerful way to use the agents is with the `return_after` argument, which automatically executes the entire pipeline to a specific level:

**Complete implementation with review edits and docstrings**:
```
@plan_new_feature I need to add support for Rosenbrock-W integration methods
to CuBIE. Research the algorithm, review how our current integrators work,
and create a plan for implementation.

return_after: docstring_guru
```
This will:
1. plan_new_feature creates plans
2. detailed_implementer creates task_list.md
3. taskmaster executes all tasks via do_task agents
4. reviewer validates and suggests edits
5. taskmaster applies review edits
6. docstring_guru adds complete documentation

**Implementation without review**:
```
@plan_new_feature Add parameter validation to solve_ivp function.

return_after: taskmaster
```
This will:
1. plan_new_feature creates plans
2. detailed_implementer creates task_list.md
3. taskmaster executes all tasks
4. STOPS (no review)

**Just planning**:
```
@plan_new_feature Research options for implementing adaptive step size control.

return_after: plan_new_feature
```
This will:
1. plan_new_feature creates plans
2. STOPS (returns for user review)

### Manual Step-by-Step Workflow

You can also invoke agents manually at each step:

**Starting a New Feature**:

```
@plan_new_feature I need to add support for Rosenbrock-W integration methods
to CuBIE. Research the algorithm, review how our current integrators work,
and create a plan for implementation.
```

**Converting Plan to Tasks**:

```
@detailed_implementer Review the agent_plan.md in .github/active_plans/rosenbrock_w/
and create a detailed task list with function signatures and implementation steps.
```

**Executing Tasks**:

```
@taskmaster Execute the complete implementation plan in 
.github/active_plans/rosenbrock_w/task_list.md, managing all do_task agents
in parallel and sequential mode as specified.
```

Or for individual task groups:

```
@do_task Execute task group 3 from task_list.md in .github/active_plans/rosenbrock_w/
```

**Reviewing Implementation**:

```
@reviewer Analyze the completed implementation in .github/active_plans/rosenbrock_w/
against the original goals and suggest improvements.
```

**Updating Documentation**:

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
│   ├── user_stories.md       # User stories (from plan_new_feature, step 1)
│   ├── human_overview.md     # High-level plan (from plan_new_feature)
│   ├── agent_plan.md         # Technical spec (from plan_new_feature)
│   ├── task_list.md          # Implementation tasks (from detailed_implementer)
│   └── review_report.md      # Review analysis (from reviewer)
└── another_feature/
    └── ...
```

## Context Directory

`.github/context/` contains shared context for agents:

```
.github/context/
└── cubie_internal_structure.md  # Maintained by agents
                                  # Updated by docstring_guru
                                  # Used by plan_new_feature, detailed_implementer, narrative_documenter
```

## Agent File Format

All agents use **Markdown with YAML front matter**:

```markdown
---
name: agent_name
description: Brief description
---

# Agent Title

Instructions in Markdown format with sections describing:
- Role and expertise
- Process and workflow
- Tools and when to use them
- Output requirements
```

Tools are described within the Markdown instructions rather than in separate configuration:
- **GitHub**: Repository operations (plan_new_feature, detailed_implementer)
- **Perplexity deep_research**: External research (plan_new_feature, only if requested)
- **Playwright**: Web automation (plan_new_feature)
- **Custom Agent Tools**: Each agent lists specific custom agents it can invoke (e.g., plan_new_feature can invoke detailed_implementer, taskmaster can invoke do_task/reviewer/docstring_guru)
- **pytest**: Test running (do_task, only for added tests with CUDASIM)
- **sphinx**: Documentation validation (docstring_guru, optional)
- **mermaid**: Diagram generation (narrative_documenter, optional)

See individual agent files for detailed tool usage instructions.

## Best Practices

### For Users

1. **Use return_after for automation**: Specify `return_after` level to automatically execute the pipeline
2. **Start with plan_new_feature**: Always begin with user story creation and planning
3. **Choose the right return_after level**:
   - `plan_new_feature`: Just planning, want to review before implementation
   - `detailed_implementer`: Planning + task breakdown, want to review tasks
   - `taskmaster`: Planning + implementation, want to review code before review
   - `reviewer`: Planning + implementation + review, want to see review before edits
   - `taskmaster_2`: Planning + implementation + review + edits, want to review before docstrings
   - `docstring_guru`: Complete pipeline including docstrings
4. **Call narrative_documenter separately**: It's outside the main pipeline
5. **Manual invocation still supported**: Call agents individually if you need fine-grained control

### For Agents

These are enforced in agent instructions:

- **ALL AGENTS**: 
  * Decode user prompts correctly - extract the problem/feature, ignore action language
  * Follow role defined in profile, not user prompt language
  * Respect file permissions strictly
- **plan_new_feature**: 
  * Create user_stories.md FIRST
  * If return_after > plan_new_feature, call detailed_implementer via custom-agent
  * Describe behavior/architecture, not implementation details
- **detailed_implementer**: 
  * If return_after > detailed_implementer, call taskmaster via custom-agent
  * Specify exact "Input Validation Required"
  * Mark task groups as PARALLEL or SEQUENTIAL
- **taskmaster**: 
  * If return_after > taskmaster, call reviewer via custom-agent after completing tasks
  * For taskmaster_2 (second invocation), apply review edits then call docstring_guru if needed
  * Never implement code directly (only manage do_task agents)
  * Use custom-agent tool to invoke do_task agents
- **do_task**: 
  * No downstream agents (leaf node)
  * Execute exactly as specified
  * Perform ONLY validation from "Input Validation Required"
- **reviewer**: 
  * If return_after > reviewer AND has suggested edits, call taskmaster again via custom-agent
  * Validate against user_stories.md
  * Be harsh but fair
- **docstring_guru**: 
  * Final step in main pipeline
  * No downstream calls (narrative_documenter is separate)
  * Can edit any code or docs files
- **narrative_documenter**: 
  * Exists outside main pipeline
  * No downstream agents
  * Can only edit docs/ directory

## Repository Conventions

All agents understand CuBIE-specific conventions from repository-level instructions:

- **Code Style**: PEP8, 79 char lines, numpydoc docstrings
- **Type Hints**: In signatures only (except CUDA device functions: docstring only)
- **Platform**: PowerShell compatible (no `&&` in commands)
- **Testing**: Pytest fixtures; no mocks/patches; CUDASIM for GPU tests
- **Breaking Changes**: Acceptable (no backwards compatibility)
- **Documentation**: RST for Sphinx, numpydoc for API, concept-based user guides

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
