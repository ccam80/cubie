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
- **taskmaster**: Files listed in task groups, `.github/active_plans/<feature>/task_list.md` (updates), `.github/active_plans/<feature>/review_report.md` (updates)
- **reviewer**: `.github/active_plans/<feature>/review_report.md`
- **docstring_guru**: Any `.py` files, `docs/` files, `.github/context/cubie_internal_structure.md`
- **narrative_documenter**: `docs/` directory only (plus `readme.md`)
- **renamer**: `name_info.md` (tracking file), any `.py` files when executing renames

All agents can **read** all files in the repository.

### Return After Levels

The pipeline supports automatic agent chaining via the `return_after` argument:

1. **plan_new_feature**: Returns `user_stories.md`, `human_overview.md`, `agent_plan.md`
2. **detailed_implementer**: Returns above + `task_list.md`
3. **taskmaster**: Returns above + `task_list.md` updated with implementation outcomes
4. **run_tests**: Returns above + `test_results.md`
5. **reviewer**: Returns above + `review_report.md`
6. **taskmaster_2**: Returns above + edits applied from review + updated `review_report.md`

**docstring_guru**, **narrative_documenter**, and **renamer** exist outside this pipeline and are called separately.

### 1. plan_new_feature
**Role**: Expert project manager and technical architect  
**Purpose**: User story creation, research, and implementation planning  
**MCP Tools**: GitHub (always), Playwright (web browsing), bash (for directory creation)
**Context**: `.github/context/cubie_internal_structure.md`, `.github/copilot-instructions.md`  
**Output**: `user_stories.md`, `human_overview.md`, `agent_plan.md` in `.github/active_plans/<feature_name>/`
**File Permissions**: Can create/edit directory and files in `.github/active_plans/<feature>/` only (creates directory first using bash)

Creates comprehensive plans with:
- User stories and acceptance criteria (first step)
- Behavior and integration descriptions (not implementation details)
- Technical decisions and trade-offs

### 2. detailed_implementer
**Role**: Operations manager and implementation planner  
**Purpose**: Convert architectural plans into detailed, executable tasks  
**MCP Tools**: GitHub
**Context**: `.github/context/cubie_internal_structure.md`, `.github/copilot-instructions.md`  
**Output**: `task_list.md` with function-level implementation tasks
**File Permissions**: Can create/edit `.github/active_plans/<feature>/task_list.md` only

Creates:
- Complete function signatures
- Detailed implementation logic
- **Input Validation Required** section (exact validation to perform)
- Dependency-ordered task groups
- Context requirements for each task

### 3. taskmaster
**Role**: Senior developer and implementation executor  
**Purpose**: Execute implementation plans by performing tasks in parallel and sequential order  
**MCP Tools**: None
**Context**: `task_list.md` from detailed_implementer  
**Output**: Execution summaries, completed implementation, ready-for-review code
**File Permissions**: Can create/edit files listed in task groups, update `.github/active_plans/<feature>/task_list.md`

Executes:
- Parallel and sequential task groups
- Dependency-ordered task implementation
- Direct code changes per specifications
- Progress tracking and issue flagging
- Handoff preparation for reviewer

### 4. reviewer
**Role**: Critical code reviewer  
**Purpose**: Validate against user stories and analyze for quality  
**MCP Tools**: None
**Context**: `.github/copilot-instructions.md`, requires `agent_plan.md`, `human_overview.md`, `user_stories.md`  
**Output**: `review_report.md` with analysis and suggested edits
**File Permissions**: Can create/edit `.github/active_plans/<feature>/review_report.md` only

Reviews for:
- User story validation (acceptance criteria met)
- Goal alignment
- Code duplication
- Unnecessary complexity
- Convention compliance
- Performance issues (buffer reuse, math vs memory)

### 5. docstring_guru
**Role**: API documentation specialist  
**Purpose**: Enforce numpydoc standards and maintain API reference docs  
**MCP Tools**: None
**Context**: `.github/copilot-instructions.md`  
**Output**: Updated docstrings, API reference files, internal structure updates
**File Permissions**: Can edit any `.py` files, `docs/` files, `.github/context/cubie_internal_structure.md`

Enforces:
- Numpydoc format for all functions/classes
- Proper type hint placement (no types in Parameters if in signature)
- Processes inline comments (keeps helpful, summarizes general ones in Notes)
- Updates .rst API reference files (touched files only)
- Searches narrative docs, reports function usage (does not update narrative)
- Updates `.github/context/cubie_internal_structure.md` with insights
- Escapes all backslashes in docstrings

### 6. narrative_documenter
**Role**: Technical storyteller for user-facing documentation  
**Purpose**: Create concept-based user guides and how-to docs in RST  
**MCP Tools**: None
**Context**: `.github/context/cubie_internal_structure.md`, `.github/copilot-instructions.md`  
**Output**: Documentation in **reStructuredText (.rst)** for Sphinx (Markdown only for readmes/summaries)
**File Permissions**: Can create/edit files in `docs/` directory only (plus `readme.md`)
**Pipeline Position**: **INDEPENDENT** - called separately, not part of the main implementation pipeline

Creates:
- How-to guides (task-oriented, .rst)
- User manual sections (concept-based, .rst - see docs/source/user_guide/)
- README updates (markdown)
- Accepts function updates from docstring_guru
- Updates narrative docs when API changes affect them

### 7. run_tests
**Role**: Test execution and reporting specialist  
**Purpose**: Run pytest with CUDA simulation and provide failure summaries  
**MCP Tools**: bash
**Context**: `task_list.md` (for tests to run)  
**Output**: Test result summaries with failure details
**File Permissions**: Read-only except can create/edit `.github/active_plans/<feature_name>/test_results.md`
**Pipeline Position**: Called once after all taskmaster invocations complete, before reviewer, and at pipeline exit

Provides:
- Runs tests with NUMBA_ENABLE_CUDASIM=1
- Excludes nocudasim and specific_algos tests by default
- Clear failure summaries (error message, not full tracebacks)
- Actionable recommendations for failures

### 8. renamer
**Role**: Name rationalization specialist  
**Purpose**: Manage and rationalize method, function, property, and attribute names  
**MCP Tools**: bash, search
**Context**: `name_info.md` (tracking file)  
**Output**: Updated `name_info.md`, renamed source files
**File Permissions**: Can create/edit `name_info.md`, edit any `.py` files when executing renames
**Pipeline Position**: **INDEPENDENT** - called separately for name rationalization work

Performs:
- **update_list**: Scan files and document all names in name_info.md
- **recommend**: Analyze names and suggest improvements based on conventions
- **rename**: Execute recommended renames across the codebase
- Tracks recommendations and execution status
- Verifies completeness of renames across repository

## Workflow

### Pipeline Architecture with return_after

The agent pipeline is coordinated by the **default Copilot agent** (not by the custom agents themselves). The default agent invokes each custom agent in sequence based on the `return_after` parameter specified by the user.

**Pipeline Flow (coordinated by default Copilot agent)**:
```
plan_new_feature → detailed_implementer → [taskmaster (per group)] → run_tests → reviewer → taskmaster (edits) → run_tests
```

**Return After Levels**:

1. **return_after=plan_new_feature**: 
   - plan_new_feature creates outputs and stops
   - Returns: user_stories.md, human_overview.md, agent_plan.md

2. **return_after=detailed_implementer**: 
   - plan_new_feature → detailed_implementer (stops)
   - Returns: above + task_list.md

3. **return_after=taskmaster**: 
   - plan_new_feature → detailed_implementer → [taskmaster per group] → run_tests (stops)
   - Returns: above + task_list.md with implementation outcomes + test_results.md

4. **return_after=reviewer**: 
   - plan_new_feature → detailed_implementer → [taskmaster per group] → run_tests → reviewer (stops)
   - Returns: above + review_report.md

5. **return_after=taskmaster_2**: 
   - Full pipeline + taskmaster applies review edits + run_tests (stops)
   - Returns: above + applied review edits + updated review_report.md + test_results.md

**renamer** is called separately, outside this pipeline.

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
└───────────────────────┘
    ↓ (if return_after > plan_new_feature)
    ↓ (default Copilot agent invokes next agent)
┌───────────────────────┐
│ detailed_implementer  │ → Reviews source code
│                       │ → Creates task_list.md with:
│                       │   - Input Validation Required
│                       │   - Dependency-ordered task groups
│                       │   - Tests to Create & Tests to Run
└───────────────────────┘
    ↓ (if return_after > detailed_implementer)
    ↓ (default Copilot agent loops for each task group)
┌───────────────────────────────────────────────────────┐
│ FOR EACH TASK GROUP:                                  │
│ ┌───────────────────────┐                            │
│ │ taskmaster            │ → Executes one task group  │
│ │                       │ → Reads context files      │
│ │                       │ → Writes code changes      │
│ │                       │ → Creates tests            │
│ │                       │ → Updates task_list.md     │
│ └───────────────────────┘                            │
└───────────────────────────────────────────────────────┘
    ↓ (after all task groups complete)
┌───────────────────────┐
│ run_tests             │ → Full test verification
│                       │ → Saves test_results.md
│                       │ → Before reviewer
└───────────────────────┘
    ↓ (if return_after > taskmaster)
    ↓ (default Copilot agent invokes next agent)
┌───────────────────────┐
│ reviewer              │ → Validates against user_stories.md
│                       │ → Checks agent_plan.md & human_overview.md
│                       │ → Analyzes buffer reuse, math vs memory
│                       │ → Creates review_report.md
└───────────────────────┘
    ↓ (if return_after > reviewer AND has edits)
    ↓ (default Copilot agent invokes taskmaster for edits)
┌───────────────────────┐
│ taskmaster (edits)    │ → Applies review edits directly
│                       │ → Updates review_report.md
└───────────────────────┘
    ↓
┌───────────────────────┐
│ run_tests             │ → Final test verification
│                       │ → Saves test_results.md
└───────────────────────┘

Separate workflow (called independently by default Copilot agent):
┌───────────────────────┐
│ renamer               │ → Manages name_info.md tracking file
│                       │ → Scans files and documents all names (update_list)
│                       │ → Recommends better names (recommend)
│                       │ → Executes renames across codebase (rename)
└───────────────────────┘
```

### Name Rationalization Flow

For improving method, function, property, and attribute names:

```
User Request (e.g., "run renamer on src/cubie/integrators")
    ↓
┌───────────────────────┐
│ renamer               │ → update_list: Scan and document all names
│ (update_list)         │ → Creates/updates name_info.md
└───────────────────────┘
    ↓
┌───────────────────────┐
│ renamer               │ → recommend: Analyze and suggest better names
│ (recommend)           │ → Updates name_info.md with recommendations
└───────────────────────┘
    ↓
┌───────────────────────┐
│ renamer               │ → rename: Execute renames in source files
│ (rename)              │ → Updates source code and name_info.md
└───────────────────────┘
```

## Usage Examples

### Using return_after for Automatic Pipeline Execution

The most powerful way to use the agents is with the `return_after` argument, which automatically executes the entire pipeline to a specific level:

**Complete implementation with review edits**:
```
@plan_new_feature I need to add support for Rosenbrock-W integration methods
to CuBIE. Research the algorithm, review how our current integrators work,
and create a plan for implementation.

return_after: taskmaster_2
```
This will:
1. plan_new_feature creates plans
2. detailed_implementer creates task_list.md
3. taskmaster executes all task groups
4. run_tests verifies tests
5. reviewer validates and suggests edits
6. taskmaster applies review edits
7. run_tests verifies final state

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
.github/active_plans/rosenbrock_w/task_list.md, implementing all task groups
in parallel and sequential mode as specified.
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

**Rationalizing Names**:

```
@renamer Run the complete name rationalization workflow on src/cubie/integrators.
Operation: update_list
Target: /home/runner/work/cubie/cubie/src/cubie/integrators
```

Then after it completes:

```
@renamer Continue with recommendations.
Operation: recommend
Chunk size: 10
```

Then after it completes:

```
@renamer Execute the renames.
Operation: rename
Chunk size: 5
```

Or run all three operations in sequence by using the default Copilot agent's automation (see `.github/copilot-instructions.md`).

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
- **bash**: Test running (run_tests with CUDASIM)
- **sphinx**: Documentation validation (docstring_guru, optional)
- **mermaid**: Diagram generation (narrative_documenter, optional)
- **bash/search**: Repository scanning and verification (renamer)

**Note**: Custom agents do NOT have the ability to invoke other custom agents. Pipeline coordination is handled by the default Copilot agent as described in `.github/copilot-instructions.md`.

See individual agent files for detailed tool usage instructions.

## Best Practices

### For Users

1. **Use return_after for automation**: Specify `return_after` level to automatically execute the pipeline
2. **Start with plan_new_feature**: Always begin with user story creation and planning
3. **Choose the right return_after level**:
   - `plan_new_feature`: Just planning, want to review before implementation
   - `detailed_implementer`: Planning + task breakdown, want to review tasks
   - `taskmaster`: Planning + implementation + tests, want to review code before review
   - `reviewer`: Planning + implementation + tests + review, want to see review before edits
   - `taskmaster_2`: Planning + implementation + review + edits + tests (complete pipeline)
4. **Call docstring_guru, narrative_documenter, and renamer separately**: They exist outside the main pipeline
5. **Manual invocation still supported**: Call agents individually if you need fine-grained control

### For Agents

These are enforced in agent instructions:

- **ALL AGENTS**: 
  * Decode user prompts correctly - extract the problem/feature, ignore action language
  * Follow role defined in profile, not user prompt language
  * Respect file permissions strictly
  * Do NOT attempt to invoke other custom agents - pipeline coordination is handled by the default Copilot agent
- **plan_new_feature**: 
  * Create user_stories.md FIRST
  * Describe behavior/architecture, not implementation details
- **detailed_implementer**: 
  * Specify exact "Input Validation Required"
  * Provide explicit context paths per task group
  * Include "Tests to Create" and "Tests to Run" sections
- **taskmaster**: 
  * Implement code directly
  * Execute one task group at a time (fresh context per group)
  * Create tests as specified, never mark skip/nocudasim/specific_algos
  * Do NOT run tests - run_tests agent handles this
- **run_tests**:
  * Runs tests with NUMBA_ENABLE_CUDASIM=1
  * Excludes nocudasim and specific_algos by default
  * Provides failure summaries, not full tracebacks
  * Saves test_results.md for the next agent
- **reviewer**: 
  * Validate against user_stories.md
  * Be harsh but fair
- **docstring_guru**: 
  * Exists outside main pipeline
  * Can edit any code or docs files
  * Process and summarize inline comments appropriately
- **narrative_documenter**: 
  * Exists outside main pipeline
  * Can only edit docs/ directory
- **renamer**:
  * Exists outside main pipeline
  * Three operations: update_list, recommend, rename
  * Tracks all changes in name_info.md
  * Verifies completeness of renames

## Repository Conventions

All agents understand CuBIE-specific conventions from repository-level instructions:

- **Code Style**: PEP8, 79 char lines, numpydoc docstrings
- **Type Hints**: In signatures only (except CUDA device functions: docstring only)
- **Platform**: PowerShell compatible (no `&&` in commands)
- **Testing**: Pytest fixtures; no mocks/patches; CUDASIM for GPU tests
- **Breaking Changes**: Acceptable (no backwards compatibility)
- **Documentation**: RST for Sphinx, numpydoc for API, concept-based user guides

See `.github/copilot-instructions.md` for complete style guidelines and `.github/context/cubie_internal_structure.md` for architecture.

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
2. taskmaster doesn't ask questions - it executes as specified
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

- **.github/copilot-instructions.md**: Complete Copilot instructions and style guidelines
- **.github/context/cubie_internal_structure.md**: CuBIE architecture reference
- **docs/**: User-facing documentation
- **Active Plans**: Check `.github/active_plans/` for ongoing work
