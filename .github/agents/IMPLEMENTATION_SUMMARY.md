# CuBIE Custom Agents Implementation Summary

This document summarizes the custom GitHub Copilot agents created for the CuBIE repository in JSON format per GitHub's agent specification, with integrated MCP tool configurations.

## Agent File Format

All agents use JSON format with the following structure:
```json
{
  "name": "agent_name",
  "description": "Brief description",
  "instructions": "Detailed instructions...",
  "context": ["file1.md", "file2.md"],
  "tools": {
    "tool_name": {
      "enabled": true/false,
      "description": "When and how to use this tool"
    }
  }
}
```

## Created Agents

All agents are located in `.github/agents/` with the `.agent` extension (JSON format).

### 1. plan_new_feature.agent ✓
- **Role**: Expert project manager and technical architect
- **MCP Tools**: github (always), perplexity deep_research (if requested), playwright
- **Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`
- **Outputs**: `user_stories.md`, `human_overview.md`, `agent_plan.md`
- **Key Features**:
  - **Creates user stories FIRST** (step 1 of planning)
  - Perplexity deep_research ONLY when explicitly requested in prompt
  - Creates architecture diagrams using Mermaid syntax
  - agent_plan.md describes behavior/architecture, NOT implementation details
  - Shortcut route: minimal bug fixes can create detailed plan directly for do_task
  - Asks for user feedback on ambiguity
  - Works out of `.github/active_plans/<feature_name>/` directory

### 2. detailed_implementer.agent ✓
- **Role**: Operations manager and implementation planner
- **MCP Tools**: github, tree-sitter (optional), code-search (optional)
- **Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`
- **Outputs**: `task_list.md` with dependency-ordered tasks
- **Key Features**:
  - Provides complete function signatures
  - **Specifies "Input Validation Required"** field for each task group
  - Organizes tasks by dependency (architecture → core → integration → tests)
  - Marks groups as SEQUENTIAL or PARALLEL
  - Includes explicit context (files and line numbers) for each task group
  - Asks for clarification when faced with design choices

### 3. do_task.agent ✓
- **Role**: Senior developer and implementer
- **MCP Tools**: pytest (optional, only for added tests), linter (optional)
- **Context**: `AGENTS.md`, `.github/copilot-instructions.md`
- **Outputs**: Git patches and updated task_list.md with outcomes
- **Key Features**:
  - Executes tasks exactly as specified (no deviations)
  - **Adds educational comments in function bodies** (NOT docstrings)
  - **Performs ONLY validation from "Input Validation Required"** - no extra validation
  - **Runs tests ONLY when explicitly added** as part of task, with NUMBA_ENABLE_CUDASIM=1
  - **Flags bugs/risks in outcomes** (not deviations from spec)
  - Never asks user for feedback (executes the plan)
  - Follows repository conventions from repo-level instructions

### 4. reviewer.agent ✓
- **Role**: Harsh critic and senior code reviewer
- **MCP Tools**: code-metrics (optional), coverage (optional)
- **Context**: `AGENTS.md`, requires `agent_plan.md`, `human_overview.md`, `user_stories.md`
- **Outputs**: `review_report.md` with analysis and suggested edits
- **Key Features**:
  - **Validates implementation against user_stories.md** and acceptance criteria
  - Checks against agent_plan.md and human_overview.md goals
  - Identifies code duplication relentlessly
  - Finds unnecessary additions that don't serve user stories/goals
  - Suggests simplifications
  - **Checks for buffer reuse opportunities**
  - **Identifies math vs memory trade-offs** (few math ops vs memory access)
  - Provides actionable, specific feedback with file/line references
  - Can hand edits back to do_task agents

### 5. docstring_guru.agent ✓
- **Role**: API documentation specialist
- **MCP Tools**: sphinx (optional), doctests (optional)
- **Context**: `AGENTS.md`
- **Outputs**: Updated docstrings, API reference files, internal structure updates
- **Key Features**:
  - Enforces numpydoc format for all functions/classes
  - **Reads function including inline/block comments**
  - **Keeps helpful comments**, removes general description comments
  - **Summarizes general comments in Notes section** of docstring
  - **Checks docstring accuracy** against current implementation
  - **No type hints in Parameters section** if already in signature
  - Type hints in docstring ONLY for CUDA device functions
  - **Escapes all backslashes** in docstrings (\\)
  - **Updates .rst API reference files** (for touched files only)
  - **Searches narrative docs** for modified functions, reports usage (doesn't update narrative)
  - **Updates `.github/context/cubie_internal_structure.md`** with architectural insights
  - Outputs function reference changes for narrative_documenter

### 6. narrative_documenter.agent ✓
- **Role**: Technical storyteller for user-facing documentation
- **MCP Tools**: mermaid (optional), markdown-lint (optional)
- **Context**: `.github/context/cubie_internal_structure.md`, `AGENTS.md`
- **Outputs**: Documentation in **reStructuredText (.rst)** for Sphinx
- **Key Features**:
  - **Works in RST format** for Sphinx (not markdown, except readmes/summaries)
  - **User manual pages are concept-based** (see docs/source/user_guide/ for examples)
  - Reviews existing user guide content for style
  - **Accepts function updates from docstring_guru**
  - Updates narrative docs when API changes affect them
  - Creates how-to guides (task-oriented, .rst)
  - Updates user guide sections (concept-based, .rst)
  - README updates (markdown only)
  - Avoids jargon, explains all math symbols
  - Escapes backslashes properly in RST

## Context Directory

`.github/context/` contains shared context for agents:

```
.github/context/
└── cubie_internal_structure.md  # Maintained by agents
                                  # Updated by: docstring_guru
                                  # Used by: plan_new_feature, detailed_implementer, narrative_documenter
```

## Workflow Changes

### Updated Planning Phase
1. plan_new_feature creates **user_stories.md** first
2. Then creates human_overview.md and agent_plan.md
3. agent_plan.md describes behavior/architecture (not function signatures)
4. Consumers: detailed_implementer and reviewer

### Updated Implementation Phase
1. detailed_implementer creates task_list.md with **"Input Validation Required"**
2. do_task performs ONLY specified validation
3. do_task adds educational comments (NOT docstrings)
4. do_task flags bugs/risks but executes as specified
5. do_task runs tests ONLY when explicitly added

### Updated Review Phase
1. reviewer receives user_stories.md, agent_plan.md, human_overview.md
2. reviewer validates against user stories
3. reviewer checks buffer reuse and math vs memory opportunities

### Updated Documentation Phase
1. docstring_guru processes inline comments
2. docstring_guru searches narrative docs, reports usage
3. docstring_guru updates cubie_internal_structure.md
4. narrative_documenter accepts function updates
5. narrative_documenter works in RST (not markdown)

## Active Plans Directory Structure

```
.github/active_plans/
├── feature_name/
│   ├── user_stories.md       # NEW: User stories (from plan_new_feature, step 1)
│   ├── human_overview.md     # High-level plan (from plan_new_feature)
│   ├── agent_plan.md         # Technical spec (from plan_new_feature)
│   ├── task_list.md          # Implementation tasks (from detailed_implementer)
│   └── review_report.md      # Review analysis (from reviewer)
└── another_feature/
    └── ...
```

## Tool Specifications in Agent Files

### plan_new_feature
```json
"tools": {
  "perplexity": {
    "enabled": true,
    "description": "Deep research - use ONLY when explicitly requested"
  },
  "playwright": {
    "enabled": true,
    "description": "Web browsing for documentation and examples"
  },
  "github": {
    "enabled": true,
    "description": "Repository exploration - always use"
  }
}
```

### detailed_implementer
```json
"tools": {
  "github": {
    "enabled": true,
    "description": "Deep code exploration and pattern analysis"
  },
  "tree-sitter": {
    "enabled": false,
    "description": "Code parsing and AST analysis"
  },
  "code-search": {
    "enabled": false,
    "description": "Semantic code search"
  }
}
```

### do_task
```json
"tools": {
  "pytest": {
    "enabled": false,
    "description": "Run tests you've added with CUDASIM"
  },
  "linter": {
    "enabled": false,
    "description": "Check PEP8 compliance"
  }
}
```

### reviewer
```json
"tools": {
  "code-metrics": {
    "enabled": false,
    "description": "Complexity analysis"
  },
  "coverage": {
    "enabled": false,
    "description": "Test coverage analysis"
  }
}
```

### docstring_guru
```json
"tools": {
  "sphinx": {
    "enabled": false,
    "description": "Validate Sphinx builds correctly"
  },
  "doctests": {
    "enabled": false,
    "description": "Run docstring examples"
  }
}
```

### narrative_documenter
```json
"tools": {
  "mermaid": {
    "enabled": false,
    "description": "Create diagrams (export for RST)"
  },
  "markdown-lint": {
    "enabled": false,
    "description": "Check markdown for readmes"
  }
}
```

## Key Behavioral Changes

### plan_new_feature
- ✓ Creates user stories FIRST
- ✓ Uses Perplexity deep_research ONLY if explicitly requested (not automatically)
- ✓ agent_plan.md describes behavior/architecture (no implementation details)
- ✓ Shortcut for minimal bug fixes
- ✓ Consumers are detailed_implementer and reviewer (not do_task)

### detailed_implementer
- ✓ Adds "Input Validation Required" field
- ✓ Specifies exact validation for do_task

### do_task
- ✓ Adds educational comments (NOT docstrings)
- ✓ Performs ONLY specified validation (no extra)
- ✓ Runs tests ONLY when explicitly added, with CUDASIM
- ✓ Flags bugs/risks in outcomes (no deviations from spec)
- ✓ Repository conventions from repo-level instructions (no specific mentions)

### reviewer
- ✓ Validates against user_stories.md
- ✓ Requires agent_plan.md and human_overview.md
- ✓ Checks buffer reuse opportunities
- ✓ Identifies math vs memory trade-offs

### docstring_guru
- ✓ Processes inline comments
- ✓ Keeps helpful comments, summarizes general ones in Notes
- ✓ No type hints in Parameters if in signature
- ✓ Escapes all backslashes (\\)
- ✓ Updates API reference (touched files only)
- ✓ Searches narrative docs, reports usage (doesn't update)
- ✓ Updates cubie_internal_structure.md
- ✓ No attrs-specific instructions (general classes)

### narrative_documenter
- ✓ Works in RST format (not markdown, except readmes)
- ✓ User manual is concept-based
- ✓ Reviews existing user guide for style
- ✓ Accepts function updates from docstring_guru
- ✓ Updates narrative docs when API changes

## Migration from YAML to JSON

Original agents were in YAML format. All agents have been converted to JSON format per GitHub Copilot Agent specification:

**Before** (YAML):
```yaml
name: agent_name
description: Agent description
instructions: |
  Long instructions...
mcp_servers:
  - server1
  - server2
```

**After** (JSON):
```json
{
  "name": "agent_name",
  "description": "Agent description",
  "instructions": "Long instructions...",
  "context": ["file1.md", "file2.md"],
  "tools": {
    "server1": {
      "enabled": true,
      "description": "When to use"
    }
  }
}
```

## Implementation Checklist

- [x] Convert all agent files to JSON format
- [x] Create `.github/context/` directory
- [x] Create `cubie_internal_structure.md` stub
- [x] Add user story creation to plan_new_feature
- [x] Add Input Validation Required to detailed_implementer
- [x] Update do_task for educational comments (not docstrings)
- [x] Update do_task to only run explicitly added tests
- [x] Update do_task to perform only specified validation
- [x] Remove specific convention mentions (use repo-level)
- [x] Update reviewer to validate against user stories
- [x] Update reviewer to check buffer reuse and math vs memory
- [x] Update docstring_guru to process inline comments
- [x] Update docstring_guru to search narrative docs
- [x] Update docstring_guru to update internal structure
- [x] Remove type hints from Parameters example
- [x] Remove attrs-specific examples
- [x] Update narrative_documenter for RST format
- [x] Update narrative_documenter for concept-based user manual
- [x] Add context passing to all agents
- [x] Add tool specifications to all agents
- [x] Update documentation files

## Files Modified

```
.github/
├── agents/
│   ├── plan_new_feature.agent         (JSON format, user stories, shortcut)
│   ├── detailed_implementer.agent     (JSON format, Input Validation Required)
│   ├── do_task.agent                  (JSON format, comments not docstrings)
│   ├── reviewer.agent                 (JSON format, user story validation)
│   ├── docstring_guru.agent           (JSON format, comment processing)
│   ├── narrative_documenter.agent     (JSON format, RST format)
│   ├── README.md                      (Updated with all changes)
│   ├── QUICKSTART.md                  (Updated workflows)
│   ├── IMPLEMENTATION_SUMMARY.md      (This file)
│   └── MCP_SETUP.md                   (Updated if needed)
└── context/
    └── cubie_internal_structure.md    (New stub file)
```

All agents now follow GitHub Copilot Agent JSON specification with integrated context passing and tool configurations.
