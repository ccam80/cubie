# CuBIE Custom Agents Implementation Summary

This document summarizes the custom GitHub Copilot agents created for the CuBIE repository and provides recommendations for additional MCP tools.

## Created Agents

All agents are located in `.github/agents/` with the `.agent` extension.

### 1. plan_new_feature.agent ✓
- **Role**: Expert project manager and technical architect
- **MCP Tools**: perplexity, playwright, github
- **Outputs**: `human_overview.md` and `agent_plan.md`
- **Key Features**:
  - One Perplexity research question per feature (quota-limited)
  - Creates architecture diagrams using Mermaid syntax
  - Asks for user feedback on ambiguity
  - Works out of `.github/active_plans/<feature_name>/` directory

### 2. detailed_implementer.agent ✓
- **Role**: Operations manager and implementation planner
- **MCP Tools**: github
- **Outputs**: `task_list.md` with dependency-ordered tasks
- **Key Features**:
  - Provides complete function signatures
  - Organizes tasks by dependency (architecture → core → integration → tests)
  - Marks groups as SEQUENTIAL or PARALLEL
  - Includes explicit context (files and line numbers) for each task group
  - Asks for clarification when faced with design choices

### 3. do_task.agent ✓
- **Role**: Senior developer and implementer
- **MCP Tools**: None (all context in task_list.md)
- **Outputs**: Git patches and updated task_list.md with outcomes
- **Key Features**:
  - Executes tasks exactly as specified (no creative additions)
  - Understands user-facing vs internal code (sanitization vs performance)
  - Never asks user for feedback (executes the plan)
  - Follows all CuBIE conventions strictly

### 4. reviewer.agent ✓
- **Role**: Harsh critic and senior code reviewer
- **MCP Tools**: None
- **Outputs**: `review_report.md` with analysis and suggested edits
- **Key Features**:
  - Identifies code duplication relentlessly
  - Finds unnecessary additions that don't serve goals
  - Suggests simplifications
  - Provides actionable, specific feedback with file/line references
  - Can hand edits back to do_task agents

### 5. docstring_guru.agent ✓
- **Role**: Technical writing specialist for API documentation
- **MCP Tools**: None
- **Outputs**: Updated docstrings in source files, .rst files, and narrative docs
- **Key Features**:
  - Enforces numpydoc style for Sphinx
  - Type hints in function signatures (NOT in docstrings)
  - Exception: numba.cuda device functions (type hints ONLY in docstring)
  - Checks and updates module docstrings
  - Updates .rst files in api_reference
  - Updates narrative docs that mention modified functions

### 6. narrative_documenter.agent ✓
- **Role**: Technical storyteller for user documentation
- **MCP Tools**: None
- **Outputs**: How-to guides, user manual sections, readme updates
- **Key Features**:
  - Avoids jargon unless clearly explained
  - Explains all mathematical symbols immediately after equations
  - Uses grounded, physical examples for all math
  - Never glib, not overly enthusiastic, almost never uses adverbs
  - Technical writer at heart (clarity over cleverness)
  - Asks for feedback when faced with ambiguity

## Handoff Configuration

All agents have properly configured handoffs:

```
plan_new_feature → detailed_implementer → do_task → reviewer → do_task (if needed)
                                                   ↓
                                          docstring_guru ↔ narrative_documenter
```

Each handoff specifies:
- Target agent name
- Description of what's being handed off
- Required files
- Parameters (where applicable)

## MCP Configuration

### Created: `.github/mcp.json`
Configures three MCP servers:
1. **perplexity**: Research API (requires `PERPLEXITY_API_KEY`)
2. **playwright**: Web automation
3. **github**: Repository operations (uses `GITHUB_TOKEN`)

### Additional Documentation
- `.github/agents/README.md`: Complete usage guide and workflow documentation
- `.github/agents/MCP_SETUP.md`: MCP server setup and troubleshooting

## Recommended Additional MCP Tools

Based on the analysis of each agent's needs, here are recommended MCP servers to enhance capabilities:

### High Priority Recommendations

#### For detailed_implementer
1. **@modelcontextprotocol/server-tree-sitter**
   - Purpose: Advanced code parsing and AST analysis
   - Benefit: Better dependency analysis and finding all call sites
   - Why: Would significantly improve task ordering and context identification

2. **@modelcontextprotocol/server-code-search**
   - Purpose: Semantic code search
   - Benefit: Find similar implementations and patterns across the repository
   - Why: Helps identify all files needing modification for architectural changes

#### For do_task
1. **@modelcontextprotocol/server-pytest**
   - Purpose: Run pytest and analyze results
   - Benefit: Immediate test feedback during implementation
   - Why: Could validate changes without human intervention

2. **@modelcontextprotocol/server-linter**
   - Purpose: Real-time linting (flake8, ruff)
   - Benefit: Catch PEP8 violations during coding
   - Why: Ensures compliance before committing

#### For reviewer
1. **@modelcontextprotocol/server-code-metrics**
   - Purpose: Complexity and quality metrics (cyclomatic complexity, etc.)
   - Benefit: Quantitative analysis to support simplification suggestions
   - Why: Adds objective data to the harsh critic's arsenal

2. **@modelcontextprotocol/server-coverage**
   - Purpose: Test coverage analysis
   - Benefit: Identify untested code paths
   - Why: Could highlight areas needing test additions

#### For docstring_guru
1. **@modelcontextprotocol/server-sphinx**
   - Purpose: Sphinx documentation builder
   - Benefit: Validate that docstring changes build correctly
   - Why: Catch Sphinx errors before they break documentation builds

2. **@modelcontextprotocol/server-doctests**
   - Purpose: Run docstring examples
   - Benefit: Ensure examples are correct and executable
   - Why: Prevents broken examples in documentation

#### For narrative_documenter
1. **@modelcontextprotocol/server-mermaid**
   - Purpose: Mermaid diagram generation
   - Benefit: Create and validate diagrams in documentation
   - Why: Enhances visual explanations in how-to guides

2. **@modelcontextprotocol/server-markdown-lint**
   - Purpose: Markdown linting
   - Benefit: Consistent documentation formatting
   - Why: Maintains professional documentation quality

### Medium Priority Recommendations

#### General Purpose (Multiple Agents)
1. **@modelcontextprotocol/server-filesystem**
   - Purpose: Enhanced file system operations
   - Benefit: Better file search across large codebases
   - Used by: detailed_implementer, docstring_guru, narrative_documenter

2. **@modelcontextprotocol/server-git**
   - Purpose: Advanced Git operations
   - Benefit: Better repository history analysis
   - Used by: plan_new_feature, reviewer

### Lower Priority (Nice to Have)

1. **@modelcontextprotocol/server-jupyter**
   - For creating example notebooks in documentation
   
2. **@modelcontextprotocol/server-sqlite**
   - For querying structured data about code metrics over time

## Agent Expertise and Context

All agents are configured with expertise in:
- ✓ Python 3.8+ development
- ✓ CUDA programming and GPU architecture
- ✓ Numba JIT compilation and CUDA kernels
- ✓ CuBIE-specific conventions (from AGENTS.md)

## Repository Structure Integration

All agents understand and work with:
- ✓ `.github/active_plans/` directory for feature planning
- ✓ One-to-three word snake_case directory names
- ✓ Structured output files (human_overview.md, agent_plan.md, task_list.md, review_report.md)
- ✓ CuBIE's architecture (batchsolving, integrators, memory, odesystems, outputhandling)

## Key Behavioral Features

### Asking for Feedback
- ✓ plan_new_feature: Asks when faced with ambiguity or design choices
- ✓ detailed_implementer: Asks for clarification on unclear specifications
- ✓ do_task: Never asks (executes plan exactly)
- ✓ reviewer: Asks for clarification on standards when unsure
- ✓ docstring_guru: Asks when function purpose is unclear
- ✓ narrative_documenter: Asks about target audience and scope

### User-Facing vs Internal Code
- ✓ do_task explicitly understands the difference
- ✓ User-facing: Validates/sanitizes inputs, helpful errors
- ✓ Internal: Trusts library inputs, optimizes for performance

## Implementation Checklist

- [x] Created `.github/agents/` directory
- [x] Created `plan_new_feature.agent` with Perplexity, Playwright, GitHub MCP
- [x] Created `detailed_implementer.agent` with GitHub MCP
- [x] Created `do_task.agent` (no MCP needed)
- [x] Created `reviewer.agent` (no MCP needed)
- [x] Created `docstring_guru.agent` (no MCP needed)
- [x] Created `narrative_documenter.agent` (no MCP needed)
- [x] Created `.github/mcp.json` with MCP server configurations
- [x] Configured handoffs between all agents
- [x] Set up `.github/active_plans/` directory workflow
- [x] Documented all agents in README.md
- [x] Provided MCP setup documentation
- [x] Validated YAML syntax in all .agent files
- [x] Validated JSON syntax in mcp.json
- [x] Included recommended additional MCP tools

## Testing Recommendations

To test the agent setup:

1. **Test plan_new_feature**:
   ```
   @plan_new_feature Create a simple feature plan for adding a new summary metric
   ```

2. **Test detailed_implementer**:
   ```
   @detailed_implementer Review the plan in .github/active_plans/test_feature/
   ```

3. **Test do_task**:
   ```
   @do_task Execute task group 1 from the task list
   ```

4. **Test reviewer**:
   ```
   @reviewer Analyze the completed test implementation
   ```

5. **Test docstring_guru**:
   ```
   @docstring_guru Review docstrings in src/cubie/outputhandling/summarymetrics/mean.py
   ```

6. **Test narrative_documenter**:
   ```
   @narrative_documenter Create a how-to guide for the test feature
   ```

## Notes

- All agents follow CuBIE conventions from AGENTS.md and .github/copilot-instructions.md
- Breaking changes are acceptable (no backwards compatibility)
- PowerShell command format (no `&&`)
- 79 character line limit, 71 character comment limit
- Numpydoc format strictly enforced
- Type hints in signatures (except CUDA device functions: docstring only)
- Attrs float pattern: underscore + property
- Never call build() directly on CUDAFactory subclasses

## Future Enhancements

Potential improvements:
- Add more MCP servers as they become available
- Create specialized agents for specific CuBIE subsystems
- Develop testing-focused agent for test generation
- Create refactoring agent for code cleanup
- Add performance profiling agent for CUDA optimization
