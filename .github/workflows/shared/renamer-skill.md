---
description: Renamer agent execution skill for managing method, function, property, and attribute names across the codebase
---

# Renamer Agent Execution Skill

## Overview

This skill defines how to execute the renamer agent to rationalize method, function, property, and attribute names in the codebase.

**Important**: The renamer agent manages a tracking file called `name_info.md` in the repository root. You coordinate the agent by invoking it with the appropriate operation and parameters.

## Command Recognition

Recognize these command variations:
- "run renamer"
- "run renamer on [file/directory]"
- "renamer update for [file/directory]"
- "renamer recommend for [file/directory]"
- "renamer rename [file/directory]"
- "renamer recommend [N] items"
- "renamer rename [N] items"

## Operations

The renamer supports three operations:

1. **update_list** - Scan files and add all names to name_info.md
2. **recommend** - Analyze items and suggest better names
3. **rename** - Execute recommended renames in source files

## Execution Workflows

### Full Workflow (no operation specified)

When user says "run renamer on [target]":

1. Invoke renamer with operation="update_list" and target file/directory
2. Wait for completion
3. Invoke renamer with operation="recommend" and chunk_size=10 (or user-specified)
4. Wait for completion
5. Invoke renamer with operation="rename" and chunk_size=5 (or user-specified)
6. Wait for completion
7. Present summary to user

### Single Operation

When user specifies an operation (e.g., "renamer recommend for src/cubie/integrators"):

1. Invoke renamer with the specified operation
2. Use reasonable chunk size (10 for recommend, 5 for rename) unless user specifies
3. Wait for completion
4. Present results to user

## Default Chunk Sizes

- **recommend**: 10 items per invocation
- **rename**: 5 items per invocation
- **update_list**: process all items

## Parameters to Pass

When invoking the renamer agent, provide:

- Target file or directory path (absolute path)
- Operation: "update_list", "recommend", or "rename"
- Chunk size (optional, for recommend/rename operations)

## Example Executions

### Example 1: Full workflow on a directory

User says: "run renamer on src/cubie/integrators"

Your action:
```
1. Invoke renamer agent with:
   - Prompt: "Operation: update_list, Target: /home/runner/work/cubie/cubie/src/cubie/integrators"
2. Wait for renamer to complete
3. Invoke renamer agent with:
   - Prompt: "Operation: recommend, Target: /home/runner/work/cubie/cubie/src/cubie/integrators, Chunk size: 10"
4. Wait for renamer to complete
5. Invoke renamer agent with:
   - Prompt: "Operation: rename, Target: /home/runner/work/cubie/cubie/src/cubie/integrators, Chunk size: 5"
6. Wait for renamer to complete
7. Present summary showing:
   - Items added to tracking
   - Recommendations made
   - Renames executed
   - Status of name_info.md
```

### Example 2: Single operation with custom chunk size

User says: "renamer recommend 15 items"

Your action:
```
1. Invoke renamer agent with:
   - Prompt: "Operation: recommend, Chunk size: 15"
2. Wait for renamer to complete
3. Present recommendations to user
```

## Important Notes

- The renamer agent is independent from the main pipeline
- It manages its own tracking file (name_info.md)
- Always use absolute paths when specifying targets
- Chunk sizes help manage large refactoring operations
- The renamer verifies completeness of renames across the repository
