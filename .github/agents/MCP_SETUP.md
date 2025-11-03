# MCP Server Setup for CuBIE Agents

This document provides additional information about the Model Context Protocol (MCP) servers configured for CuBIE's GitHub Copilot agents.

## Overview

MCP servers provide agents with access to external tools and services. The configuration is in `.github/mcp.json`.

## Configured Servers

### 1. Perplexity (perplexity)
**Used by**: plan_new_feature  
**Purpose**: External knowledge research  
**Quota**: ONE question per feature request (enforced in agent instructions)

**Setup**:
```bash
# Set environment variable
export PERPLEXITY_API_KEY="your-api-key-here"
```

**Capabilities**:
- Research academic papers and implementations
- Find state-of-the-art algorithms
- Discover best practices
- Compare technical approaches

### 2. Playwright (playwright)
**Used by**: plan_new_feature  
**Purpose**: Web browsing and automation

**Setup**:
No API key required. Automatically installs via npx.

**Capabilities**:
- Browse documentation websites
- Examine code examples online
- Search GitHub repositories
- Navigate technical resources

### 3. GitHub (github)
**Used by**: plan_new_feature, detailed_implementer  
**Purpose**: Repository operations

**Setup**:
```bash
# Uses GITHUB_TOKEN from environment
# Usually automatically available in GitHub Copilot
export GITHUB_TOKEN="your-token-here"
```

**Capabilities**:
- Search repository issues and PRs
- Read file contents
- Review commit history
- Analyze code structure

## Additional Recommended MCP Servers

Based on the agent requirements, here are additional MCP servers that could be beneficial:

### For detailed_implementer

#### Code Analysis Tools
1. **@modelcontextprotocol/server-tree-sitter**
   - Purpose: Advanced code parsing and AST analysis
   - Benefit: Better understanding of code structure and dependencies
   - Usage: Finding all call sites of a function, analyzing inheritance hierarchies

2. **@modelcontextprotocol/server-code-search**
   - Purpose: Semantic code search across repository
   - Benefit: Find similar implementations and patterns
   - Usage: Identifying all files that need modification for a change

Configuration example:
```json
"tree-sitter": {
  "description": "Tree-sitter for code analysis",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-tree-sitter"],
  "disabled": false
}
```

### For do_task

#### Development Tools
1. **@modelcontextprotocol/server-linter**
   - Purpose: Real-time linting and style checking
   - Benefit: Catch style violations during implementation
   - Usage: Ensure PEP8 compliance before committing

2. **@modelcontextprotocol/server-pytest**
   - Purpose: Run and analyze pytest results
   - Benefit: Immediate test feedback during implementation
   - Usage: Verify changes don't break existing tests

Configuration example:
```json
"pytest": {
  "description": "Pytest runner and analyzer",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-pytest"],
  "disabled": false
}
```

### For reviewer

#### Analysis Tools
1. **@modelcontextprotocol/server-code-metrics**
   - Purpose: Code complexity and quality metrics
   - Benefit: Quantitative analysis of code quality
   - Usage: Identify overly complex functions needing simplification

2. **@modelcontextprotocol/server-coverage**
   - Purpose: Test coverage analysis
   - Benefit: Find untested code paths
   - Usage: Ensure new code has adequate test coverage

Configuration example:
```json
"code-metrics": {
  "description": "Code metrics and complexity analysis",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-code-metrics"],
  "disabled": false
}
```

### For docstring_guru

#### Documentation Tools
1. **@modelcontextprotocol/server-sphinx**
   - Purpose: Sphinx documentation builder and validator
   - Benefit: Validate documentation builds correctly
   - Usage: Check that docstring changes don't break Sphinx build

2. **@modelcontextprotocol/server-doctests**
   - Purpose: Run and validate docstring examples
   - Benefit: Ensure examples in docstrings are correct
   - Usage: Test all code examples in docstrings

Configuration example:
```json
"sphinx": {
  "description": "Sphinx documentation validator",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-sphinx"],
  "disabled": false
}
```

### For narrative_documenter

#### Content Tools
1. **@modelcontextprotocol/server-mermaid**
   - Purpose: Mermaid diagram generation and validation
   - Benefit: Create and validate diagrams in documentation
   - Usage: Generate architecture and flow diagrams

2. **@modelcontextprotocol/server-markdown-lint**
   - Purpose: Markdown linting and formatting
   - Benefit: Consistent documentation formatting
   - Usage: Ensure documentation follows style guidelines

Configuration example:
```json
"mermaid": {
  "description": "Mermaid diagram generator",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-mermaid"],
  "disabled": false
}
```

## General Purpose Tools (All Agents)

### File System Operations
1. **@modelcontextprotocol/server-filesystem**
   - Purpose: Enhanced file system operations
   - Benefit: Better file search and manipulation
   - Usage: Finding files, reading multiple files efficiently

### Version Control
1. **@modelcontextprotocol/server-git**
   - Purpose: Advanced Git operations
   - Benefit: Better understanding of repository history
   - Usage: Analyzing changes, finding when bugs were introduced

## Security Considerations

### API Keys
- Store API keys in environment variables, never in code
- Use GitHub Secrets for CI/CD environments
- Rotate keys regularly

### Access Control
- Limit MCP server permissions to minimum required
- Review MCP server logs for unexpected behavior
- Disable unused servers

## Troubleshooting

### Server Won't Start
```bash
# Clear npx cache
npx clear-npx-cache

# Reinstall server
npx -y @modelcontextprotocol/server-<name>
```

### Connection Issues
```bash
# Check environment variables
echo $PERPLEXITY_API_KEY
echo $GITHUB_TOKEN

# Test server directly
npx -y @modelcontextprotocol/server-<name> --help
```

### Performance Issues
- Disable servers not actively needed
- Monitor server resource usage
- Consider running servers separately from agents

## Extending MCP Configuration

To add a new MCP server:

1. **Update `.github/mcp.json`**:
   ```json
   "new-server": {
     "description": "Server description",
     "command": "npx",
     "args": ["-y", "@scope/server-name"],
     "env": {
       "API_KEY": "${ENV_VAR}"
     },
     "disabled": false
   }
   ```

2. **Update agent configuration** in `.agent` file:
   ```yaml
   mcp_servers:
     - existing-server
     - new-server
   ```

3. **Document in README**: Update `.github/agents/README.md`

4. **Test**: Verify server works with agent

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [Available MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Creating Custom MCP Servers](https://modelcontextprotocol.io/docs/creating-servers)

## Support

For issues with MCP servers:
1. Check server documentation
2. Review logs in GitHub Copilot
3. Test server independently
4. Report issues to server maintainers
