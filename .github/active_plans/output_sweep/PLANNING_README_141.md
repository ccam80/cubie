# Issue #141 Planning Documents - README

This directory contains comprehensive planning documentation for **Issue #141: Output Functions Spree**.

## Document Overview

### 📋 Quick Start: [PLANNING_SUMMARY_141.md](PLANNING_SUMMARY_141.md)
**Size:** 6.7 KB | **Read time:** 5-10 minutes

Executive summary with quick reference tables and key decisions.

**Best for:**
- Getting an overview of the entire project
- Understanding scope and dependencies
- Quick reference during implementation

**Contents:**
- Summary of all 10 issues
- Buffer requirements table
- Implementation order by dependency
- Next steps

---

### 📖 Detailed Specs: [IMPLEMENTATION_PLAN_141.md](IMPLEMENTATION_PLAN_141.md)
**Size:** 18 KB | **Read time:** 20-30 minutes

Complete technical specifications for implementing all metrics.

**Best for:**
- Implementation reference
- Understanding exact buffer layouts
- Testing strategy details
- Risk mitigation

**Contents:**
- Detailed specification for each of 8 new metrics
- Buffer layouts with pseudocode
- Update and save logic for each metric
- Architecture changes for #76 and #125
- Testing strategy (unit, integration, system)
- File-by-file checklist
- Risk analysis and mitigation

---

### 🎨 Visual Guide: [ARCHITECTURE_DIAGRAMS_141.md](ARCHITECTURE_DIAGRAMS_141.md)
**Size:** 21 KB | **Read time:** 15-20 minutes

ASCII diagrams showing data flow and memory layouts.

**Best for:**
- Visual learners
- Understanding data flow
- Memory layout visualization
- Compilation process

**Contents:**
- Summary metric data flow diagram
- Buffer layout examples for all metric types
- Combined statistics optimization diagram
- Registry and compilation flow
- Device function chaining pattern
- Memory layout in device arrays
- Implementation phases visualization
- Testing strategy flow

---

## Quick Navigation

### By Implementation Phase

**Phase 1 (Architecture - Must be first):**
- Read: PLANNING_SUMMARY_141.md → "Phase 1" section
- Implement: save_exit_state (#76), iteration_counts (#125)
- Reference: IMPLEMENTATION_PLAN_141.md → "Phase 1"

**Phase 2 (Simple Metrics):**
- Implement: min, max_magnitude, std
- Reference: IMPLEMENTATION_PLAN_141.md → "Phase 2"
- Visual: ARCHITECTURE_DIAGRAMS_141.md → "Combined Statistics Optimization"

**Phase 3 (Peak Detection):**
- Implement: negative_peak, extrema
- Reference: IMPLEMENTATION_PLAN_141.md → "Phase 3"
- Visual: ARCHITECTURE_DIAGRAMS_141.md → "Buffer Layout Examples"

**Phase 4 (Derivatives):**
- Implement: dxdt_extrema, d2xdt2_extrema, dxdt
- Reference: IMPLEMENTATION_PLAN_141.md → "Phase 4"
- Visual: ARCHITECTURE_DIAGRAMS_141.md → "Derivative Metrics"

### By Concern

**"How does the data flow work?"**
→ ARCHITECTURE_DIAGRAMS_141.md → "Summary Metric Data Flow"

**"What's the buffer layout for metric X?"**
→ IMPLEMENTATION_PLAN_141.md → Search for metric name  
→ ARCHITECTURE_DIAGRAMS_141.md → "Buffer Layout Examples"

**"How are metrics registered?"**
→ ARCHITECTURE_DIAGRAMS_141.md → "Registry and Compilation Flow"

**"What's the testing strategy?"**
→ IMPLEMENTATION_PLAN_141.md → "Testing Strategy"  
→ ARCHITECTURE_DIAGRAMS_141.md → "Testing Strategy Flow"

**"What are the risks?"**
→ IMPLEMENTATION_PLAN_141.md → "Risks and Mitigation"

**"How long will this take?"**
→ PLANNING_SUMMARY_141.md → "Recommended Implementation Order"  
→ ARCHITECTURE_DIAGRAMS_141.md → "Implementation Phases Visualization"

---

## Issue Mapping

| Issue | Title | Phase | Complexity | Docs Section |
|-------|-------|-------|-----------|--------------|
| #76 | save_exit_state | 1 | High | Phase 1 |
| #125 | iteration_counts | 1 | High | Phase 1 |
| #63 | min | 2 | Low | Phase 2 |
| #61 | max_magnitude | 2 | Low | Phase 2 |
| #62 | std | 2 | Medium | Phase 2 |
| #64 | negative_peak | 3 | Medium | Phase 3 |
| #65 | extrema | 3 | Medium | Phase 3 |
| #66 | dxdt_extrema | 4 | High | Phase 4 |
| #67 | d2xdt2_extrema | 4 | High | Phase 4 |
| #68 | dxdt | 4 | Medium | Phase 4 |

---

## Key Architectural Constraints

These constraints apply to ALL implementations (documented in all three files):

1. **Device Function Signatures:**
   ```python
   update(value, buffer, current_index, customisable_variable)
   save(buffer, output_array, summarise_every, customisable_variable)
   ```

2. **Device Functions Must:**
   - Return nothing (mutate arrays in place)
   - Support both float32 and float64
   - Be decorated with `@cuda.jit(..., device=True, inline=True)`

3. **Each Metric Needs:**
   - `buffer_size`: int or Callable
   - `output_size`: int or Callable
   - `name`: str
   - `build()` method returning MetricFuncCache

4. **Exceptions:**
   - #76 and #125 are NOT summary metrics
   - They require architecture changes (see Phase 5)

---

## Document Changelog

### 2025-11-03
- **Initial creation** - All three planning documents created
- Comprehensive planning for all 10 issues under #141
- No code changes (planning only as requested)

---

## Notes

- **Planning Status:** ✅ Complete
- **Implementation Status:** ⏳ Not started (awaiting approval)
- **Documents Type:** Planning/Design only
- **Code Changes:** None (as requested in problem statement)

---

## Contact

For questions about these planning documents or the implementation:
- GitHub Issue: #141
- Repository: ccam80/cubie

---

**Last Updated:** 2025-11-03  
**Version:** 1.0 (Planning Phase)  
**Status:** Ready for review and approval
