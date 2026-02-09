 CuBIE Test Sweep — Progress Log

## Phase 0: Test Structure Guide
- **Status:** COMPLETE (prior session)

## Phase 1: Functionality Inventory

### Task 1.1 — Calibration Walkthrough
- **Date:** 2026-01-30
- **File:** `SingleIntegratorRunCore.py` + `SingleIntegratorRun.py`
- **Status:** COMPLETE
- **Outcome:**
  - 88 functionality items for SingleIntegratorRunCore
  - 48 functionality items for SingleIntegratorRun (39 forwarders + 2 dedicated + 7 calculation cases)
  - Interactive walkthrough with user — every block reviewed
  - 3 vestiges removed (`algorithm_key`, `compiled_loop_function`, `threads_per_loop`)
  - 2 bugs fixed (`set("algorithm")` → `{"algorithm"}` in both `_switch_algos` and `_switch_controllers`)
  - Key rule established: **Phase 1 inventories from source only, no reference to existing tests**
  - Todo item 18 added: investigate build() vs update() device function fetching

### Task 1.2 — Calibration Review
- **Status:** NOT STARTED
- **Next:** Review context cost, determine batch sizes, divide remaining ~118 files into sessions

## Phase 2: Cross-Cutting Commonality Scan
- **Status:** NOT STARTED

## Phase 3: Per-File Test Audit & Rewrite
- **Status:** NOT STARTED

## Deferred Items
- GUI files (`gui/`) excluded from test sweep (Qt dependency)
