# BatchInputHandler itertools delegation review

Summary of places where existing conditional or combining logic in
`src/cubie/batchsolving/BatchInputHandler.py` could be replaced by
`itertools` helpers instead of custom branching, with rough line savings.

## Opportunities

1. **Deduped Cartesian expansion**  
   The chain `_process_input → generate_grid → combinatorial_grid →
   unique_cartesian_product` (lines 137-223) first deduplicates values,
   then calls a bespoke helper that just wraps `itertools.product`. That
   helper can be inlined with:
   `deduped = map(dict.fromkeys, cleaned_request.values())` followed by
   `np_array(list(product(*deduped))).T`. Removing
   `unique_cartesian_product` entirely and folding the two-line call in
   `combinatorial_grid` cuts roughly 30 lines while keeping ordering.

2. **Run pairing for combinatorial alignment**  
   In `combine_grids` (lines 320-355), the combinatorial branch manually
   repeats and tiles columns. Using column views with
   `zip(*product(grid1.T, grid2.T))` and a pair of `np_column_stack`
   calls produces the same output in about three lines, trimming roughly
   3–4 lines and removes the need for separate repeat/tile reasoning.

3. **Mirrored fast-path branching**  
   `_try_fast_path_arrays` (lines 1033-1093) has two nearly identical
   branches for the states/params symmetry. Iterating over
   `product((states, params), ((self.states, self.parameters),))` (or a
   simple tuple-driven loop) and unpacking with a single block collapses
   the duplication. That consolidation would drop about 14–16 lines.
   `_is_1d_or_none` (lines 980-1010) can also lean on
   `chain.from_iterable` to replace the bespoke nested `any`, trimming a
   few more lines.

## Total estimated impact

Collectively, delegating the above to `itertools` primitives could
remove roughly 45–50 lines while preserving current behaviour and
ordering guarantees.
