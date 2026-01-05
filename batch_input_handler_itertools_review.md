# BatchInputHandler itertools delegation review

Summary of places where existing conditional or combining logic in
`src/cubie/batchsolving/BatchInputHandler.py` could be replaced by
`itertools` helpers instead of custom branching, with rough line savings.

## Opportunities

1. **Deduped Cartesian expansion**  
   For dict inputs, `_process_single_input` routes into `generate_grid`, which
   calls `combinatorial_grid`, which in turn calls
   `unique_cartesian_product` (lines 137-223) before returning to
   `_process_input`. That path already relies on
   `itertools.product` after deduplicating each vector with
   `dict.fromkeys`. The bespoke helper can be removed by performing that
   deduplication in `combinatorial_grid` and passing the result directly
   to `product`:

   ```python
   deduped = [list(dict.fromkeys(v)) for v in cleaned_request.values()]
   ```

   The behaviour stays the same, but collapsing the helper and call site
   trims roughly 30 lines.

1. **Run pairing for combinatorial alignment**  
   In `combine_grids` (lines 320-355), the combinatorial branch uses
   `np_repeat`/`np_tile` to form the Cartesian product of run columns.
   An itertools alternative would build the same combinations by
   iterating over column indices from `product(range(grid1.shape[1]),
   range(grid2.shape[1]))`, then stacking the selected views. That avoids
   explicit repeat/tile branches but offers little code reduction (at
   most a couple of lines) and would trade vectorised NumPy for Python
   loops.

1. **Mirrored fast-path branching**  
   `_try_fast_path_arrays` (lines 1033-1093) has two nearly identical
   branches for the states/params symmetry. Iterating once over a tuple
   such as:

   ```python
   (
       (states, params, self.states, self.parameters),
       (params, states, self.parameters, self.states),
   )
   ```

   and unpacking in a single block collapses the duplicate conditionals.
   That consolidation would drop about 14–16 lines. If desired,
   `_is_1d_or_none` (lines 980-1010) could also be expressed with
   `chain.from_iterable` to mirror the itertools style, though the
   existing generator expression is already clear.

## Total estimated impact

Collectively, delegating the above to `itertools` primitives could
remove roughly 44–48 lines while preserving current behaviour and
ordering guarantees, with most of the savings coming from folding
`unique_cartesian_product` and the mirrored fast path.
