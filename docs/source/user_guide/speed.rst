Making it Faster
================

TL/DR
-----
To get the best performance from Cubie, try to:

- Solve many problems at once (thousands if possible).
- Reduce the number of variables and samples you save or summarise.
- Set all parameters that you're not changing between solves to be `constants`.
- Reuse existing Solvers.

Parallelism
-----------

If we compare a like-for-like implementation of an IVP integration using Cubie
vs using an optimised CPU-utilising library like SciPy, we find that Cubie is
some linear factor slower. This is expected - GPU hardware isn't optimised
to process single tasks quickly. Instead, it's optimised to process many
tasks in parallel. Except for a penalty transferring data in and out of the
integration functions, increasing the number of problems being solved at
once by a factor \(n\) has little effect on the total time taken - you get \
(n-1\) problems solved _almost_ for free. The single best way to get a
performance gain from using Cubie is to solve more problems at once.

Memory
------
The big bottleneck in GPU computing is memory traffic. When you're completing
32,000 integrations at the same time, they all want to save a sample of
their state at the same time, and this puts a lot of pressure on the tubes
between the GPU and its memory. The more data you save, the slower it goes.
Cubie has three main levers you can pull to reduce memory traffic and speed
up your solves:

1. Reduce the number of variables you save. If you're solving a 10D system
   but only care about one variable, only save that one variable.
2. Reduce the number of samples you save. If you're solving a system for
   1000 time units but only care about the state at the end, only save the
   final state.
3. Use summary metrics. If you want to know the mean and standard
   deviation of a variable over the course of the solve, rather than save
   the whole history and process each dataset offline (slow!), use Cubie's
   built in summary metrics to calculate these on the GPU during the solve. You
   don't even need to save the state history at all!

Constants
---------
When you tell Cubie about your problem, you provide some symbols/variables
that are input-only - they don't change during the solve. If you're
brute-forcing a parameter study, you will want to be able to start an IVP from
a bunch of different values for some of these parameters. However, you may
have more parameters that you're not interested in changing between solves.
If you mark these as `constants` when defining your system of ODEs, Cubie
puts them in a different place in memory - rather than taking up space in
the scarce fast memory that needs to be able to change often, they go into
the compiled program itself. This means they require no memory traffic, and
they free up more space to run more runs at once!

Reusing Solvers
---------------
Cubie uses Numba to just-in-time compile your system of ODEs and chosen
integration algorithm into a CUDA kernel. This compilation step takes time
(up to 30s!), so if you're only solving a few problems at once, it can
dominate the total time taken. The compiled kernel is cached, and as long as
you're running the same system with the same algorithm and saving at the
same cadence (more or less, there's some other variables you might change
that could force a recompile), Cubie will reuse the existing kernel and skip
compilation for the next run. For this reason, if you're going to do
multiple batches with the same system, instead of using the :func:`cubie.solve_ivp`
function, create a :class:`cubie.Solver` object and call its :meth:`solve`
method multiple times. Keeping a reference to the :class:`cubie.Solver`
object means that subsequent calls to :meth:`solve` will be much faster.

DIRK and Newton-Krylov Solver Micro-optimizations
-------------------------------------------------
The DIRK linear solver and Newton-Krylov helper in the debugging bundle
``tests/all_in_one.py`` (device helpers collated for lineinfo debugging but
mirroring production kernels) already mirror production behavior. When
profiling or experimenting with them, the following refactors keep the
public contract but reduce GPU stalls and memory pressure.

Loop shaping and indexing
~~~~~~~~~~~~~~~~~~~~~~~~~
- Prefer fixed-trip loops over range-based indexing when stage counts,
  driver counts, or state size are compile-time constants captured by the
  factories. Guard tail work with masks rather than dynamic bounds checks.
- Replace inner ``for i in range(n)`` reductions with small fixed chunks
  (e.g. two or four iterations) fed by statically known indices to help the
  compiler unroll without explicit unroll helpers such as ``@numba.unroll``.
- Keep stride order and buffer slices stable: build local views once
  per stage and reuse them across iterations to avoid repeated slice math.
- Precompute invariants such as ``tol_squared``, ``1/denominator`` guards,
  and scaled tableau coefficients (DIRK stage weights/diagonals) outside
  hot loops.

all_sync and status flow
~~~~~~~~~~~~~~~~~~~~~~~~
- Hoist ``mask = activemask()`` (CUDA warp mask) once per solver—compute
  it once before the loop and carry it through nested loops rather than
  recomputing per iteration.
- Collapse status flags into a single integer and propagate with
  predicated writes instead of branching: update ``status`` only when
  ``status < 0`` to keep lanes aligned.
- Defer ``all_sync`` to points where all threads have computed a fresh
  predicate (e.g. post reduction) and avoid back-to-back synchronizations.
- For backtracking, track ``found_step`` with a lane-local flag and reduce
  via ``all_sync`` after residual evaluation, skipping extra synchronization
  when a warp has already converged.

CUDA-friendly tweaks
~~~~~~~~~~~~~~~~~~~~
- Use predicated updates for ``rhs`` and ``delta`` writes so divergent
  branches do not serialize the warp.
- Reuse accumulators: recycle ``temp`` for both operator output and dot
  products, and reuse ``delta`` buffers across backtracks instead of
  reallocating scratch slices.
- Stage masks once and reuse them to gate writes into shared or local
  buffers, keeping inactive lanes from touching memory.
- Pre-stage tableau rows, diagonal coefficients, and damping factors in
  registers before entering the main iteration to cut global reads.
- When using shared memory for ``solver_scratch`` or stage accumulators,
  keep access patterns contiguous and avoid mixing read/write phases
  without a clear barrier to limit bank conflicts.
- Favor predicated commits over ``if/else`` when copying increments back
  to ``stage_increment`` or ``residual`` so threads that already converged
  stay aligned.
