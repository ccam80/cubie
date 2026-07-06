Caching and Recompilation
=========================

CuBIE uses two caching layers to avoid redundant work: a *code-generation
cache* (Python source files) and Numba's *kernel disk cache* (compiled
machine code).

The ``generated/`` Directory
----------------------------

When CuBIE first compiles a system, it writes generated Python files into
a ``generated/`` directory inside the working directory (or a
user-specified cache directory).  These files contain the CUDA device
functions for the right-hand side, Jacobian helpers, and other
system-specific code.

When Recompilation Happens
--------------------------

CuBIE recompiles when any of the following change:

- The ODE equations.
- Constant values (since they are baked into the compiled code).
- Floating-point precision (``float32`` vs ``float64``).
- Algorithm choice or algorithm settings.
- Output configuration (saved variables, summary metrics).

Changing *parameters* or *initial values* does **not** trigger
recompilation---those are runtime inputs.

Numba Kernel Cache
------------------

After CuBIE generates Python source, Numba JIT-compiles it into GPU
machine code.  Numba caches compiled kernels to disk so that the second
run with the same configuration skips the JIT step entirely.  This cache
is separate from the ``generated/`` directory and is managed by Numba.

Cache Settings
--------------

On the :class:`~cubie.Solver`:

``cache=True`` (default)
   Enable both code-generation and Numba caching.

``cache=False``
   Disable caching; recompile every time.

``cache="path/to/dir"``
   Use a custom directory for generated files.

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand_prince_54",
                       cache="/tmp/cubie_cache")

Clearing Caches
---------------

Delete the ``generated/`` directory to clear the code-generation cache.
Numba's cache can be cleared by deleting the ``__pycache__`` directories
next to the generated files, or by setting ``cache=False`` for a single
run.
