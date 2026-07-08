Configuration Reference
========================

This page is the single index of every entry-point keyword argument for a
solve: the explicit parameters of :func:`~cubie.solve_ivp` and
:class:`~cubie.Solver` (:meth:`~cubie.Solver.__init__` and
:meth:`~cubie.Solver.solve`), plus how the remaining loose keyword arguments
are routed to the six underlying settings groups. Deep-dive pages for each
group are linked at the end of each section.

Entry-point signatures
-----------------------

``solve_ivp()``
~~~~~~~~~~~~~~~

:func:`~cubie.solve_ivp` (``src/cubie/batchsolving/solver.py:116-131``)
builds a :class:`~cubie.Solver` and runs a single batch solve.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``method``
     - ``"euler"``
     - Integration algorithm name (see :doc:`choosing_algorithms`).
   * - ``duration``
     - ``1.0``
     - Total integration time.
   * - ``settling_time``
     - ``0.0``
     - Warm-up period before outputs are recorded.
   * - ``t0``
     - ``0.0``
     - Initial integration time.
   * - ``save_variables``
     - ``None``
     - State/observable names to save; ``None`` saves all, ``[]`` saves
       none.
   * - ``summarise_variables``
     - ``None``
     - State/observable names to summarise; ``None`` mirrors
       ``save_variables``.
   * - ``grid_type``
     - ``"combinatorial"``
     - ``"combinatorial"`` takes every combination of the supplied
       inputs; ``"verbatim"`` pairs them run-for-run. **This default
       differs from** ``Solver.solve`` **below.**
   * - ``time_logging_level``
     - ``None``
     - Timing verbosity: ``'default'``, ``'verbose'``, ``'debug'``, or
       ``None``/``'None'`` to disable.
   * - ``nan_error_trajectories``
     - ``True``
     - Trajectories with a nonzero status code are NaN-masked in the
       output.

Any other keyword argument is forwarded to :class:`~cubie.Solver`.

``Solver.__init__()``
~~~~~~~~~~~~~~~~~~~~~

(``src/cubie/batchsolving/solver.py:278-291``)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``algorithm``
     - ``"euler"``
     - Integration algorithm name or a supplied
       :class:`~cubie.integrators.algorithms.ButcherTableau`.
   * - ``profileCUDA``
     - ``False``
     - Enables CUDA profiling of the compiled kernel.
   * - ``cache``
     - ``True``
     - Compiled-kernel disk caching; accepts ``bool``, a cache-mode
       string, or a ``Path``. See :doc:`caching`.
   * - ``time_logging_level``
     - ``None``
     - Same options as above.

``step_control_settings``, ``algorithm_settings``, ``output_settings``,
``memory_settings``, and ``loop_settings`` are explicit dictionaries that
override the per-group defaults; any of their keys may also be supplied as
loose keyword arguments (see "Kwarg routing" below).

``Solver.solve()``
~~~~~~~~~~~~~~~~~~

(``src/cubie/batchsolving/solver.py:422-436``)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``duration``
     - ``1.0``
     - Total integration time.
   * - ``settling_time``
     - ``0.0``
     - Warm-up period before outputs are recorded.
   * - ``t0``
     - ``0.0``
     - Initial integration time.
   * - ``blocksize``
     - ``256``
     - CUDA threads per block for the kernel launch.
   * - ``stream``
     - ``None``
     - CUDA stream to launch on; ``None`` uses the solver's default
       stream.
   * - ``grid_type``
     - ``"verbatim"``
     - **Differs from** ``solve_ivp``'s default of
       ``"combinatorial"`` above — only relevant when dict inputs
       trigger grid construction.
   * - ``results_type``
     - ``"full"``
     - Shape of the returned result (e.g. ``"full"`` for a
       :class:`~cubie.batchsolving.solveresult.SolveResult`, ``"raw"``
       for a plain dict). See :doc:`results`.
   * - ``nan_error_trajectories``
     - ``True``
     - Same behaviour as above; ignored when ``results_type="raw"``.

Any other keyword argument is forwarded to :meth:`~cubie.Solver.update`,
routing through the same six settings groups described next.

Kwarg routing
-------------

Beyond the explicit parameters above, ``Solver`` accepts loose keyword
arguments that are merged into six settings groups by
``merge_kwargs_into_settings`` (``src/cubie/batchsolving/solver.py:325-399``):

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Group
     - Recognised-key set
     - Deep-dive page
   * - Step control
     - ``ALL_STEP_CONTROLLER_PARAMETERS``
     - :doc:`optional_arguments`
   * - Algorithm
     - ``ALL_ALGORITHM_STEP_PARAMETERS``
     - :doc:`optional_arguments`, :doc:`choosing_algorithms`
   * - Output
     - ``ALL_OUTPUT_FUNCTION_PARAMETERS``
     - :doc:`results`
   * - Memory
     - ``ALL_MEMORY_MANAGER_PARAMETERS``
     - :doc:`memory`
   * - Loop
     - ``ALL_LOOP_SETTINGS``
     - :doc:`timing`
   * - Cache
     - ``ALL_CACHE_PARAMETERS``
     - :doc:`caching`
   * - Kernel
     - ``ALL_KERNEL_PARAMETERS``
     - :doc:`speed`

A kwarg matching none of the six sets raises ``KeyError`` at
``Solver.__init__`` (``solver.py:395-399``). Legacy timing spellings
(``dt_save``, ``dt_summarise``, ``dt_update_summaries``) also raise
``KeyError`` with a rename hint rather than being silently accepted.

Previously undocumented kwargs
-------------------------------

**blocksize**
    CUDA threads per block for the kernel launch, passed to
    :meth:`Solver.solve`. Default ``256``.

**max_registers**
    Per-thread register cap forwarded to ``cuda.jit`` via
    ``BatchSolverConfig`` (``src/cubie/batchsolving/BatchSolverConfig.py``,
    part of ``ALL_KERNEL_PARAMETERS``). Default ``None``, which leaves
    register allocation to ``ptxas``; capping trades spill traffic for
    more resident warps. See :doc:`speed`.

**step_controller**
    Selects the step-size controller by name: ``"fixed"``, ``"i"``,
    ``"pi"``, ``"pid"``, or ``"gustafsson"``
    (``src/cubie/integrators/step_control/base_step_controller.py:94-97``).
    When omitted, the chosen algorithm's own default controller is used
    (fixed for errorless tableaus, otherwise an algorithm-tuned PID — see
    :doc:`choosing_algorithms` and the per-algorithm "Defaults" sections
    in the API reference).

**mem_proportion**
    Proportion of VRAM (0.0-1.0) reserved for this solver's allocations
    (``src/cubie/memory/mem_manager.py:383,395-397``, part of
    ``ALL_MEMORY_MANAGER_PARAMETERS``). Default ``None``, which places
    the solver in the automatic pool and sizes its allocation from an
    equal share of the VRAM remaining after manually-proportioned
    instances are accounted for. See :doc:`memory`.
