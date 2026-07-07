Tutorial 2: Summaries Instead of Trajectories
=============================================

A million-run batch that saves full trajectories will drown your GPU
memory long before it runs out of compute.  Usually you don't want the
trajectories anyway — you want a number per run: the mean, the peak
count, the oscillation amplitude.  This tutorial shows how to compute
those *on the GPU during integration*, so the full time series never
exists anywhere.

Step 1: A system worth summarising
----------------------------------

The same Lotka--Volterra system as :doc:`first_sweep` — its
populations oscillate, so per-run statistics are meaningful:

.. code-block:: python

   import numpy as np
   import cubie as qb

   LV = qb.create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

Step 2: Ask for statistics, not trajectories
--------------------------------------------

``output_types`` controls what is recorded.  Skip ``"state"`` and
list summary metrics instead:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={
           "b": np.linspace(0.01, 0.05, 20),
           "d": np.linspace(0.005, 0.02, 20),
       },
       method="dormand-prince-54",
       duration=50.0,
       output_types=["mean", "max"],
       summarise_every=50.0,
   )

``summarise_every=50.0`` makes one summary window spanning the whole
run.  Shorter windows (say ``summarise_every=10.0``) give you a
statistic per 10-time-unit window instead — handy for tracking slow
drift.  (If you omit it, CuBIE defaults to one whole-run window but
warns you, because the derived timing forces a recompile whenever
``duration`` changes.)

There are 18 built-in metrics, including ``"rms"``, ``"std"``,
``"peaks"``, and first/second-derivative extrema — the full table is
in :doc:`/user_guide/results`.

Step 3: Read the summaries
--------------------------

Summaries come out as a 3-D array indexed
``[window, summary, run]``.  The friendliest accessor is
``as_numpy_per_summary``, which splits it into one array per metric,
each indexed ``[window, variable, run]``:

.. code-block:: python

   per_metric = result.as_numpy_per_summary
   print(per_metric["max"].shape)      # (1, 2, 400)
   peak_prey = per_metric["max"][0, 0, :]   # window 0, x, all runs
   mean_prey = per_metric["mean"][0, 0, :]

One quirk to know: some metrics share an accumulator and get fused
when requested together.  Asking for both ``"max"`` and ``"min"``
computes the combined ``"extrema"`` metric, so the per-metric keys
become ``"extrema_1"`` and ``"extrema_2"`` rather than ``"max"`` and
``"min"``.  Print ``list(per_metric.keys())`` when in doubt.

Step 4: Trim what you don't need
--------------------------------

Two more levers cut memory and time further:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={
           "b": np.linspace(0.01, 0.05, 20),
           "d": np.linspace(0.005, 0.02, 20),
       },
       method="dormand-prince-54",
       duration=50.0,
       settling_time=20.0,                # discard the transient
       output_types=["mean", "max"],
       summarise_every=50.0,
       summarise_variables=["x"],         # only summarise prey
   )

``summarise_variables`` (and its time-domain sibling
``save_variables``) restricts recording to the variables you name.
``settling_time`` integrates for 20 time-units *before* recording
starts, so start-up transients don't pollute your statistics.
Settling extends the run rather than eating into it — the solver
integrates for ``settling_time + duration`` in total, so the recorded
window is still the full 50 time-units and ``summarise_every=50.0``
produces exactly one summary window per run.

When to use which output
------------------------

- **Full trajectories** (``"state"``): exploring, debugging, plotting
  a handful of runs.
- **Summaries**: large sweeps, likelihood-free inference, anything
  where each run reduces to features.  Memory per run drops from
  ``n_saves x n_variables`` values to a handful.
- **Both at once** works too — the two records run at independent
  cadences (see :doc:`/user_guide/timing`).

Next: :doc:`stiff_systems` for when your system fights back.
