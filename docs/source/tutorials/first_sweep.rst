Tutorial 1: Your First Parameter Sweep
======================================

This tutorial walks the shortest path from "I have an ODE" to "I have
a heatmap of how its behaviour varies over parameters".  Total code:
about 30 lines.

We'll use the Lotka--Volterra predator--prey equations: prey ``x``
grows and gets eaten, predators ``y`` eat and die off.

Step 1: Define the system
-------------------------

Write the equations as strings.  Anything of the form ``dx = ...``
defines a state variable ``x``; everything else on the right-hand side
is declared as a parameter or constant:

.. code-block:: python

   import numpy as np
   import cubie as qb

   LV = qb.create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       constants={"a": 0.1, "c": 0.3},     # fixed for the whole batch
       parameters={"b": 0.02, "d": 0.01},  # can vary per run
       states={"x": 0.5, "y": 0.3},        # initial values
       name="LotkaVolterra",
   )

The split matters: *parameters* can take a different value in every
run of the batch, while *constants* are baked into the compiled GPU
code (which makes the kernel faster).  If you won't sweep it, make it
a constant.

Step 2: Sweep two parameters
----------------------------

Pass an array for each parameter you want to sweep.  With the default
``grid_type="combinatorial"``, CuBIE solves every combination — here
20 x 20 = 400 IVPs, all in parallel on the GPU:

.. code-block:: python

   b_values = np.linspace(0.01, 0.05, 20)
   d_values = np.linspace(0.005, 0.02, 20)

   result = qb.solve_ivp(
       LV,
       y0={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={"b": b_values, "d": d_values},
       method="dormand-prince-54",
       duration=50.0,
       save_every=0.5,
   )

``method`` selects the integration algorithm —
``"dormand-prince-54"`` is the same 5th-order adaptive method behind
MATLAB's ``ode45`` and a good default for non-stiff problems.
``save_every=0.5`` records a snapshot every half time-unit.

Step 3: Look at the results
---------------------------

The returned :class:`~cubie.batchsolving.solveresult.SolveResult`
holds a 3-D array indexed ``[time, variable, run]``:

.. code-block:: python

   trajectories = result.time_domain_array
   print(trajectories.shape)        # (101, 2, 400)
   print(result.time_domain_legend) # which variable is which index

   # Trajectory of prey (variable 0) in the very first run:
   prey_run0 = trajectories[:, 0, 0]

Before trusting numbers, check that every run integrated cleanly:

.. code-block:: python

   assert np.all(result.status_codes == 0), result.status_messages

Step 4: Map runs back to parameters
-----------------------------------

Combinatorial runs are laid out in C-order: the *last* parameter you
passed (``d``) varies fastest.  A ``reshape`` therefore recovers the
2-D parameter grid directly:

.. code-block:: python

   final_prey = trajectories[-1, 0, :]         # final x, every run
   final_prey_grid = final_prey.reshape(20, 20)  # rows: b, cols: d

Step 5: Plot
------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   im = ax.imshow(
       final_prey_grid,
       origin="lower",
       extent=[d_values[0], d_values[-1], b_values[0], b_values[-1]],
       aspect="auto",
   )
   ax.set_xlabel("d (predator growth per prey eaten)")
   ax.set_ylabel("b (predation rate)")
   ax.set_title("Final prey population")
   fig.colorbar(im)
   fig.savefig("lv_sweep.png", dpi=150)

That's the whole workflow: define once, sweep in one call, reshape,
plot.

Where to go next
----------------

- Sweep initial values too — pass arrays in ``y0`` exactly like
  parameters (:doc:`/user_guide/batching`).
- Record statistics (mean, peaks, ...) instead of full trajectories to
  save memory: :doc:`extracting_summaries`.
- Solving something stiff?  See :doc:`stiff_systems`.
