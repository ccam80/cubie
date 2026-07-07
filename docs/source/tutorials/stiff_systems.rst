Tutorial 3: Stiff Systems, Implicit Methods, and Drivers
========================================================

A *stiff* system mixes very fast and very slow dynamics.  Explicit
methods must resolve the fastest timescale even after it has decayed
away, so they crawl; implicit methods take large steps through the
slow phase and only pay attention to the fast phase when it matters.
This tutorial solves a stiff oscillator with an implicit method,
tunes it, and adds a measured forcing signal.

Step 1: A stiff test problem
----------------------------

The Van der Pol oscillator with a large damping parameter ``mu`` is
the classic stiff benchmark — it creeps along a slow branch, then
relaxes almost instantaneously:

.. code-block:: python

   import numpy as np
   import cubie as qb

   vdp = qb.create_ODE_system(
       """
       dx = v
       dv = mu * (1 - x*x) * v - x
       """,
       parameters={"mu": 50.0},
       states={"x": 2.0, "v": 0.0},
       name="VanDerPol",
   )

Step 2: Solve with an implicit method
-------------------------------------

Swapping to a stiff solver is one word.  ``"rosenbrock"`` selects the
default Rosenbrock-W method (``ros3p``) — linearly implicit, adaptive,
and a good first choice because it needs no Newton iteration:

.. code-block:: python

   result = qb.solve_ivp(
       vdp,
       y0={"x": np.array([2.0]), "v": np.array([0.0])},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       method="rosenbrock",
       duration=20.0,
       save_every=0.05,
       rtol=1e-5,
       atol=1e-8,
   )
   assert np.all(result.status_codes == 0), result.status_messages

CuBIE derives the Jacobian your implicit solver needs symbolically
from the equations — nothing extra to supply.

For *very* stiff or badly behaved problems, the fully implicit
``method="radau_iia_5"`` is the heavy artillery; ``"crank_nicolson"``
and the DIRK family sit in between (:doc:`/user_guide/choosing_algorithms`).

Step 3: When the solver struggles
---------------------------------

Newton-based implicit methods (everything except Rosenbrock-W) report
their internal effort per run.  Check it before tuning blindly:

.. code-block:: python

   result = qb.solve_ivp(
       vdp,
       y0={"x": np.array([2.0]), "v": np.array([0.0])},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       method="backwards_euler",
       duration=20.0,
       dt=0.001,
       output_types=["state", "iteration_counters"],
       save_every=0.05,
   )

   newton_iters = result.iteration_counters[:, 0, :]
   print("worst save interval:", newton_iters.max())

If runs fail (``result.status_codes != 0``) or iteration counts sit
at the ceiling, the useful knobs, in the order worth trying:

1. ``dt_max=...`` — cap the step size; overly ambitious steps start
   Newton far from the solution.
2. ``max_newton_iters=...`` (default 100) — allow more iterations for
   slowly converging systems.
3. ``preconditioner_order=3`` (default 2) — stronger preconditioning
   speeds up the inner linear solves.
4. Newton/Krylov tolerances default to a tenth of your ``atol``/
   ``rtol``, which is usually right — tighten the outer tolerances
   rather than the inner ones.

The full list lives in :doc:`/user_guide/optional_arguments`.

Step 4: Add a forcing signal (driver)
-------------------------------------

Real experiments force their systems.  A *driver* is a time-dependent
input; here we drive the oscillator with a sampled signal, as if
replaying a measurement.  In the ``drivers`` dict, ``"time"`` is a
reserved key holding the timestamps the signals are sampled at; every
other entry names a driver whose array must match ``"time"`` in
length:

.. code-block:: python

   forced = qb.create_ODE_system(
       """
       dx = v
       dv = mu * (1 - x*x) * v - x + forcing
       """,
       parameters={"mu": 50.0},
       states={"x": 2.0, "v": 0.0},
       drivers=["forcing"],
       name="ForcedVanDerPol",
   )

   t_samples = np.linspace(0.0, 20.0, 400)
   signal = 5.0 * np.sin(2.0 * np.pi * 0.25 * t_samples)

   result = qb.solve_ivp(
       forced,
       y0={"x": np.array([2.0]), "v": np.array([0.0])},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       drivers={"forcing": signal, "time": t_samples},
       method="rosenbrock",
       duration=20.0,
       save_every=0.05,
       dt_max=0.05,
   )
   assert np.all(result.status_codes == 0), result.status_messages

Note the ``dt_max=0.05`` — knob 1 from Step 3 in action.  Without it,
one of these forced runs overreaches on step size and fails its linear
solves (``MAX_LINEAR_ITERATIONS_EXCEEDED``); capping the step at the
forcing timescale fixes it.

CuBIE fits a cubic spline through your samples so adaptive steppers
can evaluate the forcing at any time point, not just your sample
times.  Interpolation options (polynomial order, periodic wrapping,
boundary conditions) are covered in :doc:`/user_guide/drivers`.

Recap
-----

- Stiff system?  Start with ``method="rosenbrock"``.
- Still failing?  Check ``status_messages`` and
  ``iteration_counters`` before turning knobs.
- Forcing data?  Declare a driver and pass ``{"name": values,
  "time": times}``.
