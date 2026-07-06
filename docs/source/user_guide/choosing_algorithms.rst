Choosing an Algorithm
=====================

CuBIE ships several integration algorithm families.  This page helps you
pick the right one.

Decision Guide
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Problem type
     - Recommended family
     - Notes
   * - Non-stiff
     - ERK
     - Fast per step; Dormand--Prince 5(4) is a good default.
   * - Mildly stiff
     - DIRK or Rosenbrock-W
     - DIRK is robust; Rosenbrock-W avoids Newton iteration.
   * - Very stiff
     - FIRK
     - Radau IIA 5 handles extreme stiffness well.
   * - Fixed step required
     - ``euler`` or non-adaptive tableau
     - Forward Euler for explicit, Backward Euler for implicit.

Available Algorithms
--------------------

**Explicit Runge--Kutta (ERK)**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``heun_21``
     - 2
     - Yes
     - Heun's method.
   * - ``bogacki_shampine_32``
     - 3(2)
     - Yes
     - Low-order, cheap.
   * - ``ralston_33``
     - 3
     - No
     - Ralston's method.
   * - ``dormand_prince_54`` / ``rk45``
     - 5(4)
     - Yes
     - Industry standard; good default.
   * - ``tsitouras_54`` / ``tsit5``
     - 5(4)
     - Yes
     - Often slightly more efficient than Dormand--Prince.
   * - ``dverk_65``
     - 6(5)
     - Yes
     - Verner's method.
   * - ``fehlberg_78`` / ``rk78``
     - 7(8)
     - Yes
     - High order; useful for smooth, high-accuracy problems.

**Diagonally Implicit Runge--Kutta (DIRK)**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``implicit_midpoint``
     - 2
     - No
     - Symmetric, energy-preserving.
   * - ``trapezoidal_dirk``
     - 2
     - No
     - Trapezoidal rule.
   * - ``sdirk_2_2``
     - 2
     - Yes
     - L-stable SDIRK.
   * - ``l_stable_dirk3``
     - 3
     - Yes
     - L-stable, 3 stages.
   * - ``l_stable_sdirk4``
     - 4
     - Yes
     - L-stable, 5 stages.
   * - ``lobatto_iiic_3``
     - 4
     - Yes
     - Default DIRK tableau.

**Fully Implicit Runge--Kutta (FIRK)**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``firk_gauss_legendre_2``
     - 4
     - Yes
     - 2-stage Gauss--Legendre; default FIRK.
   * - ``radau_iia_5`` / ``radau``
     - 5
     - Yes
     - 3-stage Radau IIA; excellent for stiff problems.

**Rosenbrock-W**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``ros3p``
     - 3
     - Yes
     - Default Rosenbrock; no Newton iteration.
   * - ``rodas3p``
     - 3
     - Yes
     - Stiffly accurate variant.
   * - ``rosenbrock_23_sciml``
     - 3
     - Yes
     - SciML-compatible tableau.

**Simple methods**

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``euler``
     - 1
     - No
     - Forward Euler; explicit.
   * - ``backwards_euler``
     - 1
     - No
     - Backward Euler; implicit, L-stable.
   * - ``backwards_euler_pc``
     - 1
     - No
     - Predictor-corrector backward Euler.
   * - ``crank_nicolson``
     - 2
     - No
     - Implicit trapezoidal rule.

Choosing a Controller
---------------------

Adaptive algorithms use an error controller to adjust the step size.
CuBIE supports several controllers:

**PI controller**
   The most common choice.  Balances responsiveness with stability.

**PID controller**
   Adds a derivative term for smoother step-size histories; can reduce
   oscillations in the step size on problems with sharp transients.

**Gustafsson controller**
   Predictive controller that accounts for the previous step's error
   ratio.  Useful when step rejections are frequent.

For most problems the default controller works well.  Adjust via
``step_control_settings`` on the :class:`~cubie.Solver`.

For the mathematical background behind these algorithms, see
:doc:`/theory/numerical_integration`.
