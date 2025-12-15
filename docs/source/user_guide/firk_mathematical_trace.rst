FIRK Step Implementation Analysis
=================================

This document provides a detailed mathematical trace of CuBIE's Fully Implicit
Runge-Kutta (FIRK) step implementation in ``generic_firk.py``, comparing it to
OrdinaryDiffEq.jl's Radau implementation and textbook formulations.

Textbook FIRK Formulation
-------------------------

Standard Fully Implicit RK Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an ODE system:

.. math::

    \frac{dy}{dt} = f(t, y)

A fully implicit Runge-Kutta method with :math:`s` stages computes:

**Stage values:**

.. math::

    Y_i = y_n + h \sum_{j=1}^{s} a_{ij} f(t_n + c_j h, Y_j), \quad i = 1, \ldots, s

**Update:**

.. math::

    y_{n+1} = y_n + h \sum_{i=1}^{s} b_i f(t_n + c_i h, Y_i)

**Error estimate (if embedded):**

.. math::

    \hat{y}_{n+1} = y_n + h \sum_{i=1}^{s} \hat{b}_i f(t_n + c_i h, Y_i)

**Error:**

.. math::

    \text{err} = y_{n+1} - \hat{y}_{n+1} = h \sum_{i=1}^{s} (b_i - \hat{b}_i) f(t_n + c_i h, Y_i) = h \sum_{i=1}^{s} d_i f(t_n + c_i h, Y_i)

where :math:`d_i = b_i - \hat{b}_i`.

Stage Increment Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative formulation uses **stage increments** instead of stage values:

Let :math:`k_i = f(t_n + c_i h, Y_i)` be the stage derivatives.

**Stage equations (in terms of increments):**

.. math::

    Y_i = y_n + h \sum_{j=1}^{s} a_{ij} k_j

.. math::

    k_i = f\left(t_n + c_i h, y_n + h \sum_{j=1}^{s} a_{ij} k_j\right)

This is a coupled nonlinear system in the unknowns :math:`k_1, \ldots, k_s`.

**Newton iteration for coupled system:**

Define the residual for the stacked unknowns :math:`K = [k_1; k_2; \ldots; k_s]`:

.. math::

    G(K) = K - F(Y(K))

where :math:`Y_i(K) = y_n + h \sum_{j=1}^{s} a_{ij} k_j`.

The Jacobian of :math:`G` with respect to :math:`K` is:

.. math::

    \frac{\partial G}{\partial K} = I - h (A \otimes J)

where :math:`A` is the :math:`s \times s` coefficient matrix and :math:`J` is
the :math:`n \times n` Jacobian of :math:`f` with respect to :math:`y`.

Radau IIA Special Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Radau IIA methods:

- The last stage node is :math:`c_s = 1` (stiffly accurate)
- The last row of :math:`A` equals :math:`b^T`: :math:`a_{s,j} = b_j` for all :math:`j`
- This means :math:`Y_s = y_{n+1}` (no separate update needed)
- Error estimation uses :math:`\hat{b}` with :math:`d_i = b_i - \hat{b}_i`


CuBIE FIRK Implementation
-------------------------

Residual Formulation
~~~~~~~~~~~~~~~~~~~~

CuBIE uses a **stage increment formulation** where the unknowns are the stage
increments (slopes) :math:`k_i`, not the stage values :math:`Y_i`.

From ``_build_n_stage_residual_lines()`` in ``nonlinear_residuals.py``:

For each stage :math:`i` and component :math:`\ell`:

1. Compute evaluation state: :math:`Y_i = \text{base\_state} + \sum_j a_{ij} \cdot u_{j}`
2. Evaluate derivative at stage state: :math:`dx_{i,\ell} = f(t + h \cdot c_i, Y_i)`
3. Residual: :math:`\text{out}_{i,\ell} = \beta \cdot M \cdot u_{i,\ell} - \gamma \cdot h \cdot dx_{i,\ell}`

**Key observation:** CuBIE solves for **scaled increments** :math:`u_i` where:

.. math::

    u_i = h \cdot k_i

This is visible because the residual is:

.. math::

    \beta M u_i - \gamma h f(Y_i) = 0 \Rightarrow u_i = \frac{\gamma h}{\beta} M^{-1} f(Y_i)

With :math:`\beta = \gamma = 1` and :math:`M = I`: :math:`u_i = h \cdot f(Y_i) = h \cdot k_i`

Stage State Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~

After the Newton solve converges, the step function reconstructs stage states
(``generic_firk.py``):

.. math::

    Y_i = y_n + \sum_{j=1}^{s} a_{ij} \cdot u_j

where :math:`u_j` are the solved stage increments.

Output Accumulation
~~~~~~~~~~~~~~~~~~~

When ``accumulates_output`` is True:

.. math::

    y_{n+1} = y_n + h \sum_{i=1}^{s} b_i \cdot f(t + c_i h, Y_i)

**Note:** This requires re-evaluating :math:`f(Y_i)` at each stage state!

Error Accumulation
~~~~~~~~~~~~~~~~~~

When ``accumulates_error`` is True:

.. math::

    \text{err} = h \sum_{i=1}^{s} d_i \cdot f(t + c_i h, Y_i)

Optimized Path: Stiffly Accurate Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Radau IIA where :math:`b = a[-1]` (stiffly accurate), CuBIE directly uses
:math:`Y_s` as :math:`y_{n+1}`, skipping the :math:`b`-weighted accumulation.


OrdinaryDiffEq.jl Radau Implementation
--------------------------------------

Core Algorithm
~~~~~~~~~~~~~~

OrdinaryDiffEq.jl uses a **transformed stage increment** formulation for efficiency.

**Key differences from CuBIE:**

1. **Transformation to complex eigenvalue space:** The coupled :math:`s \times s`
   block system is transformed using the eigenvectors of :math:`A`, reducing it
   to independent complex-valued systems.

2. **Stage increments stored differently:** They store :math:`Z = [z_1; z_2; \ldots; z_s]`
   where :math:`z_i` are related to stage increments by the transformation matrix :math:`T`.

3. **Newton iteration on transformed variables:**

   .. math::

       T^{-1} A T = \Lambda \text{ (diagonal in complex space)}

4. **Simplified LU factorization:** Only need to factor :math:`I - \gamma h J`
   for a few eigenvalues, not the full block system.


Comparison and Analysis
-----------------------

Formulation Differences
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - CuBIE
     - OrdinaryDiffEq.jl
   * - Unknowns
     - Stage increments :math:`u_i = h k_i`
     - Transformed increments :math:`z_i`
   * - Newton system
     - Full block :math:`(I \otimes M - h A \otimes J)`
     - Decoupled via eigendecomposition
   * - Preconditioner
     - Neumann series
     - Block LU with eigenvalue transformation
   * - Stiffly accurate opt.
     - Yes (``b_matches_a_row``)
     - Yes (built-in)

Potential Duplicated Work in CuBIE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue 1: Re-evaluation of f(Y_i) for output accumulation**

In ``generic_firk.py``, the post-solve loop re-evaluates :math:`f(Y_i)` at each
stage state AFTER the Newton iteration has converged.

**Why this happens:**

- The Newton solver solves for stage increments :math:`u`
- At convergence, we have :math:`u^* = h f(Y(u^*))` (for :math:`M=I, \beta=\gamma=1`)
- BUT: The Newton solver doesn't expose :math:`f(Y_i)` values - only the converged :math:`u^*`

**Issue 2: Stage state reconstruction**

The stage state :math:`Y_i` is reconstructed in the post-solve loop, but it was
ALREADY computed inside the residual function during Newton iteration. This is
truly duplicated work.

Comparison with Textbook
~~~~~~~~~~~~~~~~~~~~~~~~

CuBIE's formulation is mathematically correct:

1. **Residual:** Correctly implements :math:`G(K) = M K - h F(Y(K))` with :math:`K = u` (scaled increments)
2. **Update:** Correctly uses :math:`y_{n+1} = y_n + h \sum b_i k_i` when accumulating
3. **Error:** Correctly uses :math:`\text{err} = h \sum d_i k_i`
4. **Stiffly accurate:** Correctly recognizes when :math:`Y_s = y_{n+1}`

Potential Error Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

**No mathematical errors found in the formulation.**

However, there are **efficiency issues**:

1. Stage states :math:`Y_i` computed twice: once in residual, once after solve
2. :math:`f(Y_i)` computed twice: once in residual, once for output/error accumulation


Can We Skip Base State F(y) Recalculation?
------------------------------------------

The Question
~~~~~~~~~~~~

At the end of the Newton solve, we have converged stage increments :math:`u_i^*`.

**Can we derive f(Y_i^*) from u_i^* without re-evaluating the ODE?**

Analysis
~~~~~~~~

At Newton convergence, the residual is zero (within tolerance):

.. math::

    \beta M u_i^* - \gamma h f(t_i, Y_i^*) \approx 0

Therefore:

.. math::

    f(t_i, Y_i^*) \approx \frac{\beta}{\gamma h} M u_i^*

For the standard case :math:`\beta = \gamma = 1`, :math:`M = I`:

.. math::

    f(Y_i^*) \approx \frac{u_i^*}{h}

**However**, this is only approximate because Newton stops at a tolerance,
not exact convergence.

Using Stage Increments for Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For output (y_{n+1}):**

Textbook: :math:`y_{n+1} = y_n + h \sum b_i k_i = y_n + h \sum b_i f(Y_i)`

With :math:`u_i = h k_i`: :math:`y_{n+1} = y_n + \sum b_i u_i`

**This is exact and requires NO re-evaluation!**

**For error:**

Textbook: :math:`\text{err} = h \sum d_i f(Y_i)`

With :math:`u_i = h k_i`: :math:`\text{err} = \sum d_i u_i`

**This is also exact and requires NO re-evaluation!**


Conclusion: Base State F(y) Recalculation
-----------------------------------------

Summary
~~~~~~~

**Question:** Can CuBIE's recalculation of F(y) at the base state be skipped,
using stage increments instead?

**Answer: YES.**

The stage increments ``stage_increment[i]`` store :math:`u_i = h \cdot k_i = h \cdot f(Y_i)`
at convergence.

**For output accumulation:**

- Current: ``proposed_state = state + dt_scalar * Σ b_i * f(Y_i)``
- Equivalent: ``proposed_state = state + Σ b_i * stage_increment[i]``

**For error accumulation:**

- Current: ``error = dt_scalar * Σ d_i * f(Y_i)``
- Equivalent: ``error = Σ d_i * stage_increment[i]``

**The re-evaluation of dxdt_fn in the post-solve loop is mathematically
unnecessary** when computing output and error, as the stage increments already
encode :math:`h \cdot f(Y_i)`.

Potential Savings
~~~~~~~~~~~~~~~~~

- Eliminates :math:`s` ODE function evaluations per step (where :math:`s` is the number of stages)
- Eliminates :math:`s` observables function evaluations per step
- Eliminates redundant stage state reconstruction (already done in residual)

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

The optimization applies when:

1. ``accumulates_output`` is True (standard :math:`b`-weighted output)
2. ``accumulates_error`` is True (standard :math:`d`-weighted error)

For stiffly accurate methods (Radau IIA), the optimization is already partially
implemented via ``b_matches_a_row`` and ``b_hat_matches_a_row`` which skip
accumulation entirely.

Recommendations
~~~~~~~~~~~~~~~

1. **Remove f(Y_i) re-evaluation for output/error:**

   - Use ``stage_increment[i]`` directly instead of ``stage_rhs_flat[i]``
   - Remove the ``dt_scalar`` multiplication since :math:`u = h k`

2. **Simplify when stiffly accurate:**

   - Already optimized for ``b_matches_a_row``
   - Consider similar optimization for error when applicable

3. **Consider caching f values from Newton:**

   - The final Newton residual evaluation computes :math:`f(Y^*)`
   - Could expose this to avoid any re-computation


Appendix: Radau IIA Tableau
---------------------------

For reference, the Radau IIA 5th-order (3-stage) tableau coefficients:

- :math:`c` - Stage nodes (quadrature points within the step interval :math:`[t_n, t_n+h]`)
- :math:`A` - Coupling matrix defining how stages influence each other
- :math:`b` - Solution weights for combining stage derivatives

.. math::

    c = \begin{pmatrix} \frac{4-\sqrt{6}}{10} \\ \frac{4+\sqrt{6}}{10} \\ 1 \end{pmatrix}

.. math::

    A = \begin{pmatrix}
    \frac{88-7\sqrt{6}}{360} & \frac{296-169\sqrt{6}}{1800} & \frac{-2+3\sqrt{6}}{225} \\
    \frac{296+169\sqrt{6}}{1800} & \frac{88+7\sqrt{6}}{360} & \frac{-2-3\sqrt{6}}{225} \\
    \frac{16-\sqrt{6}}{36} & \frac{16+\sqrt{6}}{36} & \frac{1}{9}
    \end{pmatrix}

.. math::

    b = \begin{pmatrix} \frac{16-\sqrt{6}}{36} & \frac{16+\sqrt{6}}{36} & \frac{1}{9} \end{pmatrix}^T

Note: :math:`b = A_{3,:}` (stiffly accurate), so :math:`Y_3 = y_{n+1}`.
