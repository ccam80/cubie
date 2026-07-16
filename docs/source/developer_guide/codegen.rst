Code Generation Pipeline
========================

CuBIE transforms symbolic ODE definitions into compiled CUDA device
functions.  SymPy parses the input; every later stage runs on a
lightweight interned expression IR (the ``engine`` package).  This page
describes the pipeline.

Pipeline Overview
-----------------

::

   String equations
       → SymPy parser
       → IndexedBases (state/param/observable symbols)
       → engine IR (hash-consed expression nodes)
       → JVPEquations (Jacobian-vector product expressions)
       → IR printer → code strings
       → ODEFile (written to generated/ directory)
       → Numba JIT → CUDA device functions

Parser
------

The parser in ``src/cubie/odesystems/symbolic/parsing/`` tokenises the
equation strings, identifies states (variables with ``d<name>`` on the
left-hand side), parameters, constants, and observables, and produces
SymPy expressions.

IndexedBases
------------

State variables are represented as ``IndexedBase`` objects so that the
code generator can emit array-indexed CUDA code (e.g. ``x[0]``,
``x[1]``).

JVPEquations
------------

For implicit algorithms, the pipeline differentiates each RHS expression
with respect to every state variable, applies chain-rule grouping to
share subexpressions, and produces a function
:math:`(x, v) \mapsto J\,v`.  See :doc:`/theory/jacobians`.

Expression engine and printer
-----------------------------

``src/cubie/odesystems/symbolic/engine/`` holds the expression IR that
replaced SymPy in the compute phase: nodes are interned (structurally
identical expressions are the same object), so substitution,
differentiation, and common-subexpression elimination are single passes
over a DAG.  The engine's printer renders IR as Numba-CUDA source:

- Mapping function calls to their ``math.*``/builtin equivalents.
- Wrapping numeric literals in ``precision(...)`` casts.
- Rewriting small integer powers to multiplication chains and half
  powers to ``math.sqrt``.
- Emitting ``Piecewise`` selections as nested ternaries.

``print_cuda_multiple`` renders a list of assignments as source lines;
shared-subexpression extraction happens beforehand on the IR.

Solver Helpers
--------------

``get_solver_helper()`` dispatches to the appropriate code-generation
path based on the requested helper name (``"linear_operator"``,
``"prepare_jac"``, ``"calculate_cached_jvp"``,
``"time_derivative_rhs"``).

Stage Utilities
---------------

``_stage_utils`` provides helpers for FIRK methods that need to generate
code for multiple coupled stages simultaneously, including
block-structured linear algebra and transformation matrices.

Generated File Structure
------------------------

Generated files are written to ``generated/<system_name>/`` and include:

- ``rhs.py`` --- right-hand-side function.
- ``jvp.py`` --- Jacobian--vector product.
- ``prepare_jac.py`` --- shared subexpression cache.
- ``time_derivative.py`` --- explicit time derivative (for Rosenbrock-W).

These are standard Python files that Numba compiles on import.
