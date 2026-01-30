CellML Models
=============

`CellML <https://www.cellml.org/>`__ is an XML-based markup language for
describing mathematical models of biological systems.  CuBIE can import
CellML files and convert them into
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE` objects.

Loading a CellML Model
-----------------------

.. code-block:: python

   import cubie as qb

   system = qb.load_cellml_model(
       "path/to/model.cellml",
       parameters=["g_Na", "g_K"],
       observables=["I_Na", "I_K"],
   )

All symbols not listed as ``parameters`` or ``observables`` (and not
identified as state variables) are treated as constants.

Optional arguments:

``precision``
   ``np.float32`` (default) or ``np.float64``.

``name``
   Override the system name (defaults to the filename).

``show_gui``
   Launch the interactive variable-classification editor.

Installing ``cellmlmanip``
--------------------------

CellML support requires the optional ``cellmlmanip`` package:

.. code-block:: bash

   pip install cellmlmanip

Known Caveats
-------------

- Only ODE-based CellML models are supported; DAE or algebraic-only
  models will raise an error.
- Some CellML 2.0 features may not be fully handled by ``cellmlmanip``.
- Large CellML models (hundreds of states) may take noticeable time to
  parse and differentiate on first use; subsequent runs use the cache.
