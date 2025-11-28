Getting Started with Cubie
==========================

cubie is a Python library designed to provide an easy entry point for users to
perform large-scale batch integration of ivps: from many initial values or with
many different parameter sets, or both, cubie don't care.

Installation
------------

Install cubie using pip:

.. code-block:: bash

   pip install cubie

Basic Usage
-----------

To use Cubie, you need to:

1. Define a system of ODEs
2. Solve a batch of IVPs

.. code-block:: python
   :caption: Creating and solving a system of ODEs.

   import cubie as qb
   import numpy as np

   LV = qb.create_ODE_system(
           """
               dx = a*x - b*x*y
               dy = -c*y + d*x*y
               """,
           states={'x': 100, 'y': 100},
           parameters={'a': 0.01, 'b': 1, 'c': 0.01, 'd': 1},
           name="LotkaVolterra")

   solution = qb.solve_ivp(LV,
                           {'x': np.arange(100), 'y': np.arange(100)},
                           duration=1.0)

This runs 10,000 different IVPs of the Lotka-Volterra equations, starting from
every combination of x and y each ranging from 0 to 99.

Features
--------

* Forward simulation of non-stiff and (soon) stiff systems of ODEs

Requirements
------------

* Python >= 3.8
* NumPy
* Numba
* Numba-CUDA
* attrs
* SymPy

Optional Dependencies
---------------------

* Cupy-cu12x: For pool-based memory management (if you're doing a lot of
  consecutive batches of different sizes)
* Pandas: For DataFrame output support
* Matplotlib: For plotting support. Only used to plot an interpolated driver function for sanity-checks (see
  :doc:`Drivers <user_guide/drivers>`)

GPU Requirements
~~~~~~~~~~~~~~~~

cubie requires an NVIDIA GPU with compute capability 6.0 or higher (see nvidia's
documentation for details). You must have CUDA toolkit installed. Currently,
only CUDA toolkit versions 12.7 - 12.9 are tested.
