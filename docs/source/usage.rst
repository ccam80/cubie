Using cubie
===========

cubie is a Python library designed to provide an easy entry point for users to perform large-scale batch integration of ivps: from many initial values or with many different parameter sets, or both, cubie don't care.

Installation
------------

Install cubie using pip:

.. code-block:: bash

   pip install cubie

Basic Usage
-----------

Here's a simple example of how to use cubie:

.. code-block:: python

   import cubie

   # Example usage will be added as the API develops
   # The library provides GPU-accelerated Monte Carlo simulations
   # for parameter estimation in ODE systems

Features
--------

* Forward simulation of non-stiff and (soon) stiff systems of ODEs

Requirements
------------

* Python >= 3.8
* NumPy
* Numba
* Numba-CUDA
* SciPy
* attrs


Optoinal Dependencies
---------------------
* Cupy-cu12x: For pool-based memory management (if you're doing a lot of consecutive batches of different sizes)
*
GPU Requirements
~~~~~~~~~~~~~~~~

cubie requires an NVIDIA GPU with compute capability 6.0 or higher (see nvidia's documentation for details). The library is designed to leverage the power of CUDA for efficient computation.
* Numba-CUDA for GPU kernel compilation

Make sure you have appropriate CUDA drivers installed for your system.
