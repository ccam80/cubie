Using CuMC
==========

CuMC is a Python library designed to provide an easy entry point for users to perform likelihood-free Monte Carlo parameter estimation. To facilitate this, the library includes a set of forward-simulating ODE solvers designed to integrate large batches of IVPs from different initial values or with different parameters.

Installation
------------

Install CuMC using pip:

.. code-block:: bash

   pip install CC_CuMC

Basic Usage
-----------

Here's a simple example of how to use CuMC:

.. code-block:: python

   import CuMC

   # Example usage will be added as the API develops
   # The library provides GPU-accelerated Monte Carlo simulations
   # for parameter estimation in ODE systems

Features
--------

* **GPU Acceleration**: Utilizes CUDA through Numba for high-performance computing
* **Forward Simulation**: Efficient ODE solvers for batch integration
* **Monte Carlo Methods**: Tools for likelihood-free parameter estimation
* **System Models**: Pre-built and customizable system models
* **Sampling**: Advanced sampling techniques for parameter space exploration

Requirements
------------

* Python >= 3.8
* NumPy
* Numba
* Numba-CUDA
* CuPy (CUDA 12.x)
* SciPy

GPU Requirements
~~~~~~~~~~~~~~~~

CuMC requires a CUDA-compatible GPU for optimal performance. The library uses:

* Numba-CUDA for GPU kernel compilation
* CuPy for GPU array operations

Make sure you have appropriate CUDA drivers installed for your system.
