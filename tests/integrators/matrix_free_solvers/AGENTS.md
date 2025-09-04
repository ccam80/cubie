In this submodule, we aim to create "solver" machinery for use in nonlinear ODE integrators.

## Context
The code is being written as part of a project to solve large batches of large nonlinear ODEs simultaneously on a GPU.
The main constraints in these systems are GPU shared memory (for work arrays) and memory bandwidth (for reading/writing to
global memory). Compile duration is not a concern, as the runtime of the simulations is expected to be much longer than the
compilation time. We hav complete control over every function being called, as we use very few external libraries inside
CUDA code. When considering an approach, take your time to consider whether the goal would be better achieved by modifying
the problem formulation, the form of arguments, or the workings of the functions called by the part that you are working on.

## Style
Follow PEP8; max line length 79 characters, comment length 71 characters. Do not add commits that explain what you are doing 
to the user; write comments that explain unconventional or complex operations to future developers. Write numpydocs-style
docstrings for all functions and classes. Write type hints for all functions and methods.