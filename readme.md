# As-yet-unnamed CuNODE derivative (Tentatively: MCHammer)

A GPU-shaped hammer with which to hit Monte Carlo methods. While I have tried to generalise, this project was created with the end goal of running a likelihood-free particle filter using the SMC$^2$ methodology outlined in [1]. The code is written in Python, using Numba to generate CUDA device functions and kernels for the embarassingly parallel problem of simultaneous particle simulation.

This is also an opportunity to try out tests, versioning, and various CI-CD tools, so expect some rough edges.

Currently I am playing inside the __ForwardSim__ module, incorporating work from CuNODE and attempting to make it more general, so that I can use the integrators in various places inside SMC and maybe MCMC (PMMH/PGibbs) algorithms.