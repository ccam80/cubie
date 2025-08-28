Creating a "System" of Differential Equations
=============================================

The first step in solving a system of differential equations is to create a
system of differential equations. Cubie understands ODEs in the form of a
:class:`GenericODE<cubie.systemmodels.systems.GenericODE>` object, which
holds the critical \(\frac{dx}{dt}\) equations and descriptions and default
values for all of the variables that those equations depend on.

In my experience, there are many different sets of jargon and arbitrary
distinctions between variable types in different ODE-solving ecosystems.
Cubie has it's own set. The difference between these types is important for
the machinery that makes Cubie work, but probably isn't naturally intuitive
when you're just trying to solve a million IVPs with minimal fuss.

GenericODE Glossary
-------------------
- *States*: The variables that are being solved for. Each state variable
must have a \(\frac{dx}{dt}\) equation. Each state variable must also have
an initial value, which sets the starting point of the initial value problem.
- *Parameters*: Input variables that are not solved for. These set the
behaviour of the system, and in Cubie, they are one of the two inputs that
can be "batched", i.e. we can solve many IVPs with different parameter sets
simultaneously.
- *Constants*: Input variables that are not solved for, and do not change
between IVPs in a single batch. You can still change constants between
batches, but it will add a little overhead as the CUDA machine recompiles
the problem. Any parameters which will not change in a certain batch should
be moved to constants, as this will speed up the solving process.
- *Observables*: Also called auxiliary variables. These variables that are not
solved for, but are derived from the state inputs and parameters. These
typically pop up on the way to the \(\frac{dx}{dt}\) equations, and might
represent physical quantities of interest in the system. Any state variables
that don't have a \(\frac{dx}{dt}\) equation should be moved into observables.
- *Drivers*: Also called forcing terms. These are time-dependent inputs to
the system. Cubie currently only supports one set of drivers per batch (i.e.
all IVPs use the same driver), but this can be worked around by parameterising
the driver function and passing a time vector as the driver function.

Subclassing GenericODE
----------------------
One method for creating a system of ODEs, if you're Python-literate, is to
subclass the genericODE class. This is the most flexible way to create an
ODE system, but it requires a little bit of work on your part to set up the
equations in such a way that the underlying CUDA machinery can understand it
. See :file:`cubie/systemmodels/systems/ThreeCM.py` for an example of this.

Sympy-driven :class:`SymbolicODE<cubie.systemmodels.systems.SymbolicODE>`
-------------------------------------------------
If you want a more MATLAB-like experience, you can write the ODEs out in
mathematical notation using the Sympy library, and then pass them to Cubie.
This is the most user-friendly way to create a system of ODEs, and it
requires the least amount of work on your part. [create example].

Jacobians
---------
Implicit algorithms, such as the RadauIIA5 method that Cubie uses, require
the system's Jacobian. For why, see :ref:`Implicit Methods <implicit-methods>`.
You can add your own Jacobian function to the GenericODE system when
subclassing. Jacobian functions are typically large and hard to write, so if
you instead create a SymbolicODE system, Cubie will automatically generate the
Jacobian through Sympy's symbolic differentiation. This step can take some
time to execute, but is only done once per system, and the resulting Jacobian
is cached for future use.