Creating a "System" of Differential Equations
=============================================

The first step in solving a system of differential equations is to create a
system of differential equations. Cubie understands ODEs in the form of a
:class:`GenericODE<cubie.odesystems.systems.BaseODE>` object, which
holds the \(\frac{dx}{dt}\) equations and descriptions and default
values for all of the variables that those equations depend on.

Cubie uses SymPy to parse a system defined in strings, and generates a BaseODE for you.
This is the easiest way to create a system of ODEs. Here's an example of setting up the Lotka-Volterra
predator-prey equations, a common example of a nonlinear ODE system:

.. code-block:: python
    :caption: The simplest way to create a system of ODEs in Cubie.
    import cubie as qb

    LV = qb.create_ODE_system(
    """
        dx = a*x - b*x*y
        dy = -c*y + d*x*y
        """,
    name="LotkaVolterra",
    )

    print(LV)
In this example, Cubie has determined that x and y are state variables, as they have \(\frac{dx}{dt}\) equations.
\(\frac{dx}{dr}\) equations must start with a "d" followed by the variable name.
The variables a, b, c, and d have been called "parameters", as they don't have \(\frac{dx}{dt}\) equations, and they
don't appear on the left-hand side of any equations. Parameters are inputs to the system that
are expected to change between different runs of the solve; alongside the intial values of the state variables,
they form the input variables to a batch solver run. If any of your non-state input variables are not expected to
change inside each back, you can mark them as "constants" instead. This will typically speed up the solving process and
allow you to run more IVPs in a single batch, as Cubie doesn't need to worry about finding a place in memory for them.
To achieve this, we might modify the above example to:
.. code-block:: python
    :caption: Creating a system of ODEs with constants.
    import cubie as qb

    LV = qb.create_ODE_system(
    """
        dx = a*x - b*x*y
        dy = -c*y + d*x*y
        """,
    constants= {"a": 0.1, "c": 0.3},
    parameters={"b": 0.02, "d": 0.01},
    states={"x": 0.5, "y": 0.3},
    name="LotkaVolterra",
    )
    print(LV)
In the above example, we've also provided default values for the parameters and constants, and default initial values
for the states. This is optional, we could just provide a list or tuple of variable names instead, but it makes it a
little easier to specify which variables you're "batching" over when you come to solve the system.

There's one more type of variable that we can specify: observables. Observables are the results of auxiliary equations
in your problem that you might want to record the results of, but they're calculated anew at each time step rather than
carrying a state between iterations like a state. For a trivial example, we could define our Lotka-Volterra system with
an observable that tracks the predator's death rate:

.. code-block:: python
    :caption: Creating a system of ODEs with constants.
    import cubie as qb

    LV = qb.create_ODE_system(
    """
        predator_death_rate = c*y
        dx = a*x - b*x*y
        dy = -predator_death_rate + d*x*y
        """,
    constants= {"a": 0.1, "c": 0.3},
    parameters={"b": 0.02, "d": 0.01},
    states={"x": 0.5, "y": 0.3},
    observables=["predator_death_rate"],
    name="LotkaVolterra",
    )
    print(LV)

If we didn't define predator_death_rate as an observable, Cubie would treat it as an anonymous auxiliary variable, used
on the way to the results we want but not worth keeping. This behaviour applies to *all* left-hand side assignments that
do not target known states or listed observables: the variables still participate in the symbolic expressions, but they
are stored only as anonymous auxiliaries and their trajectories are not saved.

Cubie ODE System Glossary
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

Jacobians
---------
Implicit algorithms, such as the RadauIIA5 method that Cubie uses, require
the system's Jacobian. For why, see :ref:`Implicit Methods <implicit-methods>`.
Some widely-available solvers obtain this numerically by finite differences, which is
prone to error and instability, especially for stiff systems. Some solvers use auto-differentiation,
a clever way to get exact derivatives efficiently. Cubie does not currently support auto-differentiation, but
instead generates the required Jacobian functions with symbolically, with some manual chain-rule steps that
bring it closer to auto-differentiation and make it faster. The process isn't quick for big systems, but
once it's been done once, it's cached in a generated file in your working directory. You've only got to
pay the Jacobian tax once, unless you change some equations or constants, in which case the system needs
to generate everything again.