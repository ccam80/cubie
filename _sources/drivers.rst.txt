Drivers (Forcing Functions)
===========================

In some models, rates of change can be functions of time. This makes the ODE system "non-autonomous" - an "autonomous"
system is one where time isn't explicitly included in the equations. The trick that mathematicians use to handle this is
to add an extra state variable that changes at a constant rate - effectively, a clock. In Cubie, we skip the middle-state
and just allow you to use the symbol `t` in your equations, and it will be automatically evaluated with time as the
integration proceeds. If you can neatly express your variable as a function of time, this is the cleanest way to get a
forcing term into your problem. As an example, an experiment I was working on recently involved us shaking a tiny MEMS
cantilever using a piezoelectric actuator under its base, measuring how much the cantilever bent as a result, then feeding
that measured signal back into a heater on the cantilever that caused it to bend. That system could be modelled in Cubie
like this:

.. code-block::
    :linenos:

    import cubie as qb
    import numpy as np

    # Parameters
    k = 0.1      # cantilever spring constant
    c = 0.01     # cantilever damping coefficient (largely due to air)
    alpha = 0.5  # heater coupling coefficient (how much bend the heat caused)
    beta = 0.1   # heater dissipation coefficient (how quickly the heat dissipates)


    fns = ["base_wiggle = np.sin(2 * np.pi * f * t)",  # Example signal
        "f = 1e4 * t + 1e5", # a "chirp" signal, sweeping from 100kHz to 110 kHz over 1 second
        "dT = ",
        "di = ",
        "dx = ",
        "dv = ",
    ]
    constants = {}
    initial_conditions = {
        'x': 0,    # initial position
        'v': 0,    # initial velocity
        'T': 0,    # initial temperature
        'i': 0     # initial current
    }
    parameters = {
        feedback_strength = 0.5, #Amplitude of feedback signal
        feedback_offset = 0.1,   #Offset of feedback signal
    }

    sys = qb.create_ODE_system(
        fns=fns,
        parameters=parameters,
        constants=constants,
        states=initial_conditions)

    solve_ivp(
        sys,
        {'feedback_strength' : np.linspace(-1, 1, 200),
         'feedback_offset': np.linspace(-1,1,200)},
        algorithm='crank-nicolson')

This is an easy way to set up and parametrize a forcing term, and see how your system responds to different versions of
it.

Arbitrary Values
----------------
Sometimes, your forcing function might not be easily expressed as a function of time. You might have some measured data
that you want to test, you might want to use a random signal, or you might just want to throw some numbers in and see what
comes out. If your integrator is fixed-step and you can supply the value of the forcing term at each time step, this is
straightforward in theory - just pick a value for each time step, and use that. It's a bit limiting to only use fixed-step
algorithms, and if you're running a long-duration simulation at a high time resolution, you might need to store a lot of
time points. To give a bit more flexibility, Cubie can accept an array of forcing values and the time points to which they
correspond, and will interpolate between them to get the value at any time point. This allows you to use adaptive-step
algorithms, and also means you don't need to store the value at every time point if your forcing function is smooth. Here's
an example of how to do this:

.. code-block::


You have some levers that you can pull when defining your driver function. You can dictate whether the term should "wrap"
and repeat indefinitely, or whether it is set to zero outside of the range of the provided data. If it's returning to zero,
you can also dictate what happens at the ends. By default, it will fit a continuous smooth spline from your last data point
down to a zero one sample time step later.

Add plots of endpoints with different options