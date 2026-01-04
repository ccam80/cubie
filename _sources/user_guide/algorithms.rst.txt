ODE solving algorithms
======================

Solving ODEs seems like it should be straightforward - we know how much the
variables change at any instant in time (because we have the
\(\frac{dx}{dt}\) equations), so we can just keep adding up those changes from
some starting point and hey, presto, we have the solution! Alas, it is not that
simple. The key shortfall of this approach is that the derivatives are
continuous functions, and we can only calculate the change over a finite step in
time. Because the derivatives can change between the start and end of any size
step that we take, our calculated change can end up being slightly different
than the actual change.

Put simply, most real systems evolve in curvy lines, and we can only calculate
change in straight-line steps.

Euler's method
--------------

Euler did a lot of cool stuff. In the ODE world, the most intuitive method of
solving ODEs is named after him: Euler's method. In Euler's method, we calculate
the derivative at the start of the step, and move forward in that direction for
the whole time step. Euler's method says:

.. math::

   x_{n+1} = x_n + h f(t_n, x_n)


Runge-Kutta methods
-------------------

Runge-Kutta methods are a whole family of algorithms based on the same
underlying idea: instead of just calculating the gradient at the start of the
step, we calculate it at many sub-steps along the way, then average the results
to get a more accurate estimate of the change over the whole step. The most
commonly used Runge-Kutta method is probably the fourth-order Runge-Kutta method (RK4), which
calculates the gradient at four points along the step. One handy feature of
these algorithms is that we can calculate two at once - a fourth order and a
fifth order - without doing much more computation work. We find the difference
between these two estimates, and use that as an "error" estimate - if the error
is too high, we're probably not close to the true solution, so we can reduce the
step size and try again. This is the basis of adaptive-step algorithms, which
are very useful for solving ODEs in practice. Whenever there's a sharp or steep
corner in your variables, the average over the whole step will be inaccurate, so
an adaptive algorithm can take small steps on these sharp corners, and then take
big steps (so work faster) in nice straight-line sections.

.. _implicit-methods:

Implicit vs Explicit
--------------------

Explicit methods step forward through the sub-steps in time, calculating one,
then using that to calculate the next, and so on. Implicit methods, on the other
hand, use the solution to later sub-steps to calculate the solution to earlier
sub-steps. This makes them better at handling stiff systems (ones which have
very different time scales for different variables). To pull this off, however,
they need to solve the system of equations for all steps simultaneously, which
is computationally intense. Cubie uses a matrix-free solver for this process,
which means that it doesn't need to store big matrices in memory, allowing it to
solve much larger problems than would otherwise be possible.

To form the equations for this solver, Cubie needs the system's Jacobian matrix, which is the matrix of all partial derivatives
of the function (i.e. each element contains an expression for how f(x_i) will change if x_j changes. Cubie uses symbolic differentiation
to generate this when it's compiling the solver, and applies some algorithmic chain-rule steps to make the process a bit more
efficient.