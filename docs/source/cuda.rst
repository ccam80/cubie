Using CUDA for IVPs
===================
Cubie takes advantage of the widespread availability of NVIDIA GPUs in modern workstation computers. NVIDIA have
developed an architecture called CUDA which allows developers to write software to be executed on their GPUs instead of
the computer's CPU.

What is a GPU?
--------------
Graphics Processing Units (GPUs) are a separate computing device from your computer's CPU (the main processor/brain of
the computer). They're often physically separate devices inside the computer box - they used to be referred to as
"graphics cards" because they come as a separate "card" that you plug in. GPUs were designed to process graphics
(unsurprisingly). Graphics processing requires a lot of independent calculations to be done in parallel (think one
calculation for each pixel on the screen), so GPUs have a whole heap of small, simple processors that can all work
separately in parallel.

What is CUDA
------------
CUDA is a framework that one GPU manufacturer, NVIDIA, has developed to allow users to write software that can run on
the many simple processors in their GPUs, turning them into practical computing devices rather than just graphics
processors. It turns out that the parallel processing power of GPUs is particularly useful for machine learning
problems, which is why NVIDIA have gone absolutely bananas as the worlds tech dollars have poured into AI development.

Cubie
-----
Cubie uses CUDA to solve initial value problems (IVPs) in parallel. Each IVP is worked on by one of the simple
processors in a GPU, so it's a fair bit slower (let's say 10x, as a approximation) than your computer can do them.
However, adding another one of those simple processors in parallel to solve another IVP is basically free, so as long as
you have more than 10 to do, the whole process ends up being faster. There are a lot of processors on a modern GPU - it takes
about as long to do ~32000 IVPs as it does to do one!

Limitations and bottlenecks
---------------------------
The main limitation of using CUDA for these problems is that GPUs don't have as much memory as the rest of the computer,
and writing to and from that memory can be much slower than writing to and from the CPU memory. The larger your ODE system,
the more numbers it has to store, and so the more memory it needs, and the less IVPs you can solve at once.
Cubie has a few tricks to get around this:
- **Constants vs Parameters**: Any parameters that will not change between IVPs in a single batch can be called
"constants". Constants don't need to be stored in the memory - they're baked into the actual code that the GPU processes
instead. This can cut down the required memory by a lot!
- **Matrix-free Jacobians**: Implicit algorithms require the solving of a system of simultaneous equations. The
straightforward approach for this is to set up a matrix of the system's Jacobian at each point, then use a matrix solver
to get the solution. If a system has n state variables, then the Jacobian is an n x n matrix. For a third-order
algorithm, the matrix to solve is 3n x 3n. For a 40-state system, without the jacobian, we typically need to store about
3 x n (120) numbers in memory. One Jacobian is 40 x 40 (1600) numbers, and three Jacobians is 12000 numbers. A whole "block"
of memory can only store about 15000 numbers (heavily simplified), so we go from being able to solve 125 IVPs per block to only being able to solve 1.
For this reason, Cubie uses Sympy to take a symbolic function and find it's analytical Jacobian, then turn that into
a function - a set of equations that is stored as code rather than numbers in a matrix. That way, we can calculate the
Jacobian on the fly, without needing to store the matrix!
- **Selective saving**: Saving all variables and observables at every time step can take up a lot of memory, and really
slows things down. Instead, Cubie allows you to pick and choose which variables you want to save. All states are critical
in working out the values at the next step, but you might only be interested in the outcome of one or two of them. Just
save those ones!
- **Live summarising**: For some applications, you might only be using the time-domain output of a solver to find some
summary statistic, like the maximum value of a variable, or the location of a "peak". In these cases, Cubie allows you to
calculate these summaries as-you-go, never saving the full time-domain output. This can drastically improve the speed of
the solver, and let you run much larger problems than you otherwise could.