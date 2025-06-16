Currently unhoused info:
- Noise sigmas - these are used only in a Euler-maruyama loop (currently), so should belong to the integrator. This may require experimentation.s

The default constants/parameters dict layout makes it tricky to move terms between parameters and constants to optimise execution between runs with different
fixed/free parameters.

Having initial values inside the system seems like a big fudge. Why would they be in there? Perhaps they could be in there as defaults only?

Move back to cupy arrays for mem managment help

x = cuda.threadIdx.x
bx = cuda.blockIdx.x
if x == 0 and bx == 0:
    from pdb import set_trace;
    set_trace()