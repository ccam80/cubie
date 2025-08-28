#9129000395194988293

# This file was generated automatically by Cubie. Don't make changes in here - they'll just be overwritten! Instead, modify the sympy input which you used to define the file.
from numba import cuda




# AUTO-GENERATED DXDT FACTORY
def dxdt_factory(constants, precision):
    """Auto-generated dxdt factory."""
    @cuda.jit((precision[:],
               precision[:],
               precision[:],
               precision[:],
               precision[:]),
              device=True,
              inline=True)
    def dxdt(state, parameters, driver, observables, dxdt):
        _cse0 = constants[0]*state[0] - constants[1]*state[1]
        dxdt[1] = _cse0
        dxdt[0] = -_cse0
    
    return dxdt
