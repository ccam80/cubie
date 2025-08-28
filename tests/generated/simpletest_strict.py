# -5891218735421558153

# This file was generated automatically by Cubie. Don't make changes in here - they'll just be overwritten! Instead, modify the sympy input which you used to define the file.
from numba import cuda


# AUTO-GENERATED JACOBIAN-VECTOR PRODUCT FACTORY
def jvp_factory(constants, precision):
    """Auto-generated Jacobian factory."""

    @cuda.jit(
        (precision[:], precision[:], precision[:], precision[:], precision[:]),
        device=True,
        inline=True,
    )
    def jvp(state, parameters, drivers, v, jvp):
        j_01 = 0
        j_11 = 2 * state[1]
        _cse0 = drivers[1] * parameters[0]
        aux_1 = _cse0 * state[0] + constants[0] * drivers[0]
        j_00 = _cse0
        j_10 = 2 * _cse0 + 1
        aux_2 = (
            aux_1
            + constants[1] * constants[1] * parameters[1]
            + state[0]
            + state[1] * state[1]
        )
        jvp[0] = j_00 * v[0] + j_01 * v[1]
        jvp[1] = j_10 * v[0] + j_11 * v[1]

    return jvp


# AUTO-GENERATED VECTOR-JACOBIAN PRODUCT FACTORY
def vjp_factory(constants, precision):
    """Auto-generated Jacobian factory."""

    @cuda.jit(
        (precision[:], precision[:], precision[:], precision[:], precision[:]),
        device=True,
        inline=True,
    )
    def vjp(state, parameters, drivers, v, vjp):
        j_01 = 0
        j_11 = 2 * state[1]
        _cse0 = drivers[1] * parameters[0]
        vjp[1] = j_01 * v[0] + j_11 * v[1]
        aux_1 = _cse0 * state[0] + constants[0] * drivers[0]
        j_00 = _cse0
        j_10 = 2 * _cse0 + 1
        aux_2 = (
            aux_1
            + constants[1] * constants[1] * parameters[1]
            + state[0]
            + state[1] * state[1]
        )
        vjp[0] = j_00 * v[0] + j_10 * v[1]

    return vjp


# AUTO-GENERATED DXDT FACTORY
def dxdt_factory(constants, precision):
    """Auto-generated dxdt factory."""

    @cuda.jit(
        (precision[:], precision[:], precision[:], precision[:], precision[:]),
        device=True,
        inline=True,
    )
    def dxdt(state, parameters, drivers, observables, dxdt):
        observables[0] = (
            constants[0] * drivers[0] + drivers[1] * parameters[0] * state[0]
        )
        dxdt[0] = constants[1] + observables[0]
        observables[1] = (
            constants[1] * constants[1] * parameters[1]
            + observables[0]
            + state[0]
            + state[1] * state[1]
        )
        dxdt[1] = constants[0] + observables[0] + observables[1]
    
    return dxdt
