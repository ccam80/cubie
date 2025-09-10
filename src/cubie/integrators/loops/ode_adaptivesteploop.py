from numba import cuda

from cubie.integrators.loops.ivp_loop import IVPLoop


class ODEAdaptiveStepLoop(IVPLoop):

    def __init__(self):
        pass

    def build_loop(self):

        @cuda.jit
        def adaptive_step_loop():
            pass