from numba import cuda

from cubie.integrators.loops.BaseIntegratorLoop import BaseIntegratorLoop


class ODEAdaptiveStepLoop(BaseIntegratorLoop):

    def __init__(self):
        pass

    def build_loop(self):

        @cuda.jit
        def adaptive_step_loop():
            pass