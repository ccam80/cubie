from numba import cuda

from cubie.integrators.steps.baseAlgorithmStep import BaseAlgorithmStep


class ODEExplicitStep(BaseAlgorithmStep):
    def __init__(self):
        pass

    def build_step(self):

        @cuda.jit
        def explicit_step():
            pass