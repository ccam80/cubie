from numba import cuda

from cubie.integrators.algorithms_.base_algorithm_step import BaseAlgorithmStep


class ODEExplicitStep(BaseAlgorithmStep):
    def __init__(self):
        pass

    def build_step(self):

        @cuda.jit
        def explicit_step():
            pass