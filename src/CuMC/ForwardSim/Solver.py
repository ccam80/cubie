import numpy as np


class Solver:
    """
    User-facing class for batch-solving systems of ODEs. Accepts and sanitises user-world inputs, and passes them to
    GPU-world integrating functions. This class instantiates and owns a SolverKernel, which in turn interfaces
    distributes parameter and initial value sets to a groupd of SingleIntegratorRun device functions that perform
    each integration. The only part of this machine that the user must configure themself before using is the system
    model, which contains the ODEs to be solved.
    """
