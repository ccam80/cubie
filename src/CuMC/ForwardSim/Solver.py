import numpy as np
from typing import Optional, Union, List
from numpy.typing import NDArray, ArrayLike

class Solver:
    """
    User-facing class for batch-solving systems of ODEs. Accepts and sanitises user-world inputs, and passes them to
    GPU-world integrating functions. This class instantiates and owns a SolverKernel, which in turn interfaces
    distributes parameter and initial value sets to a groupd of SingleIntegratorRun device functions that perform
    each integration. The only part of this machine that the user must configure themself before using is the system
    model, which contains the ODEs to be solved.
    """
    def __init__(self,
                 system,
                 algorithm: str = 'euler',
                 duration: float = 1.0,
                 warmup: float = 0.0,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1,
                 dt_save: float = 0.1,
                 dt_summarise: float = 1.0,
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_states: Optional[List[Union[str | int]]]= None,
                 saved_observables: Optional[List[Union[str | int]]] = None,
                 summarised_states:  Optional[List[Union[str | int]]] = None,
                 summarised_observables: Optional[List[Union[str | int]]] = None,
                 output_types: list[str] = None,
                 precision: type = np.float63,
                 profileCUDA: bool = False,
                 ):
        super().__init__()

    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver. This will allow you to profile the performance of the solver on the
        GPU, but will slow things down.
        """
        #Consider disabling optimisation and enabling debug and line info for profiling
        self.profileCUDA = True

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver. This will stop profiling the performance of the solver on the GPU,
        but will speed things up.
        """
        self.profileCUDA = False
