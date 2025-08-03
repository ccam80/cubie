from typing import Optional, Union, List

from CuMC.ForwardSim.BatchConfigurator import BatchConfigurator
from CuMC.ForwardSim.BatchSolverKernel import BatchSolverKernel
from CuMC.ForwardSim.UserArrays import UserArrays
import numpy as np


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
                 saved_state_indices: Optional[List[Union[str | int]]] = None,
                 saved_observable_indices: Optional[List[Union[str | int]]] = None,
                 summarised_state_indices: Optional[List[Union[str | int]]] = None,
                 summarised_observable_indices: Optional[List[Union[str | int]]] = None,
                 output_types: list[str] = None,
                 precision: type = np.float64,
                 profileCUDA: bool = False,
                 ):
        super().__init__()
        self.kernel = BatchSolverKernel(system,
                                       algorithm=algorithm,
                                       duration=duration,
                                       warmup=warmup,
                                       dt_min=dt_min,
                                       dt_max=dt_max,
                                       dt_save=dt_save,
                                       dt_summarise=dt_summarise,
                                       atol=atol,
                                       rtol=rtol,
                                       saved_state_indices=saved_state_indices,
                                       saved_observable_indices=saved_observable_indices,
                                       summarised_state_indices=summarised_state_indices,
                                       summarised_observable_indices=summarised_observable_indices,
                                       output_types=output_types,
                                       precision=precision,
                                       profileCUDA=profileCUDA)

        self.batch_config = BatchConfigurator.from_system(system)

        self.UserArrays = UserArrays()
    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver. This will allow you to profile the performance of the solver on the
        GPU, but will slow things down.
        """
        # Consider disabling optimisation and enabling debug and line info for profiling
        self.profileCUDA = True

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver. This will stop profiling the performance of the solver on the GPU,
        but will speed things up.
        """
        self.profileCUDA = False

    @property
    def precision(self) -> type:
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.precision` from the child BatchSolverKernel object."""
        return self.kernel.precision

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.summary_legend_per_variable` from the child BatchSolverKernel object."""
        return self.kernel.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.saved_state_indices` from the child BatchSolverKernel object."""
        return self.kernel.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.saved_observable_indices` from the child BatchSolverKernel object."""
        return self.kernel.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.summarised_state_indices` from the child BatchSolverKernel object."""
        return self.kernel.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.summarised_observable_indices` from the child BatchSolverKernel object."""
        return self.kernel.summarised_observable_indices

    @property
    def active_output_arrays(self) -> 'ActiveOutputs':
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.active_output_arrays` from the child BatchSolverKernel object."""
        return self.kernel.active_output_arrays

    @property
    def state_dev_array(self) -> 'MappedArray':
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.state_dev_array` from the child BatchSolverKernel object."""
        return self.kernel.state_dev_array

    @property
    def observables_dev_array(self) -> 'MappedArray':
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.observables_dev_array` from the child BatchSolverKernel object."""
        return self.kernel.observables_dev_array

    @property
    def state_summaries_dev_array(self) -> 'MappedArray':
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.state_summaries_dev_array` from the child BatchSolverKernel object."""
        return self.kernel.state_summaries_dev_array

    @property
    def observable_summaries_dev_array(self) -> 'MappedArray':
        """Exposes :attr:`~CuMC.ForwardSim.BatchSolverKernel.observable_summaries_dev_array` from the child BatchSolverKernel object."""
        return self.kernel.observable_summaries_dev_array
