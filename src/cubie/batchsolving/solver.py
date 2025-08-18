from typing import Optional, Union, List
from typing import TYPE_CHECKING
import warnings

import numpy as np
from cubie.memory import default_memmgr
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solveresult import SolveResult
from cubie.batchsolving.solveresult import SolveSpec


if TYPE_CHECKING:
    from numba.cuda.cudadrv import MappedNDArray
def solve_ivp(
    system,
    y0,
    parameters,
    forcing_vectors,
    dt_eval,
    method="euler",
    duration=1.0,
    settling_time=0.0,
    **options,
) -> SolveResult:
    """ Solve a batch problem using the provided system model and
    parameters. This is a convenience function that
    creates a Solver instance and calls its solve method. It is intended for
    one-off batch solves where the user
    doesn't mind the overhead of creating and destroying a Solver instance.
    For repeated solves, it is recommended to
    instantiate a Solver object and use its solve method, to take advantage
    of it reusing some expensive components.
    """
    solver = Solver(system,
                    algorithm=method,
                    dt_save=dt_eval,
                    duration=duration,
                    warmup=settling_time,
                    **options)
    results = solver.solve(y0,
                         parameters,
                         forcing_vectors,
                         duration=duration,
                         warmup=settling_time,
                         **options)



class Solver:
    """
    User-facing class for batch-solving systems of ODEs. Accepts and
    sanitises user-world inputs, and passes them to
    GPU-world integrating functions. This class instantiates and owns a
    SolverKernel, which in turn interfaces
    distributes parameter and initial value sets to a groupd of
    SingleIntegratorRun device functions that perform
    each integration. The only part of this machine that the user must
    configure themself before using is the system
    model, which contains the ODEs to be solved.
    """

    def __init__(
        self,
        system,
        algorithm: str = "euler",
        duration: float = 1.0,
        warmup: float = 0.0,
        dt_min: float = 0.01,
        dt_max: float = 0.1,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        saved_states: Optional[List[Union[str | int]]] = None,
        saved_observables: Optional[List[Union[str | int]]] = None,
        summarised_states: Optional[List[Union[str | int]]] = None,
        summarised_observables: Optional[List[Union[str | int]]] = None,
        output_types: list[str] = None,
        precision: type = np.float64,
        profileCUDA: bool = False,
        memory_manager=default_memmgr,
        stream_group="solver",
        mem_proportion=None,
        **kwargs,
    ):
        super().__init__()
        interface = SystemInterface.from_system(system)
        self.system_interface = interface

        (saved_state_indices,
         saved_observable_indices,
         summarised_state_indices,
         summarised_observable_indices) = self._variable_indices_from_list(
                saved_states,
                saved_observables,
                summarised_states,
                summarised_observables)


        self.grid_builder = BatchGridBuilder(interface)

        self.kernel = BatchSolverKernel(
            system,
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
            profileCUDA=profileCUDA,
            memory_manager=memory_manager,
            stream_group=stream_group,
            mem_proportion=mem_proportion,
        )

    def _variable_indices_from_list(self,
                                    saved_states,
                                    saved_observables,
                                    summarised_states,
                                    summarised_observables):
        """Get integer indices from SystemInterface given a list of labels"""
        if saved_states is not None:
            saved_state_indices = self.system_interface.state_indices(saved_states)
        else:
            saved_state_indices = None
        if saved_observables is not None:
            saved_observable_indices = self.system_interface.observable_indices(
                saved_observables)
        else:
            saved_observable_indices = None
        if summarised_states is not None:
            summarised_state_indices = self.system_interface.state_indices(
                    summarised_states)
        else:
            summarised_state_indices = None
        if summarised_observables is not None:
            summarised_observable_indices = self.system_interface.observable_indices(
                    summarised_observables)
        else:
            summarised_observable_indices = None

        return (saved_state_indices,
                saved_observable_indices,
                summarised_state_indices,
                summarised_observable_indices)

    def solve(self,
              initial_values,
              parameters,
              forcing_vectors=None,
              duration=1.0,
              settling_time=0.0,
              blocksize=256,
              stream=None,
              chunk_axis='run',
              grid_type: str = 'combinatorial',
              results_type: str = 'full',
              **kwargs):
        """Solve a batch IVP problem using the Solver's settings and the
        provided inputs."""
        if kwargs:
            self.update(kwargs)

        inits, params = self.grid_builder(states=initial_values,
                                          params=parameters,
                                          kind=grid_type)
        self.kernel.run(
            inits=inits,
            params=params,
            forcing_vectors=forcing_vectors,
            duration=duration,
            warmup=settling_time,
            blocksize=blocksize,
            stream=stream,
            chunk_axis=chunk_axis,
        )

        return SolveResult.from_solver(self, results_type=results_type)

    def update(self, updates_dict, silent=False, **kwargs):
        """Update settings in this and all children of this solver.

        Any settings in the integrator or system can be modified through
        this function, including those that are not exposed directly.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        updates_dict = self.update_saved_variables(updates_dict)

        recognised = set()
        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.update_memory_settings(updates_dict, silent=True)
        all_unrecognized -= self.system_interface.update(updates_dict, silent=True)
        all_unrecognized -= self.kernel.update(updates_dict, silent=True)

        if "profileCUDA" in updates_dict:
            if updates_dict["profileCUDA"]:
                self.enable_profiling()
            else:
                self.disable_profiling()
            recognised.add("profileCUDA")

        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    def update_saved_variables(self, updates_dict):
        """Interprets list of labels or indices and returns updates dict"""
        saved_states = updates_dict.pop("saved_states", None)
        saved_observables = updates_dict.pop("saved_observables", None)
        summarised_states = updates_dict.pop("summarised_states", None)
        summarised_observables = updates_dict.pop("summarised_observables",
                                                  None)

        (saved_state_indices,
         saved_observable_indices,
         summarised_state_indices,
         summarised_observable_indices) = self._variable_indices_from_list(
            saved_states,
            saved_observables,
            summarised_states,
            summarised_observables
        )

        if saved_state_indices is not None:
            updates_dict[saved_state_indices] = saved_state_indices
        if saved_observable_indices is not None:
            updates_dict[saved_observable_indices] = saved_observable_indices
        if summarised_state_indices is not None:
            updates_dict[summarised_state_indices] = summarised_state_indices
        if summarised_observable_indices is not None:
            updates_dict[summarised_observable_indices] = \
                summarised_observable_indices

        return updates_dict

    def update_memory_settings(self,
                               updates_dict=None,
                               silent=False,
                               **kwargs):
        """
        Update memory manager parameters. Possible keys:
        - mem_proportion: float, the proportion of the GPU VRAM to set aside for
        the solver
        - allocator: str, the allocator to use for memory allocation.
        """
        if updates_dict is None:

            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()
        all_unrecognized = set(updates_dict.keys())
        recognised = set()

        if "mem_proportion" in updates_dict:
            self.memory_manager.set_manual_proportion(
                    updates_dict["mem_proportion"])
            recognised.add("mem_proportion")
        if "allocator" in updates_dict:
            self.memory_manager.set_allocator(updates_dict["allocator"])
            recognised.add("allocator")

        recognised = set(recognised)
        all_unrecognized -= set(recognised)
        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver. This will allow you to profile
        the performance of the solver on the
        GPU, but will slow things down.
        """
        # Consider disabling optimisation and enabling debug and line info
        # for profiling
        self.kernel.enable_profiling()

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver. This will stop profiling the
        performance of the solver on the GPU,
        but will speed things up.
        """
        self.kernel.disable_profiling()

    def get_state_indices(self,
                          state_labels: Optional[List[str]] = None):
        """
        Get the indices of the states in the system based on the provided
        state labels.
        If no labels are provided, returns all state indices.
        """
        return self.system_interface.get_state_indices(state_labels)

    def get_observable_indices(self,
                               observable_labels: Optional[List[str]] = None):
        """
        Get the indices of the observables in the system based on the provided
        observable labels.
        If no labels are provided, returns all observable indices.
        """
        return self.system_interface.get_observable_indices(observable_labels)

    @property
    def precision(self) -> type:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.precision` from
        the child BatchSolverKernel object."""
        return self.kernel.precision

    @property
    def system_sizes(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.system_sizes`
        from the child BatchSolverKernel object."""
        return self.kernel.system_sizes

    @property
    def output_array_heights(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.output_array_heights`
        from the child BatchSolverKernel object.
        """
        return self.kernel.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .summaries_buffer_sizes` from the child BatchSolverKernel object."""
        return self.kernel.summaries_buffer_sizes

    @property
    def num_runs(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.num_runs` from
        the child BatchSolverKernel object."""
        return self.kernel.num_runs

    @property
    def output_length(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.output_length`
        from the child BatchSolverKernel object."""
        return self.kernel.output_length

    @property
    def summaries_length(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.summaries_length`
        from the child BatchSolverKernel object."""
        return self.kernel.summaries_length

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .summary_legend_per_variable` from the child BatchSolverKernel
        object."""
        return self.kernel.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .saved_state_indices` from the child BatchSolverKernel object."""
        return self.kernel.saved_state_indices

    @property
    def saved_states(self):
        """Returns a list of state labels for the saved states."""
        return self.system_interface.state_labels(
                self.saved_state_indices)

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .saved_observable_indices` from the child BatchSolverKernel object."""
        return self.kernel.saved_observable_indices

    @property
    def saved_observables(self):
        """Returns a list of observable labels for the saved observables."""
        return self.system_interface.observable_labels(
                self.saved_observable_indices)
    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .summarised_state_indices` from the child BatchSolverKernel object."""
        return self.kernel.summarised_state_indices

    @property
    def summarised_states(self):
        """Returns a list of state labels for the summarised states."""
        return self.system_interface.state_labels(
                self.summarised_state_indices)

    @property
    def summarised_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .summarised_observable_indices` from the child BatchSolverKernel
        object."""
        return self.kernel.summarised_observable_indices

    @property
    def summarised_observables(self):
        """Returns a list of observable labels for the summarised observables."""
        return self.system_interface.observable_labels(
                self.summarised_observable_indices)

    @property
    def active_output_arrays(self) -> ActiveOutputs:
        """Exposes
        :attr:`~cubie.batchsolving.BatchSolverKernel.active_output_arrays` from
        the child BatchSolverKernel object."""
        return self.kernel.active_output_arrays

    @property
    def state(self):
        """Exposes :attr:~cubie.batchsolving.BatchSolverKernel.state from the
        child BatchSolverKernel object."""
        return self.kernel.state
    @property
    def observables(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.observables`
        from the child BatchSolverKernel object."""
        return self.kernel.observables

    @property
    def state_summaries(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .state_summaries` from the child BatchSolverKernel object."""
        return self.kernel.state_summaries

    @property
    def observable_summaries(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.
        observable_summaries` from the child BatchSolverKernel object."""
        return self.kernel.observable_summaries

    @property
    def parameters(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.parameters`
        from the child BatchSolverKernel object."""
        return self.kernel.parameters

    @property
    def initial_values(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.initial_values`
        from the child BatchSolverKernel object."""
        return self.kernel.initial_values

    @property
    def forcing_vectors(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.forcing_vectors`
         from the child BatchSolverKernel object."""
        return self.kernel.forcing_vectors

    @property
    def save_time(self) -> bool:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.save_time` from
        the child BatchSolverKernel object."""
        return self.kernel.save_time

    @property
    def output_types(self) -> list[str]:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.output_types`
        from the child BatchSolverKernel object."""
        return self.kernel.output_types

    @property
    def output_stride_order(self) -> List[str]:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.output_stride_order
        ` from the child BatchSolverKernel object."""
        return self.kernel.output_stride_order

    @property
    def input_variables(self) -> List[str]:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.input_variables
        ` from the child BatchSolverKernel object."""
        return self.system_interface.all_input_labels

    @property
    def output_variables(self) -> List[str]:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .output_variables` from the child BatchSolverKernel object."""
        return self.system_interface.all_output_labels

    @property
    def chunk_axis(self) -> str:
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.chunk_axis`
        from the child BatchSolverKernel object."""
        return self.kernel.chunk_axis

    @property
    def chunks(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel.chunks` from the
        child BatchSolverKernel object."""
        return self.kernel.chunks

    @property
    def memory_manager(self):
        """Returns the memory manager the solver is registered with."""
        return self.kernel.memory_manager

    @property
    def stream_group(self):
        """Returns the stream_group the solver is in."""
        return self.kernel.stream_group

    @property
    def mem_proportion(self):
        """Returns the memory proportion the solver is assigned."""
        return self.kernel.mem_proportion

    @property
    def solve_info(self):
        """Returns a SolveSpec object with details of the solver run."""
        return SolveSpec(
                dt_min = self.kernel.dt_min,
                dt_max = self.kernel.dt_max,
                dt_save = self.kernel.dt_save,
                dt_summarise = self.kernel.dt_summarise,
                duration = self.kernel.duration,
                warmup = self.kernel.warmup,
                atol = self.kernel.atol,
                rtol = self.kernel.rtol,
                algorithm = self.kernel.algorithm,
                saved_states = self.saved_states,
                saved_observables = self.saved_observables,
                summarised_states = self.summarised_states,
                summarised_observables = self.summarised_states,
                output_types = self.kernel.output_types,
                precision = self.precision
        )

