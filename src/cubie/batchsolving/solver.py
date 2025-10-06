"""High level batch-solver interface.

This module exposes the user-facing :class:`Solver` class and a convenience
wrapper :func:`solve_ivp` for solving batches of initial value problems on the
GPU.
"""

from typing import Dict, List, Optional, Union, Any

import numpy as np

from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solveresult import SolveResult, SolveSpec
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.memory import default_memmgr
from cubie.odesystems.baseODE import BaseODE
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie._utils import merge_component_settings


def solve_ivp(
    system,
    y0,
    parameters = None,
    drivers: Optional[Dict[str, object]] = None,
    dt_eval = 1e-2,
    method="euler",
    duration=1.0,
    settling_time=0.0,
    grid_type='combinatorial',
    **kwargs,
) -> SolveResult:
    """Solve a batch initial value problem.

    Parameters
    ----------

    system : object
        System model defining the differential equations.
    y0 : array-like
        Initial state values for each run.
    parameters : array-like or dict
        Parameter values for each run.
    drivers : dict[str, object], optional
        Driver configuration to interpolate during integration.
    dt_eval : float
        Interval at which solution values are stored.
    method : str, optional
        Integration algorithm to use. Default is ``"euler"``.
    duration : float, optional
        Total integration time. Default is ``1.0``.
    settling_time : float, optional
        Warm-up period prior to storing outputs. Default is ``0.0``.
    grid_type : str, optional
        'verbatim' or 'combinatorial' strategy for building the batch. Verbatim
        will pair each input value and parameter set so that index 0 of all
        inputs form one "run". 'combinatorial' will generate every combination
        of every input variable, which scales combinatorially.
    **kwargs
        Additional keyword arguments passed to :class:`Solver`.

    Returns
    -------
    SolveResult
        Results returned from :meth:`Solver.solve`.
    """
    solver = Solver(
        system,
        algorithm=method,
        dt_save=dt_eval,
        duration=duration,
        warmup=settling_time,
        **kwargs,
    )
    results = solver.solve(
        y0,
        parameters,
        drivers=drivers,
        duration=duration,
        warmup=settling_time,
        grid_type=grid_type,
        **kwargs,
    )
    return results


class Solver:
    """User-facing interface for solving batches of ODE systems.

    Parameters
    ----------
    system : BaseODE
        System model containing the ODEs to integrate.
    algorithm : str, optional
        Integration algorithm to use. Defaults to ``"euler"``.
    duration : float, optional
        Total integration time. Defaults to ``1.0``.
    warmup : float, optional
        Warm-up period before outputs are stored. Defaults to ``0.0``.
    dt_min, dt_max : float, optional
        Minimum and maximum timestep sizes. Defaults are ``0.01`` and ``0.1``.
    dt_save : float, optional
        Sampling interval for storing state values. Default is ``0.1``.
    dt_summarise : float, optional
        Interval for computing summary outputs. Default is ``1.0``.
    atol, rtol : float, optional
        Absolute and relative error tolerances. Defaults are ``1e-6``.
    saved_states, saved_observables : list of str or int, optional
        Variables to save verbatim at each output step.
    summarised_states, summarised_observables : list of str or int, optional
        Variables for which summary statistics are computed.
    output_types : list of str, optional
        Output arrays to generate (e.g. ``["state"]``).
    precision : type, optional
        Floating point precision. Defaults to ``numpy.float64``.
    profileCUDA : bool, optional
        Enable CUDA profiling. Defaults to ``False``.
    memory_manager : MemoryManager, optional
        Manager responsible for CUDA memory allocations.
    stream_group : str, optional
        Name of the CUDA stream group used by the solver. Defaults to
        ``"solver"``.
    mem_proportion : float, optional
        Proportion of GPU memory reserved for the solver.
    step_control_settings : dict[str, object], optional
        Explicit controller configuration that overrides solver defaults.
    **kwargs
        Additional keyword arguments forwarded to internal components.

    Attributes
    ----------
    system_interface : SystemInterface
        Object translating between user-world and GPU-world variables.
    grid_builder : BatchGridBuilder
        Helper that constructs integration grids from user inputs.
    kernel : BatchSolverKernel
        Low-level component that executes integrations on the GPU.
    """

    def __init__(
        self,
        system: BaseODE,
        algorithm: str = "euler",
        duration: float = 1.0,
        warmup: float = 0.0,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        saved_states: Optional[List[Union[str, int]]] = None,
        saved_observables: Optional[List[Union[str, int]]] = None,
        summarised_states: Optional[List[Union[str, int]]] = None,
        summarised_observables: Optional[List[Union[str, int]]] = None,
        output_types: list[str] = None,
        profileCUDA: bool = False,
        memory_manager=default_memmgr,
        stream_group="solver",
        mem_proportion=None,
        step_control_settings: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        super().__init__()
        precision = system.precision
        interface = SystemInterface.from_system(system)
        self.system_interface = interface
        self.driver_interpolator = ArrayInterpolator(
            precision=precision,
            input_dict={
                "placeholder": np.zeros(6, dtype=precision),
                "dt": 0.1,
            },
        )


        (
            saved_state_indices,
            saved_observable_indices,
            summarised_state_indices,
            summarised_observable_indices,
        ) = self._variable_indices_from_list(
            saved_states,
            saved_observables,
            summarised_states,
            summarised_observables,
        )

        self.grid_builder = BatchGridBuilder(interface)

        step_settings, recognized_kwargs = merge_component_settings(
                kwargs,
                step_control_settings,
                ALL_STEP_CONTROLLER_PARAMETERS,
        )

        self.kernel = BatchSolverKernel(
            system,
            algorithm=algorithm,
            duration=duration,
            warmup=warmup,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
            output_types=output_types,
            profileCUDA=profileCUDA,
            memory_manager=memory_manager,
            stream_group=stream_group,
            mem_proportion=mem_proportion,
            step_control_settings=step_control_settings,
        )

        if set(kwargs) - set(recognized_kwargs):
            raise KeyError(f"Unrecognized keyword arguments: "
                           f"{set(kwargs) - set(recognized_kwargs)}")

    def _variable_indices_from_list(
        self,
        saved_states,
        saved_observables,
        summarised_states,
        summarised_observables,
    ):
        """Translate variable labels into index arrays.

        Parameters
        ----------
        saved_states, saved_observables, summarised_states,
        summarised_observables : list[str] or None
            Lists of variable labels. ``None`` leaves the corresponding
            selection unchanged.

        Returns
        -------
        tuple of ndarray or None
            Index arrays for each variable list in the order provided.
        """

        def _get_indices(getter, items):
            return getter(items) if items is not None else None

        saved_state_indices = _get_indices(
            self.system_interface.state_indices, saved_states
        )
        saved_observable_indices = _get_indices(
            self.system_interface.observable_indices, saved_observables
        )
        summarised_state_indices = _get_indices(
            self.system_interface.state_indices, summarised_states
        )
        summarised_observable_indices = _get_indices(
            self.system_interface.observable_indices, summarised_observables
        )

        return (
            saved_state_indices,
            saved_observable_indices,
            summarised_state_indices,
            summarised_observable_indices,
        )

    def solve(
        self,
        initial_values,
        parameters,
        drivers: Optional[Dict[str, Any]] = None,
        duration=1.0,
        settling_time=0.0,
        blocksize=256,
        stream=None,
        chunk_axis="run",
        grid_type: str = "combinatorial",
        results_type: str = "full",
        **kwargs,
    ):
        """Solve a batch initial value problem.

        Parameters
        ----------
        initial_values : array-like or dict
            Initial state values for each integration run.
        parameters : array-like or dict
            Parameter values for each run.
        drivers : dict[str, object], optional
            Driver samples or configuration matching
            :class:`cubie.integrators.array_interpolator.ArrayInterpolator`.
        duration : float, optional
            Total integration time. Default is ``1.0``.
        settling_time : float, optional
            Warm-up period before recording outputs. Default ``0.0``.
        blocksize : int, optional
            CUDA block size used for kernel launch. Default ``256``.
        stream : cuda.stream or None, optional
            Stream on which to execute the kernel. ``None`` uses the solver's
            default stream.
        chunk_axis : {'run', 'time'}, optional
            Dimension along which to chunk when memory is limited. Default
            ``'run'``.
        grid_type : {'combinatorial', 'verbatim'}, optional
            Strategy for constructing the integration grid from inputs.
        results_type : {'full', 'numpy', 'numpy_per_summary', 'pandas'}, optional
            Format of returned results. Default ``'full'``.
        **kwargs
            Additional options forwarded to :meth:`update`.

        Returns
        -------
        SolveResult
            Collected results from the integration run.
        """
        if kwargs:
            self.update(kwargs)

        inits, params = self.grid_builder(
            states=initial_values, params=parameters, kind=grid_type
        )

        if drivers is not None:
            ArrayInterpolator.check_against_system_drivers(drivers,
                                                               self.system)
            fn_changed = self.driver_interpolator.update_from_dict(drivers)
        if fn_changed:
            self.update({"driver_function": (
                self.driver_interpolator.device_function)})

        self.kernel.run(
            inits=inits,
            params=params,
            driver_coefficients=self.driver_interpolator.coefficients,
            duration=duration,
            warmup=settling_time,
            blocksize=blocksize,
            stream=stream,
            chunk_axis=chunk_axis,
        )

        return SolveResult.from_solver(self, results_type=results_type)

    def update(self, updates_dict, silent=False, **kwargs):
        """Update solver, integrator and system settings.

        Parameters
        ----------
        updates_dict : dict
            Mapping of attribute names to new values.
        silent : bool, optional
            If ``True`` unknown keys are ignored instead of raising
            ``KeyError``. Defaults to ``False``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        set
            Set of keys that were successfully updated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        updates_dict = self.update_saved_variables(updates_dict)


        driver_recognised = self.driver_interpolator.update(updates_dict,
                                                      silent=True)
        if driver_recognised:
            updates_dict["driver_function"] = (
                self.driver_interpolator.evaluation_function)

        recognised = set()
        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= driver_recognised
        all_unrecognized -= self.update_memory_settings(
            updates_dict, silent=True
        )
        all_unrecognized -= self.system_interface.update(
            updates_dict, silent=True
        )
        all_unrecognized -= self.kernel.update(updates_dict, silent=True)

        if "profileCUDA" in updates_dict: # pragma: no cover
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
        """Interpret label lists and insert resolved indices.

        Parameters
        ----------
        updates_dict : dict
            Dictionary potentially containing ``saved_states``,
            ``saved_observables``, ``summarised_states`` or
            ``summarised_observables`` entries as labels.

        Returns
        -------
        dict
            Updated dictionary with label lists replaced by index arrays.
        """
        labels = [
            ("saved_states", "saved_state_indices"),
            ("saved_observables", "saved_observable_indices"),
            ("summarised_states", "summarised_state_indices"),
            ("summarised_observables", "summarised_observable_indices"),
        ]

        label_values = [updates_dict.pop(label, None) for label, _ in labels]

        indices = self._variable_indices_from_list(*label_values)

        for (_, key), value in zip(labels, indices):
            if value is not None:
                updates_dict[key] = value

        return updates_dict

    def update_memory_settings(
        self, updates_dict=None, silent=False, **kwargs
    ):
        """Update memory manager parameters.

        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of settings to update.
        silent : bool, optional
            If ``True`` unknown keys are ignored. Default ``False``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        set
            Set of keys that were successfully updated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()
        all_unrecognized = set(updates_dict.keys())
        recognised = set()

        if "mem_proportion" in updates_dict:
            self.memory_manager.set_manual_proportion(
                self.kernel, updates_dict["mem_proportion"]
            )
            recognised.add("mem_proportion")
        if "allocator" in updates_dict:
            self.memory_manager.set_allocator(
                self.kernel, updates_dict["allocator"]
            )
            recognised.add("allocator")

        recognised = set(recognised)
        all_unrecognized -= set(recognised)
        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    def enable_profiling(self):
        """Enable CUDA profiling for the solver.

        Profiling collects detailed performance information but can slow down
        execution considerably.
        """
        # Consider disabling optimisation and enabling debug and line info
        # for profiling
        self.kernel.enable_profiling()

    def disable_profiling(self):
        """Disable CUDA profiling for the solver.

        This reverts the effect of :meth:`enable_profiling` and returns the
        solver to normal execution speed.
        """
        self.kernel.disable_profiling()

    def get_state_indices(self, state_labels: Optional[List[str]] = None):
        """Return indices for the specified state variables.

        Parameters
        ----------
        state_labels : list[str], optional
            Labels of states to query. ``None`` returns indices for all states.

        Returns
        -------
        ndarray
            Integer indices corresponding to the requested states.
        """
        return self.system_interface.state_indices(state_labels)

    def get_observable_indices(
        self, observable_labels: Optional[List[str]] = None
    ):
        """Return indices for the specified observables.

        Parameters
        ----------
        observable_labels : list[str], optional
            Labels of observables to query. ``None`` returns indices for all
            observables.

        Returns
        -------
        ndarray
            Integer indices corresponding to the requested observables.
        """
        return self.system_interface.observable_indices(observable_labels)

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
        return self.system_interface.state_labels(self.saved_state_indices)

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .saved_observable_indices` from the child BatchSolverKernel object."""
        return self.kernel.saved_observable_indices

    @property
    def saved_observables(self):
        """Returns a list of observable labels for the saved observables."""
        return self.system_interface.observable_labels(
            self.saved_observable_indices
        )

    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~cubie.batchsolving.BatchSolverKernel
        .summarised_state_indices` from the child BatchSolverKernel object."""
        return self.kernel.summarised_state_indices

    @property
    def summarised_states(self):
        """Returns a list of state labels for the summarised states."""
        return self.system_interface.state_labels(
            self.summarised_state_indices
        )

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
            self.summarised_observable_indices
        )

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
    def driver_coefficients(self):
        """Expose driver coefficients staged on the BatchSolverKernel."""
        return self.kernel.driver_coefficients

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
    def system(self) -> "BaseODE":
        """Returns the system the solver is associated with."""
        return self.kernel.system

    @property
    def solve_info(self):
        """Returns a SolveSpec object with details of the solver run."""
        return SolveSpec(
            dt_min=self.kernel.dt_min,
            dt_max=self.kernel.dt_max,
            dt_save=self.kernel.dt_save,
            dt_summarise=self.kernel.dt_summarise,
            duration=self.kernel.duration,
            warmup=self.kernel.warmup,
            atol=self.kernel.atol,
            rtol=self.kernel.rtol,
            algorithm=self.kernel.algorithm,
            saved_states=self.saved_states,
            saved_observables=self.saved_observables,
            summarised_states=self.summarised_states,
            summarised_observables=self.summarised_observables,
            output_types=self.kernel.output_types,
            precision=self.precision,
        )
