"""High level batch-solver interface.

This module exposes the user-facing :class:`Solver` class and a convenience
wrapper :func:`solve_ivp` for solving batches of initial value problems on the
GPU.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from cubie._utils import PrecisionDType
from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solveresult import SolveResult, SolveSpec
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.memory.mem_manager import ALL_MEMORY_MANAGER_PARAMETERS
from cubie.odesystems.baseODE import BaseODE
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie._utils import merge_kwargs_into_settings
from cubie.outputhandling.output_functions import (
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)


def solve_ivp(
    system: BaseODE,
    y0: Union[np.ndarray, Dict[str, np.ndarray]],
    parameters: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    drivers: Optional[Dict[str, object]] = None,
    dt_save: Optional[float] = None,
    method: str = "euler",
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    grid_type: str = "combinatorial",
    **kwargs: Any,
) -> SolveResult:
    """Solve a batch initial value problem.

    Parameters
    ----------
    system
        System model defining the differential equations.
    y0
        Initial state values for each run as arrays or dictionaries mapping
        labels to arrays.
    parameters
        Parameter values for each run as arrays or dictionaries mapping labels
        to arrays.
    drivers
        Driver configuration to interpolate during integration.
    dt_save
        Interval at which solution values are stored.
    method
        Integration algorithm to use. Default is ``"euler"``.
    duration
        Total integration time. Default is ``1.0``.
    settling_time
        Warm-up period prior to storing outputs. Default is ``0.0``.
    t0
        Initial integration time supplied to the solver. Default is ``0.0``.
    grid_type
        ``"verbatim"`` pairs each input vector while ``"combinatorial"``
        produces every combination of provided values.
    **kwargs
        Additional keyword arguments passed to :class:`Solver`.

    Returns
    -------
    SolveResult
        Results returned from :meth:`Solver.solve`.
    """
    loop_settings = kwargs.pop("loop_settings", None)
    if dt_save is not None:
        kwargs.setdefault("dt_save", dt_save)

    solver = Solver(
        system,
        algorithm=method,
        loop_settings=loop_settings,
        **kwargs,
    )
    results = solver.solve(
        y0,
        parameters,
        drivers=drivers,
        duration=duration,
        warmup=settling_time,
        t0=t0,
        grid_type=grid_type,
        **kwargs,
    )
    return results


class Solver:
    """User-facing interface for solving batches of ODE systems.

    Parameters
    ----------
    system
        System model containing the ODEs to integrate.
    algorithm
        Integration algorithm to use. Defaults to ``"euler"``.
    profileCUDA
        Enable CUDA profiling. Defaults to ``False``.
    step_control_settings
        Explicit controller configuration that overrides solver defaults.
    algorithm_settings
        Explicit algorithm configuration overriding solver defaults.
    output_settings
        Explicit output configuration overriding solver defaults. Individual
        selectors such as ``saved_states`` may also be supplied as keyword
        arguments.
    memory_settings
        Explicit memory configuration overriding solver defaults. Keys like
        ``memory_manager`` or ``mem_proportion`` may likewise be provided as
        keyword arguments.
    loop_settings
        Explicit loop configuration overriding solver defaults. Keys such as
        ``dt_save`` and ``dt_summarise`` may also be supplied as loose keyword
        arguments.
    strict
        If ``True`` unknown keyword arguments raise ``KeyError``.
    **kwargs
        Additional keyword arguments forwarded to internal components.

    Notes
    -----
    Instances coordinate batch grid construction, kernel configuration, and
    driver interpolation so that :meth:`solve` orchestrates a complete GPU
    integration run.
    """

    def __init__(
        self,
        system: BaseODE,
        algorithm: str = "euler",
        profileCUDA: bool = False,
        step_control_settings: Optional[Dict[str, object]] = None,
        algorithm_settings: Optional[Dict[str, object]] = None,
        output_settings: Optional[Dict[str, object]] = None,
        memory_settings: Optional[Dict[str, object]] = None,
        loop_settings: Optional[Dict[str, object]] = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        if output_settings is None:
            output_settings = {}
        if memory_settings is None:
            memory_settings = {}
        if step_control_settings is None:
            step_control_settings = {}
        if algorithm_settings is None:
            algorithm_settings = {}
        if loop_settings is None:
            loop_settings = {}

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

        self.grid_builder = BatchGridBuilder(interface)

        recognized_kwargs: set[str] = set()

        output_settings, output_recognized = merge_kwargs_into_settings(
            kwargs=kwargs, valid_keys=ALL_OUTPUT_FUNCTION_PARAMETERS,
            user_settings=output_settings)
        self.convert_output_labels(output_settings)

        memory_settings, memory_recognized = merge_kwargs_into_settings(
            kwargs=kwargs, valid_keys=ALL_MEMORY_MANAGER_PARAMETERS,
            user_settings=memory_settings)

        step_settings, step_recognized = merge_kwargs_into_settings(
            kwargs=kwargs, valid_keys=ALL_STEP_CONTROLLER_PARAMETERS,
            user_settings=step_control_settings)
        algorithm_settings, algorithm_recognized = merge_kwargs_into_settings(
            kwargs=kwargs, valid_keys=ALL_ALGORITHM_STEP_PARAMETERS,
            user_settings=algorithm_settings)
        algorithm_settings["algorithm"] = algorithm
        loop_settings, loop_recognized = merge_kwargs_into_settings(
            kwargs=kwargs, valid_keys=ALL_LOOP_SETTINGS,
            user_settings=loop_settings)
        recognized_kwargs = (step_recognized | algorithm_recognized
                             | output_recognized | memory_recognized
                             | loop_recognized)

        self.kernel = BatchSolverKernel(
            system,
            loop_settings=loop_settings,
            profileCUDA=profileCUDA,
            step_control_settings=step_settings,
            algorithm_settings=algorithm_settings,
            output_settings=output_settings,
            memory_settings=memory_settings,
        )

        if strict:
            if set(kwargs) - recognized_kwargs:
                raise KeyError(
                    "Unrecognized keyword arguments: "
                    f"{set(kwargs) - recognized_kwargs}"
                )

    def convert_output_labels(
        self,
        output_settings: Dict[str, Any],
    ) -> None:
        """Resolve output label settings in-place.

        Parameters
        ----------
        output_settings
            Mapping of output configuration keys recognised by the solver.
            Entries describing saved or summarised selectors are replaced with
            integer indices when provided.

        Returns
        -------
        None
            This method mutates ``output_settings`` in-place.

        Raises
        ------
        ValueError
            If the settings dict contains duplicate entries, for example both
            ``"saved_states"`` and ``"saved_state_indices"``.

        Notes
        -----
        Users may supply selectors as labels or integers; this resolver ensures
        that downstream components receive numeric indices and canonical keys.
        """

        resolvers = {
            "saved_states": self.system_interface.state_indices,
            "saved_state_indices": self.system_interface.state_indices,
            "summarised_states": self.system_interface.state_indices,
            "summarised_state_indices": self.system_interface.state_indices,
            "saved_observables": self.system_interface.observable_indices,
            "saved_observable_indices": (
                self.system_interface.observable_indices
            ),
            "summarised_observables": self.system_interface.observable_indices,
            "summarised_observable_indices": (
                self.system_interface.observable_indices
            ),
        }

        labels2index_keys = {
            "saved_states": "saved_state_indices",
            "saved_observables": "saved_observable_indices",
            "summarised_states": "summarised_state_indices",
            "summarised_observable_indices": (
                "summarised_observable_indices"
            ),
        }
        # Replace any labels with integer indices
        for key, resolver in resolvers.items():
            values = output_settings.get(key)
            if values is not None:
                output_settings[key] = resolver(values)

        # Replace names for a list of labels, e.g. saved_states, with the
        # indices key that outputfunctions expects
        for inkey, outkey in labels2index_keys.items():
            indices = output_settings.pop(inkey, None)
            if indices is not None:
                if output_settings.get(outkey, None) is not None:
                    raise ValueError(
                        "Duplicate output settings provided: got "
                        f"{inkey}={output_settings[inkey]} and "
                        f"{outkey} = {output_settings[outkey]}"
                    )
                output_settings[outkey] = indices

    def solve(
        self,
        initial_values: Union[np.ndarray, Dict[str, np.ndarray]],
        parameters: Union[np.ndarray, Dict[str, np.ndarray]],
        drivers: Optional[Dict[str, Any]] = None,
        duration: float = 1.0,
        settling_time: float = 0.0,
        t0: float = 0.0,
        blocksize: int = 256,
        stream: Any = None,
        chunk_axis: str = "run",
        grid_type: str = "combinatorial",
        results_type: str = "full",
        **kwargs: Any,
    ) -> SolveResult:
        """Solve a batch initial value problem.

        Parameters
        ----------
        initial_values
            Initial state values for each integration run.
        parameters
            Parameter values for each run.
        drivers
            Driver samples or configuration matching
            :class:`cubie.integrators.array_interpolator.ArrayInterpolator`.
        duration
            Total integration time. Default is ``1.0``.
        settling_time
            Warm-up period before recording outputs. Default ``0.0``.
        t0
            Initial integration time. Default ``0.0``.
        blocksize
            CUDA block size used for kernel launch. Default ``256``.
        stream
            Stream on which to execute the kernel. ``None`` uses the solver's
            default stream.
        chunk_axis
            Dimension along which to chunk when memory is limited. Default is
            ``"run"``.
        grid_type
            Strategy for constructing the integration grid from inputs.
        results_type
            Format of returned results, for example ``"full"`` or ``"numpy"``.
        **kwargs
            Additional options forwarded to :meth:`update`.

        Returns
        -------
        SolveResult
            Collected results from the integration run.
        """
        if kwargs:
            self.update(kwargs, silent=True)

        inits, params = self.grid_builder(
            states=initial_values, params=parameters, kind=grid_type
        )

        fn_changed = False  # ensure defined if drivers is None
        if drivers is not None:
            ArrayInterpolator.check_against_system_drivers(
                drivers, self.system
            )
            fn_changed = self.driver_interpolator.update_from_dict(drivers)
        if fn_changed:
            self.update(
                {"driver_function": self.driver_interpolator.evaluation_function,
                 "driver_del_t": self.driver_interpolator.driver_del_t}
            )

        self.kernel.run(
            inits=inits,
            params=params,
            driver_coefficients=self.driver_interpolator.coefficients,
            duration=duration,
            warmup=settling_time,
            t0=t0,
            blocksize=blocksize,
            stream=stream,
            chunk_axis=chunk_axis,
        )

        return SolveResult.from_solver(self, results_type=results_type)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update solver, integrator, and system settings.

        Parameters
        ----------
        updates_dict
            Mapping of attribute names to new values.
        silent
            If ``True`` unknown keys are ignored instead of raising
            ``KeyError``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        Set[str]
            Set of keys that were successfully updated.

        Raises
        ------
        KeyError
            If ``silent`` is ``False`` and unknown settings are supplied.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        self.convert_output_labels(updates_dict)

        driver_recognised = self.driver_interpolator.update(
            updates_dict, silent=True
        )
        if driver_recognised:
            updates_dict["driver_function"] = (
                self.driver_interpolator.evaluation_function
            )
            updates_dict["driver_del_t"] = (
                self.driver_interpolator.driver_del_t
            )


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

        if "profileCUDA" in updates_dict:  # pragma: no cover
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

    def update_memory_settings(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update memory manager parameters.

        Parameters
        ----------
        updates_dict
            Mapping of memory manager settings to update.
        silent
            If ``True`` unknown keys are ignored instead of raising
            ``KeyError``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        Set[str]
            Set of keys that were successfully updated.

        Raises
        ------
        KeyError
            If ``silent`` is ``False`` and unknown settings are supplied.
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

    def enable_profiling(self) -> None:
        """Enable CUDA profiling for the solver.

        Returns
        -------
        None
            This method alters kernel profiling configuration in-place.
        """
        # Consider disabling optimisation and enabling debug and line info
        # for profiling
        self.kernel.enable_profiling()

    def disable_profiling(self) -> None:
        """Disable CUDA profiling for the solver.

        Returns
        -------
        None
            This method alters kernel profiling configuration in-place.
        """
        self.kernel.disable_profiling()

    def get_state_indices(
        self, state_labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return indices for the specified state variables.

        Parameters
        ----------
        state_labels
            Labels of states to query. ``None`` returns indices for all states.

        Returns
        -------
        np.ndarray
            Integer indices corresponding to the requested states.
        """
        return self.system_interface.state_indices(state_labels)

    def get_observable_indices(
        self, observable_labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return indices for the specified observables.

        Parameters
        ----------
        observable_labels
            Labels of observables to query. ``None`` returns indices for all
            observables.

        Returns
        -------
        np.ndarray
            Integer indices corresponding to the requested observables.
        """
        return self.system_interface.observable_indices(observable_labels)

    @property
    def precision(self) -> PrecisionDType:
        """Expose the kernel precision."""
        return self.kernel.precision

    @property
    def system_sizes(self):
        """Expose cached system size metadata."""
        return self.kernel.system_sizes

    @property
    def output_array_heights(self):
        """Expose output array heights from the kernel."""
        return self.kernel.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """Expose summary buffer sizes."""
        return self.kernel.summaries_buffer_sizes

    @property
    def num_runs(self):
        """Expose the number of runs in the last solve."""
        return self.kernel.num_runs

    @property
    def output_length(self):
        """Expose the flattened output length."""
        return self.kernel.output_length

    @property
    def summaries_length(self):
        """Expose the flattened summary length."""
        return self.kernel.summaries_length

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Expose summary legends keyed by variable index."""
        return self.kernel.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Expose saved state indices."""
        return self.kernel.saved_state_indices

    @property
    def saved_states(self):
        """List saved state labels."""
        return self.system_interface.state_labels(self.saved_state_indices)

    @property
    def saved_observable_indices(self):
        """Expose saved observable indices."""
        return self.kernel.saved_observable_indices

    @property
    def saved_observables(self):
        """List saved observable labels."""
        return self.system_interface.observable_labels(
            self.saved_observable_indices
        )

    @property
    def summarised_state_indices(self):
        """Expose summarised state indices."""
        return self.kernel.summarised_state_indices

    @property
    def summarised_states(self):
        """List summarised state labels."""
        return self.system_interface.state_labels(
            self.summarised_state_indices
        )

    @property
    def summarised_observable_indices(self):
        """Expose summarised observable indices."""
        return self.kernel.summarised_observable_indices

    @property
    def summarised_observables(self):
        """List summarised observable labels."""
        return self.system_interface.observable_labels(
            self.summarised_observable_indices
        )

    @property
    def active_output_arrays(self) -> ActiveOutputs:
        """Expose active output array containers."""
        return self.kernel.active_output_arrays

    @property
    def state(self):
        """Expose latest state outputs."""
        return self.kernel.state

    @property
    def observables(self):
        """Expose latest observable outputs."""
        return self.kernel.observables

    @property
    def state_summaries(self):
        """Expose state summary outputs."""
        return self.kernel.state_summaries

    @property
    def observable_summaries(self):
        """Expose observable summary outputs."""
        return self.kernel.observable_summaries

    @property
    def parameters(self):
        """Expose parameter array used in the last run."""
        return self.kernel.parameters

    @property
    def initial_values(self):
        """Expose initial values array used in the last run."""
        return self.kernel.initial_values

    @property
    def driver_coefficients(self):
        """Expose driver interpolation coefficients."""
        return self.kernel.driver_coefficients

    @property
    def save_time(self) -> bool:
        """Return whether time points are saved."""
        return self.kernel.save_time

    @property
    def output_types(self) -> List[str]:
        """List active output types."""
        return self.kernel.output_types

    @property
    def state_stride_order(self) -> Tuple[str, ...]:
        """Describe the stride order of state arrays."""
        return self.kernel.state_stride_order

    @property
    def input_variables(self) -> List[str]:
        """List all input variable labels."""
        return self.system_interface.all_input_labels

    @property
    def output_variables(self) -> List[str]:
        """List all output variable labels."""
        return self.system_interface.all_output_labels

    @property
    def chunk_axis(self) -> str:
        """Return the axis used for chunking large runs."""
        return self.kernel.chunk_axis

    @property
    def chunks(self):
        """Return the number of chunks used in the last run."""
        return self.kernel.chunks

    @property
    def memory_manager(self):
        """Return the associated memory manager instance."""
        return self.kernel.memory_manager

    @property
    def stream_group(self):
        """Return the CUDA stream group assigned to this solver."""
        return self.kernel.stream_group

    @property
    def mem_proportion(self):
        """Return the proportion of global memory allocated."""
        return self.kernel.mem_proportion

    @property
    def system(self) -> "BaseODE":
        """Return the underlying ODE system instance."""
        return self.kernel.system

    # Pass-through properties for solve_info components
    @property
    def dt(self) -> Optional[float]:
        """Return the fixed-step size or ``None`` for adaptive controllers."""
        return self.kernel.dt

    @property
    def dt_min(self) -> Optional[float]:
        """Return the minimum step size for adaptive controllers."""
        return self.kernel.dt_min

    @property
    def dt_max(self) -> Optional[float]:
        """Return the maximum step size for adaptive controllers."""
        return self.kernel.dt_max

    @property
    def dt_save(self):
        """Return the interval between saved outputs."""
        return self.kernel.dt_save

    @property
    def dt_summarise(self):
        """Return the interval between summary computations."""
        return self.kernel.dt_summarise

    @property
    def duration(self):
        """Return the requested integration duration."""
        return self.kernel.duration

    @property
    def warmup(self):
        """Return the warm-up period length."""
        return self.kernel.warmup

    @property
    def t0(self) -> float:
        """Return the starting integration time."""

        return self.kernel.t0

    @property
    def atol(self) -> Optional[float]:
        """Return the absolute tolerance for adaptive controllers."""
        return self.kernel.atol

    @property
    def rtol(self) -> Optional[float]:
        """Return the relative tolerance for adaptive controllers."""
        return self.kernel.rtol

    @property
    def algorithm(self):
        """Return the configured algorithm name."""
        return self.kernel.algorithm

    @property
    def solve_info(self) -> SolveSpec:
        """Construct a SolveSpec describing the current configuration."""
        return SolveSpec(
            dt=self.dt,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_save=self.dt_save,
            dt_summarise=self.dt_summarise,
            duration=self.duration,
            warmup=self.warmup,
            t0=self.t0,
            atol=self.atol,
            rtol=self.rtol,
            algorithm=self.algorithm,
            saved_states=self.saved_states,
            saved_observables=self.saved_observables,
            summarised_states=self.summarised_states,
            summarised_observables=self.summarised_observables,
            output_types=self.output_types,
            precision=self.precision,
        )
