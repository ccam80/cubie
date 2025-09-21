"""Single integrator run coordination for CUDA-based ODE solving.

This module provides the :class:`SingleIntegratorRun` class which
coordinates the modular integrator loop
(:class:`~cubie.integrators.loops.ode_loop.IVPLoop`) and its
dependencies. It performs dependency injection for the algorithm step,
step-size controller, and output handlers before exposing the compiled
device loop for execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, Callable

import numpy as np
from numpy.typing import ArrayLike

from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.integrators.algorithms_ import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.odesystems.ODEData import SystemSizes
from cubie.integrators.step_control import get_controller


class SingleIntegratorRun(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system : BaseODE
        The ODE system to integrate.
    algorithm : str, default="explicit_euler"
        Name of the algorithm step implementation.
    dt_min, dt_max : float, default (0.01, 0.1)
        Minimum and maximum step size targets forwarded to the controller.
    dt_save : float, default=0.1
        Cadence for saving state and observable outputs.
    dt_summarise : float, default=1.0
        Cadence for summary calculations.
    atol, rtol : float, default=1e-6
        Error tolerances for adaptive methods.
    saved_state_indices, saved_observable_indices : ArrayLike, optional
        Indices of states/observables to save during integration.
    summarised_state_indices, summarised_observable_indices :
        ArrayLike, optional
        Indices of states/observables to include in summary calculations.
    output_types : list[str], optional
        Output modes requested from the loop.
    step_controller_kind : str, optional
        Optional override for the controller type (default ``"fixed"``).
    algorithm_parameters : dict[str, Any], optional
        Additional keyword arguments forwarded to the algorithm step
        constructor.
    step_controller_parameters : dict[str, Any], optional
        Additional keyword arguments forwarded to the controller constructor.
    """

    def __init__(
        self,
        system,
        algorithm: str = "euler",
        dt_min: float = 0.01,
        dt_max: float = 0.1,
        fixed_step_size:float = 0.01,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        saved_state_indices: Optional[ArrayLike] = None,
        saved_observable_indices: Optional[ArrayLike] = None,
        summarised_state_indices: Optional[ArrayLike] = None,
        summarised_observable_indices: Optional[ArrayLike] = None,
        output_types: Optional[list[str]] = None,
        step_controller_kind: str = "fixed",
        algorithm_parameters: Optional[Dict[str, Any]] = None,
        step_controller_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:

        self._system = system
        system_sizes = system.sizes
        n = system_sizes.states

        self._output_functions = OutputFunctions(
            max_states=n,
            max_observables=system_sizes.observables,
            output_types=output_types,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
        )

        buffer_sizes = LoopBufferSizes.from_system_and_output_fns(
            self._system, self._output_functions
        )

        self.config = IntegratorRunSettings(
            algorithm=algorithm,
            step_controller_kind=step_controller_kind or "fixed",
            buffer_sizes=buffer_sizes
        )

        self._step_controller = self.instantiate_controller(
                step_controller_kind,
                dt_min=dt_min,
                dt_max=dt_max,
                fixed_step_size=fixed_step_size,
                atol=atol,
                rtol=rtol,
                **(step_controller_parameters or {}),
        )

        fixed = not self._step_controller.is_adaptive
        self._algo_step = self.instantiate_step_object(
                algorithm,
                n=n,
                fixed_step=fixed,
                dxdt_function=self._system.dxdt,
                solver_function_getter=self._system.get_solver_helper,
                step_size=fixed_step_size,
                **(algorithm_parameters or {}),
        )
        self._step_controller.update(algorithm_order=self._algo_step.order)

        self._integrator_instance: Optional[object] = None
        self._compiled_loop = None
        self._loop_cache_valid = False

    @property
    def loop_buffer_sizes(self):
        """
        Get buffer sizes required for the integration loop.

        Returns
        -------
        LoopBufferSizes
            Buffer size configuration for the integration loop.
        """
        return LoopBufferSizes.from_system_and_output_fns(
            self._system, self._output_functions
        )

    def check_compatibility(self):
        pass

    def instantiate_step_object(self,
                                kind: str = 'euler',
                                n: int = 1,
                                fixed_step: bool = False,
                                dxdt_function: Optional[Callable] = None,
                                solver_function_getter: Optional[Callable] = None,
                                step_size: float = 1e-3,
                                **kwargs):
        """Instantiate the algorithm step.

        Parameters
        ----------
        kind : str, default='euler'
            The algorithm to use.
        n : int, default=1
            System size; number of states
        fixed_step: bool, default=False
            True if the step controller used is fixed-step.
        dxdt_function : Device function
            The device function which calculates the derivative.
        solver_function_getter : Callable
            The :method:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE
            .get_solver_gelper` factory method which returns a nonlinear
            solver and observables function.
        step_size : float, default=1e-3
            Step-size for fixed-stepping algorithm

        kwargs
        ------
        Individual algorithm step parameters. These vary by algorithm. See:
            :class:`~cubie.integrators.algorithms_.explicit_euler
            .ExplicitEulerStep`,
            :class:`~cubie.integrators.algorithms_.backwards_euler
            .BackwardsEulerStep`,
            :class:`~cubie.integrators.algorithms_.crank_nicolson
            .CrankNicolsonStep`,
            :class:`~cubie.integrators.algorithms_
            .backwards_euler_predict_correct.BackwardsEulerPCStep`,

            """
        if fixed_step:
            kwargs["dt"] = step_size
        algorithm = get_algorithm_step(kind,
                                  precision=self.precision,
                                  n=n,
                                  dxdt_function=dxdt_function,
                                  solver_function_getter=solver_function_getter,
                                  **kwargs)
        return algorithm

    def instantiate_controller(self,
                               kind: str = 'fixed',
                               order: int = 1,
                               dt_min: float = 1e-3,
                               dt_max: float = 1e-1,
                               atol: Union[float, np.ndarray] = 1e-6,
                               rtol: Union[float, np.ndarray] = 1e-6,
                               fixed_step_size:float = 0.01,
                               **kwargs):
        """Instantiate the step controller.

        Parameters
        ----------
        kind : str, default='fixed'
            One of "fixed", "i", "pi", "pid", or "gustafsson". Selects the
            type of step controller to intantiate.
        order : int, default=1
            order of the algorithm method - sets controller characteristics.
        dt_min : float, optional, default=1e-3
            The minimum step size for an adaptive controller.
        dt_max : float, optional, default=1e-1
            The maximum step size for an adaptive controller.
        atol : float or np.ndarray, optional, default=1e-6
            Absolute tolerance - either a scalar for all values, or a vector
            of a tolerance for each state variable
        rtol : float or np.ndarray, optional, default=1e-6
            Relative tolerance - either a scalar for all values, or a vector
            of a tolerance for each state variable
        fixed_step_size : float, optional, default=0.01
            The fixed step size for a fixed controller.
        Kwargs
        ------
        kwargs : keyword arguments
            Additional parameters to pass to the controller constructor for
            advanced customisation. See:
            :class:`~.adaptive_I_controller.AdaptiveIController`,
            :class:`~.adaptive_PI_controller.AdaptivePIController`,
            :class:`~.adaptive_PID_controller.AdaptivePIDController`,
            :class:`~.gustafsson_controller.GustafssonController`, and
            :class:`~.fixed_step_controller.FixedStepController` for
            details.

            Returns:
            BaseStepController
                The step controller instance
        """
        if kind == 'fixed':
            controller = get_controller(kind,
                    precision=self.precision,
                    fixed_step_size=fixed_step_size)
        else:
            controller = get_controller(kind,
                    precision=self.precision,
                    algorithm_order=order,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    atol=atol,
                    rtol=rtol,
                    **kwargs)

        return controller

    def instantiate_loop(self,
                         buffer_sizes: LoopBufferSizes = None,
                         ):
        """Instantiate the integrator loop."""
        loop = IVPLoop(
            self.precision,
            self._step_controller,
            self._system,
            self._output_functions,
        )
        return loop

    @property
    def output_array_heights(self):
        """
        Get the heights of output arrays.

        Returns
        -------
        OutputArrayHeights
            Output array height configuration from the OutputFunctions object.
        """
        return self._output_functions.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """
        Get buffer sizes for summary calculations.

        Returns
        -------
        SummaryBufferSizes
            Summary buffer size configuration from the OutputFunctions object.
        """
        return self._output_functions.summaries_buffer_sizes

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Update parameters across all components.

        This method sends all parameters to all child components with
        silent=True to avoid spurious warnings, then checks if any parameters
        were not recognized by any component.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of parameters to update.
        silent : bool, default=False
            If True, suppress warnings about unrecognized parameters.
        **kwargs
            Parameter updates to apply as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.

        Raises
        ------
        KeyError
            If parameters are not recognized by any component and silent=False.

        Notes
        -----
        If the algorithm or step controller kind is updated, the respective
        child is reinstantiated with the new kind, and previous settings are
        provided via the child's update method. This means that any settings in
        the new child that aren't in the old child will be ignored. If there are
        settings included in the update dict, these will be set in the new
        child.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        recognized = set()

        if "algorithm" in updates_dict.keys():
            # If the algorithm is being updated, we need to reset the
            # integrator instance
            new_algo_key = updates_dict["algorithm"].lower()
            if new_algo_key != self.algorithm_key:
                old_settings = self._algo_step.settings_dict
                _algo_step = self.instantiate_step_object(
                    new_algo_key,
                    n=self.system_sizes.states,
                    fixed_step=not self._step_controller.is_adaptive,
                    dxdt_function=self._system.dxdt,
                    solver_function_getter=self._system.get_solver_helper,
                )
                _algo_step.update(old_settings, silent=True)
                self._algo_step = _algo_step
            recognized.add("algorithm")

        if "step_controller_kind" in updates_dict.keys():
            new_step_controller_kind = updates_dict["step_controller_kind"]
            if new_step_controller_kind != self.step_controller_kind:
                old_settings = self._step_controller.settings_dict
                _step_controller = self.instantiate_controller(
                        new_step_controller_kind,
                )
                _step_controller.update(old_settings, silent=True)
            recognized.add("step_controller_kind")


        recognized |= self._algo_step.update(updates_dict, silent=True)
        recognized |= self._step_controller.update(updates_dict, silent=True)
        recognized |= self._system.update(updates_dict, silent=True)
        recognized |= self._output_functions.update(updates_dict, silent=True)
        recognized |= self.update_compile_settings(updates_dict, silent=True)

        all_unrecognized -= recognized
        if all_unrecognized and not silent:
            raise KeyError(
                f"Unrecognized parameters: {all_unrecognized}"
            )
        if recognized:
            self._invalidate_cache()

        return recognized

    def build(self) -> Callable:
        """Instantiate the step controller, algorithm step, and loop."""
        #How to rebuild and pass in step_controller and algorithm_step?
        loop_fn = self._integrator_instance.device_function
        return loop_fn

    # ------------------------------------------------------------------
    # Buffer sizing and metadata
    # ------------------------------------------------------------------
    @property
    def loop_buffer_sizes(self) -> LoopBufferSizes:
        """Return buffer sizes required for the integration loop."""

        return self.compile_settings.buffer_sizes

    @property
    def output_array_heights(self):
        """Return output array heights from the output functions."""

        return self._output_functions.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """Return summary buffer sizes from the output functions."""

        return self._output_functions.summaries_buffer_sizes

    # ------------------------------------------------------------------
    # Resource requirements
    # ------------------------------------------------------------------
    @property
    def shared_memory_elements(self) -> int:
        """Return required shared-memory elements for the loop."""

        if not self.cache_valid:
            self.build()
        return self._integrator_instance.shared_memory_elements

    @property
    def shared_memory_bytes(self) -> int:
        """Return required shared-memory size in bytes."""

        datasize = self.precision(0.0).nbytes
        return int(self.shared_memory_elements * datasize)

    # ------------------------------------------------------------------
    # Convenience reach-through properties
    # ------------------------------------------------------------------
    @property
    def precision(self):
        """Return the numerical precision type for the system."""
        return self._system.precision

    @property
    def algorithm_key(self):
        """Return the algorithm key for the step algorithm."""
        return self.compile_settings.algorithm_key

    @property
    def step_controller_kind(self):
        """Return the step controller kind for the step algorithm."""
        return self.compile_settings.step_controller_kind

    @property
    def threads_per_loop(self) -> int:
        """Return the number of threads required for the step algorithm."""
        return self._algo_step.threads_per_step

    @property
    def dxdt_function(self):
        """Return the derivative function from the system."""

        return self._system.dxdt_function

    @property
    def save_state_func(self):
        """Return the state saving function."""

        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self):
        """Return the summary update function."""

        return self._output_functions.update_summaries_func

    @property
    def save_summaries_func(self):
        """Return the summary saving function."""

        return self._output_functions.save_summary_metrics_func

    @property
    def loop_step_config(self):
        """Return the timing configuration for the loop."""

        return self.config.loop_step_config

    @property
    def fixed_step_size(self):
        """Return the fixed step size if provided by the algorithm."""
        settings = getattr(self._algo_step, "compile_settings", None)
        if settings is not None and hasattr(settings, "dt"):
            return settings.fixed_step_size
        return None

    @property
    def dt_save(self) -> float:
        """Return the save interval."""

        return self.config.dt_save

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""

        return self.config.dt_summarise

    @property
    def system_sizes(self) -> SystemSizes:
        """Return the system size information."""

        return self._system.sizes

    @property
    def compile_flags(self):
        """Return compilation flags for output functions."""

        return self._output_functions.compile_flags

    @property
    def output_types(self):
        """Return configured output types."""

        return self._output_functions.output_types

    @property
    def summary_legend_per_variable(self):
        """Return the summary legend mapping."""

        return self._output_functions.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Return indices of states to save."""

        return self._output_functions.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Return indices of observables to save."""

        return self._output_functions.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Return indices of states included in summaries."""

        return self._output_functions.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Return indices of observables included in summaries."""

        return self._output_functions.summarised_observable_indices

    @property
    def save_time(self) -> bool:
        """Return whether the loop saves time values."""

        return self._output_functions.save_time

    @property
    def system(self):
        """Return the underlying ODE system."""
        return self._system

