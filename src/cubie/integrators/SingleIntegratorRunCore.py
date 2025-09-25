"""Single integrator run coordination for CUDA-based ODE solving.

This module provides the :class:`SingleIntegratorRunCore` class which
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
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.loops.ode_loop_config import LoopSharedIndices, \
    LoopLocalIndices
from cubie.outputhandling import OutputCompileFlags
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.integrators.step_control import get_controller


class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system : BaseODE
        The ODE system to integrate.
    algorithm : str, default="euler"
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
        super().__init__()
        config = IntegratorRunSettings(
            precision=system.precision,
            algorithm=algorithm,
            step_controller_kind=step_controller_kind or "fixed",
        )
        self.setup_compile_settings(config)
        self._system = system
        system_sizes = system.sizes

        self._output_functions = OutputFunctions(
            max_states=system_sizes.states,
            max_observables=system_sizes.observables,
            output_types=output_types,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
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
            n=system.sizes.states,
            fixed_step=fixed,
            dxdt_function=self._system.dxdt_function,
            get_solver_helper_fn=self._system.get_solver_helper,
            step_size=fixed_step_size,
            **(algorithm_parameters or {}),
        )

        if self._step_controller.is_adaptive:
            self._step_controller.update(algorithm_order=self._algo_step.order)

        self._loop = self.instantiate_loop(
                n_states=system_sizes.states,
                n_parameters=system_sizes.parameters,
                n_observables=system_sizes.observables,
                n_drivers=system_sizes.drivers,
                n_state_summaries=self._output_functions.n_summarised_states,
                n_observable_summaries=self._output_functions
                .n_summarised_observables,
                controller_local_elements=self._step_controller
                .local_memory_elements,
                algorithm_local_elements=self._algo_step
                .persistent_local_required,
                compile_flags=self._output_functions.compile_flags,
                dt_save=dt_save,
                dt_summarise=dt_summarise,
                dt0=self._step_controller.dt0,
                dt_min=self._step_controller.dt_min,
                dt_max=self._step_controller.dt_max,
                is_adaptive=self._step_controller.is_adaptive,
        )

    def check_compatibility(self):
        if (not self._algo_step.is_adaptive and
                self._step_controller.is_adaptive):
            raise ValueError("Adaptive step controller cannot be used with "
                             "fixed-step algorithm.")

    def instantiate_step_object(self,
                                kind: str = 'euler',
                                n: int = 1,
                                fixed_step: bool = False,
                                dxdt_function: Optional[Callable] = None,
                                get_solver_helper_fn: Optional[Callable]
                                = None,
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
        get_solver_helper_fn : Callable
            The :method:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE
            .get_solver_gelper` factory method which returns a nonlinear
            solver and observables function.
        step_size : float, default=1e-3
            Step-size for fixed-stepping algorithm

        kwargs
        ------
        Individual algorithm step parameters. These vary by algorithm. See:
            :class:`~cubie.integrators.algorithms.euler
            .ExplicitEulerStep`,
            :class:`~cubie.integrators.algorithms.backwards_euler
            .BackwardsEulerStep`,
            :class:`~cubie.integrators.algorithms.crank_nicolson
            .CrankNicolsonStep`,
            :class:`~cubie.integrators.algorithms
            .backwards_euler_predict_correct.BackwardsEulerPCStep`,

            """
        if kind.lower() in ["euler"]: # fixed step algorithms
            kwargs = {"dt": step_size}
        algorithm = get_algorithm_step(kind,
                                  precision=self.precision,
                                  n=n,
                                  dxdt_function=dxdt_function,
                                  get_solver_helper_fn=get_solver_helper_fn,
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
                    dt=fixed_step_size)
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
                         n_states: int,
                         n_parameters: int,
                         n_observables: int,
                         n_drivers: int,
                         n_state_summaries: int,
                         n_observable_summaries: int,
                         controller_local_elements: int,
                         algorithm_local_elements: int,
                         compile_flags: OutputCompileFlags,
                         dt_save: float,
                         dt_summarise: float,
                         dt0: float,
                         dt_min: float,
                         dt_max: float,
                         is_adaptive: bool,
                         ):
        """Instantiate the integrator loop."""
        shared_indices = LoopSharedIndices.from_sizes(
                n_states=n_states,
                n_observables=n_parameters,
                n_parameters=n_observables,
                n_drivers=n_drivers,
                n_state_summaries=n_state_summaries,
                n_observable_summaries=n_observable_summaries
        )
        local_indices = LoopLocalIndices.from_sizes(
                n_states=n_states,
                controller_len=controller_local_elements,
                algorithm_len=algorithm_local_elements
        )

        loop = IVPLoop(self.precision,
                       shared_indices,
                       local_indices,
                       compile_flags,
                       dt_save=dt_save,
                       dt_summarise=dt_summarise,
                       dt0=dt0,
                       dt_min=dt_min,
                       dt_max=dt_max,
                       is_adaptive=is_adaptive )
        return loop

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

        if "step_controller_kind" in updates_dict.keys():
            new_step_controller_kind = updates_dict["step_controller_kind"]
            if new_step_controller_kind != self.step_controller_kind:
                old_settings = self._step_controller.settings_dict
                _step_controller = self.instantiate_controller(
                        new_step_controller_kind,
                )
                _step_controller.update(old_settings, silent=True)
            recognized.add("step_controller_kind")

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
                    dxdt_function=self._system.dxdt_function,
                    get_solver_helper_fn=self._system.get_solver_helper,
                )
                _algo_step.update(old_settings, silent=True)
                self._algo_step = _algo_step
            recognized.add("algorithm")
            updates_dict.update({'algorithm_order':self._algo_step.order})

        output_recognized = self._output_functions.update(updates_dict,
                                                      silent=True)
        ctrl_recognized = self._step_controller.update(updates_dict,
                                                      silent=True)
        step_recognized = self._algo_step.update(updates_dict, silent=True)
        system_recognized = self._system.update(updates_dict, silent=True)
        loop_recognized = self._loop.update(updates_dict, silent=True)

        recognized |= output_recognized
        recognized |= ctrl_recognized
        recognized |= step_recognized
        recognized |= system_recognized
        recognized |= loop_recognized

        #Recalculate settings derived from changes in children
        if system_recognized:
            updates_dict.update({'n': self._system.sizes.states})
        if output_recognized:
            updates_dict.update({
                'n_saved_states': self._output_functions.n_saved_states,
                'n_summarised_states':
                    self._output_functions.n_summarised_states,
                'compile_flags': self._output_functions.compile_flags,
            })
        if ctrl_recognized:
            updates_dict.update(
                {
                    "is_adaptive": self._step_controller.is_adaptive,
                    "dt_min": self._step_controller.dt_min,
                    "dt_max": self._step_controller.dt_max,
                    "dt0": self._step_controller.dt0,
                }
            )
        if step_recognized:
            updates_dict.update(
                {
                    "threads_per_step": self._algo_step.threads_per_step,
                }
            )

        system_sizes=self.system_sizes
        shared_indices = LoopSharedIndices.from_sizes(
            n_states=system_sizes.states,
            n_observables=system_sizes.parameters,
            n_parameters=system_sizes.observables,
            n_drivers=system_sizes.drivers,
            n_state_summaries=self._output_functions.n_summarised_states,
            n_observable_summaries=self._output_functions
            .n_summarised_observables,
        )
        local_indices = LoopLocalIndices.from_sizes(
                n_states=system_sizes.states,
                controller_len=self._step_controller.local_memory_elements,
                algorithm_len=self._algo_step.persistent_local_required,
        )
        updates_dict.update({'shared_buffer_indices': shared_indices,
                             'local_indices': local_indices})

        recognized |= self.update_compile_settings(updates_dict, silent=True)

        all_unrecognized -= recognized
        if all_unrecognized and not silent:
            raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        if recognized:
            self._invalidate_cache()

        self.check_compatibility()

        return recognized

    def build(self) -> Callable:
        """Instantiate the step controller, algorithm step, and loop."""

        # Lowest level - check for changes in dxdt_fn, get_solver_helper_fn
        dxdt_fn = self._system.dxdt_function
        get_solver_helper_fn = self._system.get_solver_helper
        compiled_fns_dict = {}
        if dxdt_fn != self._algo_step.dxdt_function:
            compiled_fns_dict['dxdt_function'] = dxdt_fn
        if get_solver_helper_fn != self._algo_step.get_solver_helper_fn:
            compiled_fns_dict['get_solver_helper_fn'] = get_solver_helper_fn

        #Build algorithm fn after change made
        self._algo_step.update(compiled_fns_dict)

        compiled_functions = {
            'save_state_fn': self._output_functions.save_state_func,
            'update_summaries_fn': self._output_functions.update_summaries_func,
            'save_summaries_fn': self._output_functions.save_summary_metrics_func,
            'step_controller_fn': self._step_controller.device_function,
            'step_fn': self._algo_step.step_function,
        }

        self._loop.update(compiled_functions)
        loop_fn = self._loop.device_function

        return loop_fn

