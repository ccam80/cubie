"""Single integrator run coordination for CUDA-based ODE solving.

This module provides the :class:`SingleIntegratorRunCore` class which
coordinates the modular integrator loop
(:class:`~cubie.integrators.loops.ode_loop.IVPLoop`) and its dependencies.

Notes
-----
Dependency injection of the algorithm step, controller, and output
handlers occurs during initialisation so that the compiled CUDA loop can
be rebuilt when any component is reconfigured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

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


if TYPE_CHECKING:  # pragma: no cover - imported for static typing only
    from cubie.integrators.algorithms.base_algorithm_step import (
        BaseAlgorithmStep,
    )
    from cubie.integrators.step_control.base_step_controller import (
        BaseStepController,
    )
    from cubie.odesystems.baseODE import BaseODE


class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system
        ODE system whose device functions drive the integration.
    algorithm
        Name of the algorithm step implementation. Defaults to ``"euler"``.
    dt_min
        Minimum step size forwarded to the controller. Defaults to ``0.01``.
    dt_max
        Maximum step size forwarded to the controller. Defaults to ``0.1``.
    fixed_step_size
        Step size supplied to fixed-step algorithms. Defaults to ``0.01``.
    dt_save
        Interval used when saving full state trajectories. Defaults to
        ``0.1``.
    dt_summarise
        Interval used when saving summary metrics. Defaults to ``1.0``.
    atol
        Absolute tolerance supplied to adaptive controllers. Defaults to
        ``1e-6``.
    rtol
        Relative tolerance supplied to adaptive controllers. Defaults to
        ``1e-6``.
    saved_state_indices
        Indices of state variables saved by the output handler.
    saved_observable_indices
        Indices of observables saved by the output handler.
    summarised_state_indices
        Indices of state variables summarised by the output handler.
    summarised_observable_indices
        Indices of observables summarised by the output handler.
    output_types
        Output modes requested from the output handler.
    driver_function
        Optional device function which interpolates arbitrary driver inputs
    step_controller_kind
        Controller family used to manage step sizes. Defaults to
        ``"fixed"``.
    algorithm_parameters
        Additional keyword arguments forwarded to the algorithm factory.
    step_controller_parameters
        Additional keyword arguments forwarded to the controller factory.

    Returns
    -------
    None
        Initialises the integration loop and associated components.
    """

    def __init__(
        self,
        system: BaseODE,
        algorithm: str = "euler",
        dt_min: float = 0.01,
        dt_max: float = 0.1,
        fixed_step_size: Optional[float] =None,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        saved_state_indices: Optional[ArrayLike] = None,
        saved_observable_indices: Optional[ArrayLike] = None,
        summarised_state_indices: Optional[ArrayLike] = None,
        summarised_observable_indices: Optional[ArrayLike] = None,
        output_types: Optional[list[str]] = None,
        driver_function: Optional[Callable] = None,
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

        if fixed_step_size is None:
            fixed_step_size = dt_min
        self._step_controller = self.instantiate_controller(
                step_controller_kind,
                dt_min=dt_min,
                dt_max=dt_max,
                fixed_step_size=fixed_step_size,
                atol=atol,
                rtol=rtol,
                **(step_controller_parameters or {}),
        )

        self._algo_step = self.instantiate_step_object(
            algorithm,
            n=system.sizes.states,
            dxdt_function=self._system.dxdt_function,
            observables_function=self._system.observables_function,
            get_solver_helper_fn=self._system.get_solver_helper,
            driver_function=driver_function,
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
                state_summaries_buffer_height= self._output_functions
                .state_summaries_buffer_height,
                observable_summaries_buffer_height= self._output_functions
                .observable_summaries_buffer_height,
                is_adaptive=self._step_controller.is_adaptive,
                driver_function=driver_function
        )

    def check_compatibility(self) -> None:
        """Validate that algorithm and controller step modes are aligned.

        Raises
        ------
        ValueError
            Raised when an adaptive controller is paired with a fixed-step
            algorithm.
        """

        if (not self._algo_step.is_adaptive and
                self._step_controller.is_adaptive):
            raise ValueError(
                "Adaptive step controller cannot be used with fixed-step "
                "algorithm.",
            )

    def instantiate_step_object(
        self,
        kind: str = "euler",
        n: int = 1,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        step_size: float = 1e-3,
        **kwargs: Any,
    ) -> BaseAlgorithmStep:
        """Instantiate the algorithm step.

        Parameters
        ----------
        kind
            Algorithm identifier recognised by
            :func:`cubie.integrators.algorithms.get_algorithm_step`.
        n
            Number of state variables supplied to the algorithm constructor.
        dxdt_function
            Device function computing the derivative of the ODE system.
        observables_function
            Device function computing system observables.
        get_solver_helper_fn
            Factory returning linear solver helpers for implicit algorithms.
        step_size
            Step size supplied to fixed-step algorithms.
        **kwargs
            Additional configuration forwarded to the algorithm factory.

        Returns
        -------
        BaseAlgorithmStep
            Instantiated algorithm step configured for the current system.

        Notes
        -----
        Supported identifiers include ``"euler"``, ``"backwards_euler"``,
        ``"backwards_euler_pc"``, and ``"crank_nicolson"``.
        """
        if kind.lower() in ["euler"]:  # fixed step algorithms
            kwargs.update({"dt": step_size})
        algorithm = get_algorithm_step(
            kind,
            precision=self.precision,
            n=n,
            dxdt_function=dxdt_function,
            get_solver_helper_fn=get_solver_helper_fn,
            observables_function=observables_function,
            driver_function=driver_function,
            **kwargs,
        )
        return algorithm

    def instantiate_controller(
        self,
        kind: str = "fixed",
        order: int = 1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        atol: Union[float, np.ndarray] = 1e-6,
        rtol: Union[float, np.ndarray] = 1e-6,
        fixed_step_size: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseStepController:
        """Instantiate the step controller.

        Parameters
        ----------
        kind
            Controller identifier accepted by
            :func:`cubie.integrators.step_control.get_controller`.
        order
            Order of the paired algorithm used for adaptive controllers.
        dt_min
            Minimum permitted step size for adaptive controllers.
        dt_max
            Maximum permitted step size for adaptive controllers.
        atol
            Absolute tolerance forwarded to adaptive controllers.
        rtol
            Relative tolerance forwarded to adaptive controllers.
        fixed_step_size
            Step size supplied to fixed controllers.
        **kwargs
            Additional configuration forwarded to the controller factory.

        Returns
        -------
        BaseStepController
            Instantiated controller configured for the integration run.

        Notes
        -----
        Supported identifiers include ``"fixed"``, ``"i"``, ``"pi"``,
        ``"pid"``, and ``"gustafsson"``.
        """
        if kind == 'fixed':
            if fixed_step_size is None:
                fixed_step_size = dt_min
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

    def instantiate_loop(
        self,
        n_states: int,
        n_parameters: int,
        n_observables: int,
        n_drivers: int,
        state_summaries_buffer_height: int,
        observable_summaries_buffer_height: int,
        controller_local_elements: int,
        algorithm_local_elements: int,
        compile_flags: OutputCompileFlags,
        dt_save: float,
        dt_summarise: float,
        dt0: float,
        dt_min: float,
        dt_max: float,
        is_adaptive: bool,
        driver_function: Optional[Callable] = None,
    ) -> IVPLoop:
        """Instantiate the integrator loop.

        Parameters
        ----------
        n_states
            Number of state variables in the system.
        n_parameters
            Number of persistent parameters available to the loop.
        n_observables
            Number of observables emitted by the system.
        n_drivers
            Number of external driver signals consumed by the loop.
        n_state_summaries
            Number of state summary metrics produced by outputs.
        n_observable_summaries
            Number of observable summary metrics produced by outputs.
        controller_local_elements
            Persistent local memory elements required by the controller.
        algorithm_local_elements
            Persistent local memory elements required by the algorithm.
        compile_flags
            Output function compile flags generated by
            :class:`cubie.outputhandling.OutputFunctions`.
        dt_save
            Loop interval for saving states.
        dt_summarise
            Loop interval for saving summary metrics.
        dt0
            Initial step size selected by the controller.
        dt_min
            Minimum allowed step size.
        dt_max
            Maximum allowed step size.
        is_adaptive
            Whether the controller performs adaptive stepping.

        Returns
        -------
        IVPLoop
            Configured loop instance ready for CUDA compilation.
        """
        shared_indices = LoopSharedIndices.from_sizes(
                n_states=n_states,
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_drivers=n_drivers,
                state_summaries_buffer_height=state_summaries_buffer_height,
                observable_summaries_buffer_height
                =observable_summaries_buffer_height
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
                       driver_function=driver_function,
                       is_adaptive=is_adaptive)
        return loop

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> set[str]:
        """Update parameters across all components.

        Parameters
        ----------
        updates_dict
            Dictionary of parameters to update.
        silent
            If ``True``, suppress warnings about unrecognised parameters.
        **kwargs
            Additional updates provided as keyword arguments.

        Returns
        -------
        set[str]
            Names of parameters that were recognised and applied.

        Raises
        ------
        KeyError
            Raised when unrecognised parameters remain and ``silent`` is
            ``False``.

        Notes
        -----
        When algorithm or controller kinds change, new instances are
        created and primed with settings from their predecessors before
        applying ``updates_dict``. Parameters present only on the new
        instance are ignored unless explicitly provided in the update.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()

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
                self._step_controller = _step_controller
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
                    dxdt_function=self._system.dxdt_function,
                    observables_function=self._system.observables_function,
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
            state_summaries_buffer_height=self._output_functions
            .state_summaries_buffer_height,
            observable_summaries_buffer_height=self._output_functions
            .observable_summaries_buffer_height,
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
        """Instantiate the step controller, algorithm step, and loop.

        Returns
        -------
        Callable
            Compiled CUDA loop callable ready for execution on device.
        """

        # Lowest level - check for changes in dxdt_fn, get_solver_helper_fn
        dxdt_fn = self._system.dxdt_function
        observables_fn = self._system.observables_function
        get_solver_helper_fn = self._system.get_solver_helper
        compiled_fns_dict = {}
        if dxdt_fn != self._algo_step.dxdt_function:
            compiled_fns_dict['dxdt_function'] = dxdt_fn
        if observables_fn != self._algo_step.observables_function:
            compiled_fns_dict['observables_function'] = observables_fn
        if get_solver_helper_fn != self._algo_step.get_solver_helper_fn:
            compiled_fns_dict['get_solver_helper_fn'] = get_solver_helper_fn

        #Build algorithm fn after change made
        self._algo_step.update(compiled_fns_dict)

        compiled_functions = {
            'save_state_fn': self._output_functions.save_state_func,
            'update_summaries_fn': self._output_functions.update_summaries_func,
            'save_summaries_fn': self._output_functions.save_summary_metrics_func,
            'step_controller_fn': self._step_controller.device_function,
            'step_function': self._algo_step.step_function,
            'observables_fn': observables_fn
        }

        self._loop.update(compiled_functions)
        loop_fn = self._loop.device_function

        return loop_fn

