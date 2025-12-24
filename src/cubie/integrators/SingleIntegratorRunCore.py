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

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import warnings

import attrs

from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie._utils import PrecisionDType, unpack_dict_values
from cubie.buffer_registry import buffer_registry
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling import OutputCompileFlags
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.integrators.step_control import get_controller


if TYPE_CHECKING:  # pragma: no cover - imported for static typing only
    from cubie.odesystems.baseODE import BaseODE


@attrs.define
class SingleIntegratorRunCache(CUDAFunctionCache):
    """Cache for SingleIntegratorRunCore device function.
    
    Attributes
    ----------
    single_integrator_function
        Compiled CUDA loop callable ready for execution on device.
    """
    single_integrator_function: Callable = attrs.field(eq=False)

class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system
        ODE system whose device functions drive the integration.
    loop_settings
        Mapping of compile-critical loop configuration forwarded to the
        :class:`cubie.integrators.loops.ode_loop.IVPLoop`. Recognised keys
        include ``"dt_save"`` and ``"dt_summarise"``. When ``None`` the loop
        falls back to built-in defaults.
    output_settings
        Mapping forwarded to :class:`cubie.outputhandling.output_functions.
        OutputFunctions`. Recognised keys include ``"output_types"`` and
        the saved or summarised selector fields:
        ``"saved_state_indices"``, ``"saved_observable_indices"``,
        ``"summarised_state_indices"``, and
        ``"summarised_observable_indices"``.
    driver_function
        Optional device function which interpolates arbitrary driver inputs
        for use by step algorithms.
    algorithm_settings
        Mapping forwarded to :func:`cubie.integrators.algorithms.get_algorithm_step`
        containing ``"algorithm"`` and any additional parameters required by
        the selected step factory. When ``None`` the algorithm defaults are
        used.
    step_control_settings
        Mapping merged with the algorithm defaults before calling
        :func:`cubie.integrators.step_control.get_controller`. Include
        ``"step_controller"`` to select a controller family and provide bounds
        such as ``"dt_min"`` and ``"dt_max"`` when configuring adaptive
        controllers. Supported identifiers include ``"fixed"``, ``"i"``,
        ``"pi"``, ``"pid"``, and ``"gustafsson"``. When ``None`` the
        algorithm defaults are used.

    Returns
    -------
    None
        Initialises the integration loop and associated components.
    """

    def __init__(
        self,
        system: "BaseODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        driver_function: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        step_control_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if step_control_settings is None:
            step_control_settings = {}
        if algorithm_settings is None:
            algorithm_settings = {}
        if output_settings is None:
            output_settings = {}
        if loop_settings is None:
            loop_settings = {}

        precision = system.precision

        self._system = system
        system_sizes = system.sizes
        n = system_sizes.states

        # Outputsettings may/may not include precision, so we pop it here to
        # ensure that it gets passed a precision matching system's
        _ = output_settings.pop("precision", None)
        self._output_functions = OutputFunctions(
            max_states=system_sizes.states,
            max_observables=system_sizes.observables,
            precision=precision,
            **output_settings,
        )

        dt = step_control_settings.get("dt", None)
        algorithm_settings["n"] = n
        if dt is not None:
            algorithm_settings["dt"] = dt
        algorithm_settings["driver_function"] = driver_function
        # Thread the driver time-derivative through to algorithm factories
        algorithm_settings["driver_del_t"] = driver_del_t
        self._algo_step = get_algorithm_step(
                precision=precision,
                settings=algorithm_settings,
        )
        # Fetch and override controller defaults from algorithm settings
        controller_settings = (
            self._algo_step.controller_defaults.step_controller.copy())
        controller_settings.update(step_control_settings)
        controller_settings["n"] = system_sizes.states
        controller_settings["algorithm_order"] = self._algo_step.order

        self._step_controller = get_controller(
            precision=precision,
            settings=controller_settings,
        )

        self.check_compatibility(
            algorithm_settings["algorithm"],
            controller_settings["step_controller"],
            precision,
        )

        loop_settings["dt0"] = self._step_controller.dt0
        loop_settings["dt_min"] = self._step_controller.dt_min
        loop_settings["dt_max"] = self._step_controller.dt_max
        loop_settings["is_adaptive"] = self._step_controller.is_adaptive

        config = IntegratorRunSettings(
            precision=system.precision,
            algorithm=algorithm_settings["algorithm"],
            step_controller=controller_settings["step_controller"],
        )

        self.setup_compile_settings(config)


        self._loop = self.instantiate_loop(
            precision=precision,
            n_states=system_sizes.states,
            n_parameters=system_sizes.parameters,
            n_observables=system_sizes.observables,
            n_drivers=system_sizes.drivers,
            compile_flags=self._output_functions.compile_flags,
            state_summaries_buffer_height=self._output_functions.state_summaries_buffer_height,
            observable_summaries_buffer_height=self._output_functions.observable_summaries_buffer_height,
            loop_settings=loop_settings,
            driver_function=driver_function,
        )

        # Register algorithm step and controller buffers with loop as parent
        buffer_registry.get_child_allocators(
            self._loop, self._algo_step, name="algorithm"
        )
        buffer_registry.get_child_allocators(
                self._loop, self._step_controller, name='controller'
        )


    @property
    def n_error(self) -> int:
        """Return the length of the shared error buffer."""

        if self._algo_step.is_adaptive:
            return int(self._system.sizes.states)
        return 0

    @property
    def device_function(self):
        """Return the compiled CUDA solver kernel.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output("single_integrator_function")

    def check_compatibility(
        self,
        algorithm_name: str = None,
        controller_name: str = None,
        precision: PrecisionDType = None,
    ) -> None:
        """Validate algorithm and controller compatibility.

        This method checks whether the chosen integration algorithm and step
        controller are compatible. When an adaptive controller is paired with
        a fixed-step (errorless) algorithm, this method replaces the adaptive
        controller with a fixed-step controller and issues a warning.

        The validation is performed during integrator initialization, after
        both the algorithm and controller have been instantiated but before
        the CUDA loop is compiled.

        Parameters
        ----------
        algorithm_name : str, optional
            Name of the algorithm being used. If not provided, retrieved from
            compile_settings.
        controller_name : str, optional
            Name of the controller being used. If not provided, retrieved from
            compile_settings.
        precision : PrecisionDType, optional
            Numerical precision for the controller. If not provided, retrieved
            from system.

        Notes
        -----
        When an incompatible configuration is detected (adaptive controller
        with errorless algorithm), the controller is automatically replaced
        with a fixed-step controller using dt0 from the original controller.
        A warning is issued to inform the user of this automatic correction.

        Valid combinations:
        - Adaptive algorithm + adaptive controller: Valid and recommended
        - Errorless algorithm + fixed controller: Valid
        - Adaptive algorithm + fixed controller: Valid (uses fixed step)
        - Errorless algorithm + adaptive controller: Auto-corrected with warning
        """

        if (not self._algo_step.is_adaptive and
                self._step_controller.is_adaptive):
            dt = self._step_controller.dt0
            
            # Get names from arguments or compile_settings
            if algorithm_name is None:
                algorithm_name = self.compile_settings.algorithm
            if controller_name is None:
                controller_name = self.compile_settings.step_controller
            if precision is None:
                precision = self._system.precision
            
            warnings.warn(
                f"Adaptive step controller '{controller_name}' cannot be "
                f"used with fixed-step algorithm '{algorithm_name}'. "
                f"The algorithm does not provide an error estimate "
                f"required for adaptive stepping. "
                f"Replacing with fixed-step controller (dt={dt}).",
                UserWarning,
                stacklevel=3
            )
            
            # Replace with fixed step controller
            self._step_controller = get_controller(
                precision=precision,
                settings={
                    "step_controller": "fixed",
                    "dt": dt,
                    "n": self._system.sizes.states,
                },
                warn_on_unused=False,
            )
            
        # Set the is_controller_fixed attribute on the algorithm
        self._algo_step.is_controller_fixed = (
            not self._step_controller.is_adaptive
        )

    def instantiate_loop(
        self,
        precision: PrecisionDType,
        n_states: int,
        n_parameters: int,
        n_observables: int,
        n_drivers: int,
        state_summaries_buffer_height: int,
        observable_summaries_buffer_height: int,
        compile_flags: OutputCompileFlags,
        loop_settings: Dict[str, Any],
        driver_function: Optional[Callable] = None,
    ) -> IVPLoop:
        """Instantiate the integrator loop.

        Parameters
        ----------
        precision
            Numerical precision used when compiling the loop.
        n_states
            Number of state variables in the system.
        n_parameters
            Number of persistent parameters available to the loop.
        n_observables
            Number of observables emitted by the system.
        n_drivers
            Number of external driver signals consumed by the loop.
        state_summaries_buffer_height
            Height of the state summary buffer managed by the outputs.
        observable_summaries_buffer_height
            Height of the observable summary buffer managed by the outputs.
        compile_flags
            Output function compile flags generated by
            :class:`cubie.outputhandling.OutputFunctions`.
        loop_settings
            Mapping of loop configuration overrides forwarded directly to the
            :class:`~cubie.integrators.loops.ode_loop.IVPLoop` constructor.
        driver_function
            Optional device function that evaluates drivers for proposed times.

        Returns
        -------
        IVPLoop
            Configured loop instance ready for CUDA compilation.
        """
        n_counters = 4 if compile_flags.save_counters else 0
        
        loop_kwargs = dict(loop_settings)

        # Build the loop with individual parameters (new API)
        loop_kwargs.update(
            precision=precision,
            n_states=n_states,
            compile_flags=compile_flags,
            n_parameters=n_parameters,
            n_drivers=n_drivers,
            n_observables=n_observables,
            n_error=self.n_error,
            n_counters=n_counters,
            state_summaries_buffer_height=state_summaries_buffer_height,
            observable_summaries_buffer_height=observable_summaries_buffer_height,
        )
        if "driver_function" not in loop_kwargs:
            loop_kwargs["driver_function"] = driver_function

        loop = IVPLoop(**loop_kwargs)
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
        When algorithm or controller selections change, new instances are
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

        # Flatten any nested dict values so that all parameters are
        # top-level keys before passing to sub-components. For example,
        # step_controller_settings={'dt_min': 0.01} becomes dt_min=0.01.
        # This ensures sub-components (algorithm, controller, output
        # functions) receive only flat parameter sets.
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        all_unrecognized = set(updates_dict.keys())
        recognized = set()
        system_recognized = self._system.update(updates_dict, silent=True)

        # Capture n whether or not system updated, in case of an algo/step swap
        updates_dict.update({'n': self._system.sizes.states})

        # Capture outputsettings-generated compile settings and pass on
        out_rcgnzd = self._output_functions.update(updates_dict, silent=True)
        if out_rcgnzd:
            updates_dict.update({**self._output_functions.buffer_sizes_dict})

        # Capture algorithm-generated compile settings and pass on
        step_recognized = self._switch_algos(updates_dict)
        step_recognized |= self._algo_step.update(updates_dict, silent=True)
        if step_recognized:
            updates_dict.update(
                {"threads_per_step": self._algo_step.threads_per_step}
            )

        updates_dict["algorithm_order"] = self._algo_step.order

        ctrl_rcgnzd = self._switch_controllers(updates_dict)
        ctrl_rcgnzd |= self._step_controller.update(updates_dict, silent=True)
        if ctrl_rcgnzd:
            updates_dict.update(
                {
                    "is_adaptive": self._step_controller.is_adaptive,
                    "dt_min": self._step_controller.dt_min,
                    "dt_max": self._step_controller.dt_max,
                    "dt0": self._step_controller.dt0,
                }
            )

        # Re-register algo and controller buffers to refresh sizing in loop
        buffer_registry.get_child_allocators(
                self._loop, self._algo_step, name='algorithm'
        )
        buffer_registry.get_child_allocators(
                self._loop, self._step_controller, name='controller'
        )
        loop_recognized = self._loop.update(updates_dict, silent=True)
        recognized |= self.update_compile_settings(updates_dict, silent=True)

        recognized |= (out_rcgnzd | ctrl_rcgnzd | step_recognized |
                       system_recognized | loop_recognized)

        all_unrecognized -= recognized
        if all_unrecognized and not silent:
            raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        if recognized:
            self._invalidate_cache()

        self.check_compatibility()

        # Include unpacked dict keys in recognized set
        return recognized | unpacked_keys

    def _switch_algos(self, updates_dict):
        if "algorithm" not in updates_dict:
            return set()
        precision = updates_dict.get('precision', self.precision)

        buffer_registry.reset()

        new_algo = updates_dict.get("algorithm").lower()
        if new_algo != self.compile_settings.algorithm:
            old_settings = self._algo_step.settings_dict
            old_settings["algorithm"] = new_algo
            self._algo_step = get_algorithm_step(
                    precision=precision,
                    settings=old_settings,
            )
            self.compile_settings.algorithm = new_algo
        updates_dict["algorithm"] = new_algo

        # Update any not-deliberately-updated controller settings with defaults
        algo_defaults = self._algo_step.controller_defaults.step_controller
        for key, value in algo_defaults.items():
            if key not in updates_dict:
                updates_dict[key] = value
        updates_dict["algorithm_order"] = self._algo_step.order
        return set("algorithm")

    def _switch_controllers(self, updates_dict):
        if "step_controller" not in updates_dict:
            return set()
        precision = updates_dict.get('precision', self.precision)

        buffer_registry.reset()
        new_controller = updates_dict.get("step_controller").lower()

        if new_controller != self.compile_settings.step_controller:
            old_settings = self._step_controller.settings_dict
            old_settings["step_controller"] = new_controller
            old_settings["algorithm_order"] = updates_dict.get(
                "algorithm_order", self._algo_step.order)
            self._step_controller = get_controller(
                    precision=precision,
                    settings=old_settings,
            )
            self.compile_settings.step_controller = new_controller
        updates_dict["step_controller"] = new_controller
        return set("step_controller")

    def build(self) -> SingleIntegratorRunCache:
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
            compiled_fns_dict["dxdt_function"] = dxdt_fn
        if observables_fn != self._algo_step.observables_function:
            compiled_fns_dict["observables_function"] = observables_fn
        if get_solver_helper_fn != self._algo_step.get_solver_helper_fn:
            compiled_fns_dict['get_solver_helper_fn'] = get_solver_helper_fn

        #Build algorithm fn after change made
        self._algo_step.update(compiled_fns_dict)

        # Re-register algo and controller buffers to refresh sizing in loop
        buffer_registry.get_child_allocators(
                self._loop, self._algo_step, name='algorithm'
        )
        buffer_registry.get_child_allocators(
                self._loop, self._step_controller, name='controller'
        )
        compiled_functions = {
            'save_state_fn': self._output_functions.save_state_func,
            'update_summaries_fn': self._output_functions.update_summaries_func,
            'save_summaries_fn': self._output_functions.save_summary_metrics_func,
            'step_controller_fn': self._step_controller.device_function,
            'step_function': self._algo_step.step_function,
            'observables_fn': observables_fn}

        self._loop.update(compiled_functions)
        loop_fn = self._loop.device_function

        return SingleIntegratorRunCache(single_integrator_function=loop_fn)
