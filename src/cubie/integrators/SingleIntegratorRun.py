"""High-level property aggregation for single integrator runs.

This module exposes :class:`SingleIntegratorRun`, a thin wrapper around
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
that presents compiled loop artifacts, controllers, and algorithm steps as
read-only properties for downstream consumers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np

from cubie._utils import PrecisionDType
from cubie.integrators.SingleIntegratorRunCore import SingleIntegratorRunCore
from cubie.odesystems.ODEData import SystemSizes


if TYPE_CHECKING:  # pragma: no cover - type checking import only
    from cubie.odesystems.baseODE import BaseODE


class SingleIntegratorRun(SingleIntegratorRunCore):
    """Expose aggregated read-only properties for integrator runs.

    Notes
    -----
    Instantiation, updates, and compilation are provided by
    :class:`SingleIntegratorRunCore`. This subclass intentionally limits
    itself to property-based access so that front-end utilities can inspect
    compiled CUDA components without mutating internal state.
    """

    # ------------------------------------------------------------------
    # Compile settings
    # ------------------------------------------------------------------
    @property
    def precision(self) -> PrecisionDType:
        """Return the numerical precision configured for the run."""

        return self.compile_settings.precision

    @property
    def numba_precision(self) -> type:
        """Return the Numba compatible precision for the run."""

        return self.compile_settings.numba_precision

    @property
    def algorithm(self) -> str:
        """Return the configured algorithm identifier."""

        return self.compile_settings.algorithm

    @property
    def algorithm_key(self) -> str:
        """Return the canonical algorithm identifier."""

        return self.compile_settings.algorithm

    @property
    def step_controller(self) -> str:
        """Return the configured step-controller identifier."""

        return self.compile_settings.step_controller

    # ------------------------------------------------------------------
    # Aggregated memory usage
    # ------------------------------------------------------------------
    @property
    def shared_memory_elements(self) -> int:
        """Return total shared-memory elements required by the loop."""

        loop_shared = (
            self._loop.shared_memory_elements
            if hasattr(self._loop, "shared_memory_elements")
            else 0
        )
        algorithm_shared = (
            self._algo_step.shared_memory_required
            if hasattr(self._algo_step, "shared_memory_required")
            else 0
        )
        return loop_shared + algorithm_shared

    @property
    def shared_memory_bytes(self) -> int:
        """Return total shared-memory usage in bytes."""

        element_count = self.shared_memory_elements
        itemsize = np.dtype(self.precision).itemsize
        return element_count * itemsize

    @property
    def local_memory_elements(self) -> int:
        """Return total persistent local-memory requirement."""

        loop = (
            self._loop.local_memory_elements
            if hasattr(self._loop, "local_memory_elements")
            else 0
        )
        algorithm = (
            self._algo_step.persistent_local_required
            if hasattr(self._algo_step, "persistent_local_required")
            else 0
        )
        controller = (
            self._step_controller.local_memory_elements
            if hasattr(self._step_controller, "local_memory_elements")
            else 0
        )
        return loop + algorithm + controller

    @property
    def compiled_loop_function(self) -> Callable:
        """Return the compiled loop function."""

        return self.device_function

    @property
    def threads_per_loop(self) -> int:
        """Return the number of CUDA threads required per system."""

        return self.threads_per_step

    @property
    def dt0(self) -> float:
        """Return the starting step size from the controller."""

        return self._step_controller.dt0

    @property
    def dt_min(self) -> float:
        """Return the minimum allowable step size."""

        return self._step_controller.dt_min

    @property
    def dt_max(self) -> float:
        """Return the maximum allowable step size."""

        return self._step_controller.dt_max

    @property
    def is_adaptive(self) -> bool:
        """Return whether adaptive stepping is active."""

        return self._step_controller.is_adaptive

    @property
    def system(self) -> BaseODE:
        """Return the underlying ODE system."""

        return self._system

    @property
    def system_sizes(self) -> SystemSizes:
        """Return the size descriptor for the ODE system."""

        return self._system.sizes

    @property
    def save_summaries_func(self) -> Callable:
        """Return the summary saving function from the output handlers."""

        return self.save_summary_metrics_func

    @property
    def dxdt_function(self) -> Callable:
        """Return the derivative function used by the integration step."""

        return self._algo_step.dxdt_function


    # ------------------------------------------------------------------
    # Loop properties
    # ------------------------------------------------------------------
    @property
    def dt_save(self) -> float:
        """Return the loop save interval."""

        return self._loop.dt_save

    @property
    def dt_summarise(self) -> float:
        """Return the loop summary interval."""

        return self._loop.dt_summarise

    @property
    def shared_buffer_indices(self) -> Any:
        """Return the shared buffer index layout."""

        return self._loop.shared_buffer_indices

    @property
    def buffer_indices(self) -> Any:
        """Return shared buffer indices."""

        return self._loop.buffer_indices

    @property
    def local_indices(self) -> Any:
        """Return loop local-memory indices."""

        return self._loop.local_indices

    @property
    def shared_memory_elements_loop(self) -> int:
        """Return the loop contribution to shared memory."""

        return self._loop.shared_memory_elements

    @property
    def local_memory_elements_loop(self) -> int:
        """Return the loop contribution to local memory."""

        return self._loop.local_memory_elements

    @property
    def compile_flags(self) -> Any:
        """Return loop compile flags."""

        return self._loop.compile_flags

    @property
    def save_state_fn(self) -> Callable:
        """Return the loop state-save function."""

        return self._loop.save_state_fn

    @property
    def update_summaries_fn(self) -> Callable:
        """Return the loop summary-update function."""

        return self._loop.update_summaries_fn

    @property
    def save_summaries_fn(self) -> Callable:
        """Return the loop summary-save function."""

        return self._loop.save_summaries_fn

    @property
    def control_device_function(self) -> Callable:
        """Return the compiled controller device function."""

        return self._loop.step_controller_fn

    @property
    def compiled_loop_step_function(self) -> Callable:
        """Return the compiled algorithm step function."""

        return self._loop.step_function

    # ------------------------------------------------------------------
    # Step controller properties
    # ------------------------------------------------------------------
    @property
    def local_memory_elements_controller(self) -> int:
        """Return the controller contribution to local memory."""

        return self._step_controller.local_memory_elements

    @property
    def min_gain(self) -> Optional[float]:
        """Return the minimum step size gain."""

        controller = self._step_controller
        return controller.min_gain if hasattr(controller, "min_gain") else None

    @property
    def max_gain(self) -> Optional[float]:
        """Return the maximum step size gain."""

        controller = self._step_controller
        return controller.max_gain if hasattr(controller, "max_gain") else None

    @property
    def safety(self) -> Optional[float]:
        """Return the controller safety factor."""

        controller = self._step_controller
        return controller.safety if hasattr(controller, "safety") else None

    @property
    def algorithm_order(self) -> Optional[int]:
        """Return the algorithm order assumed by the controller."""

        controller = self._step_controller
        if hasattr(controller, "algorithm_order"):
            return controller.algorithm_order
        return None

    @property
    def atol(self) -> Optional[Any]:
        """Return the absolute tolerance array."""

        controller = self._step_controller
        return controller.atol if hasattr(controller, "atol") else None

    @property
    def rtol(self) -> Optional[Any]:
        """Return the relative tolerance array."""

        controller = self._step_controller
        return controller.rtol if hasattr(controller, "rtol") else None

    @property
    def kp(self) -> Optional[float]:
        """Return the proportional gain."""

        controller = self._step_controller
        return controller.kp if hasattr(controller, "kp") else None

    @property
    def ki(self) -> Optional[float]:
        """Return the integral gain."""

        controller = self._step_controller
        return controller.ki if hasattr(controller, "ki") else None

    @property
    def kd(self) -> Optional[float]:
        """Return the derivative gain."""

        controller = self._step_controller
        return controller.kd if hasattr(controller, "kd") else None

    @property
    def gamma(self) -> Optional[float]:
        """Return the Gustafsson damping factor."""

        controller = self._step_controller
        return controller.gamma if hasattr(controller, "gamma") else None

    @property
    def max_newton_iters(self) -> Optional[int]:
        """Return the maximum Newton iterations used by the controller."""

        controller = self._step_controller
        if hasattr(controller, "max_newton_iters"):
            return controller.max_newton_iters
        return None

    @property
    def dt(self) -> Optional[float]:
        """Return the fixed step size for fixed controllers."""
        controller = self._step_controller
        return controller.dt if hasattr(controller, "dt") else None

    @property
    def control_settings(self) -> Dict[str, Any]:
        """Return the controller settings dictionary."""

        return dict(self._step_controller.settings_dict)

    # ------------------------------------------------------------------
    # Algorithm step properties
    # ------------------------------------------------------------------
    @property
    def threads_per_step(self) -> int:
        """Return the number of threads required by the step function."""

        return self._algo_step.threads_per_step

    @property
    def uses_multiple_stages(self) -> bool:
        """Return whether the algorithm uses multiple stages."""

        return self._algo_step.is_multistage

    @property
    def adapts_step(self) -> bool:
        """Return whether the algorithm inherently adapts its step."""

        return self._algo_step.is_adaptive

    @property
    def shared_memory_required_step(self) -> int:
        """Return algorithm shared-memory requirements."""

        return self._algo_step.shared_memory_required

    @property
    def local_scratch_required_step(self) -> int:
        """Return scratch local-memory requirements for the algorithm."""

        return self._algo_step.local_scratch_required

    @property
    def local_memory_required_step(self) -> int:
        """Return persistent local-memory requirements for the algorithm."""

        return self._algo_step.persistent_local_required

    @property
    def implicit_step(self) -> bool:
        """Return whether the algorithm is implicit."""

        return self._algo_step.is_implicit

    @property
    def order(self) -> int:
        """Return the algorithm order."""

        return self._algo_step.order

    @property
    def integration_step_function(self) -> Callable:
        """Return the compiled step function."""

        return self._algo_step.step_function

    @property
    def nonlinear_solver_function(self) -> Callable:
        """Return the compiled nonlinear solver function."""

        return self._algo_step.nonlinear_solver_function

    @property
    def state_count(self) -> int:
        """Return the algorithm state count."""

        return self._algo_step.n

    @property
    def solver_helper(self) -> Callable:
        """Return the solver helper factory used by the algorithm."""

        return self._algo_step.get_solver_helper_fn

    @property
    def beta_coefficient(self) -> Optional[Any]:
        """Return the implicit beta coefficient."""

        step = self._algo_step
        return step.beta if hasattr(step, "beta") else None

    @property
    def gamma_coefficient(self) -> Optional[Any]:
        """Return the implicit gamma coefficient."""

        step = self._algo_step
        return step.gamma if hasattr(step, "gamma") else None

    @property
    def mass_matrix(self) -> Optional[Any]:
        """Return the implicit mass matrix."""

        step = self._algo_step
        return step.mass_matrix if hasattr(step, "mass_matrix") else None

    @property
    def preconditioner_order(self) -> Optional[int]:
        """Return the implicit preconditioner order."""

        step = self._algo_step
        return (
            step.preconditioner_order
            if hasattr(step, "preconditioner_order")
            else None
        )

    @property
    def linear_solver_tolerance(self) -> Optional[float]:
        """Return the linear solve tolerance."""

        step = self._algo_step
        return (
            step.krylov_tolerance
            if hasattr(step, "krylov_tolerance")
            else None
        )

    @property
    def max_linear_iterations(self) -> Optional[int]:
        """Return the maximum linear iterations."""

        step = self._algo_step
        return step.max_linear_iters if hasattr(step, "max_linear_iters") else None

    @property
    def linear_correction_type(self) -> Optional[Any]:
        """Return the linear correction strategy."""

        step = self._algo_step
        return (
            step.linear_correction_type
            if hasattr(step, "linear_correction_type")
            else None
        )

    @property
    def newton_tolerance(self) -> Optional[float]:
        """Return the nonlinear solve tolerance."""

        step = self._algo_step
        return (
            step.newton_tolerance
            if hasattr(step, "newton_tolerance")
            else None
        )

    @property
    def newton_iterations_limit(self) -> Optional[int]:
        """Return the maximum Newton iterations for the step."""

        step = self._algo_step
        return (
            step.max_newton_iters
            if hasattr(step, "max_newton_iters")
            else None
        )

    @property
    def newton_damping(self) -> Optional[float]:
        """Return the Newton damping factor."""

        step = self._algo_step
        return (
            step.newton_damping
            if hasattr(step, "newton_damping")
            else None
        )

    @property
    def newton_max_backtracks(self) -> Optional[int]:
        """Return the maximum Newton backtracking steps."""

        step = self._algo_step
        return (
            step.newton_max_backtracks
            if hasattr(step, "newton_max_backtracks")
            else None
        )

    @property
    def integration_step_size(self) -> Optional[float]:
        """Return the fixed step size for explicit steps."""

        step = self._algo_step
        return step.dt if hasattr(step, "dt") else None

    # ------------------------------------------------------------------
    # Output function properties
    # ------------------------------------------------------------------
    @property
    def save_state_func(self) -> Callable:
        """Return the compiled state saving function."""

        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self) -> Callable:
        """Return the compiled summary update function."""

        return self._output_functions.update_summaries_func

    @property
    def save_summary_metrics_func(self) -> Callable:
        """Return the compiled summary saving function."""

        return self._output_functions.save_summary_metrics_func

    @property
    def output_types(self) -> Any:
        """Return the configured output types."""

        return self._output_functions.output_types

    @property
    def output_compile_flags(self) -> Any:
        """Return the output compile flags."""

        return self._output_functions.compile_flags

    @property
    def save_time(self) -> bool:
        """Return whether time saving is enabled."""

        return self._output_functions.save_time

    @property
    def saved_state_indices(self) -> Any:
        """Return the saved state indices."""

        return self._output_functions.saved_state_indices

    @property
    def saved_observable_indices(self) -> Any:
        """Return the saved observable indices."""

        return self._output_functions.saved_observable_indices

    @property
    def summarised_state_indices(self) -> Any:
        """Return the summarised state indices."""

        return self._output_functions.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> Any:
        """Return the summarised observable indices."""

        return self._output_functions.summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """Return the number of saved states."""

        return self._output_functions.n_saved_states

    @property
    def n_saved_observables(self) -> int:
        """Return the number of saved observables."""

        return self._output_functions.n_saved_observables

    @property
    def state_summaries_output_height(self) -> int:
        """Return the state summary output height."""

        return self._output_functions.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """Return the observable summary output height."""

        return self._output_functions.observable_summaries_output_height

    @property
    def summary_buffer_height_per_variable(self) -> int:
        """Return the summary buffer height per variable."""

        return self._output_functions.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """Return the total state summary buffer height."""

        return self._output_functions.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Return the total observable summary buffer height."""

        return self._output_functions.observable_summaries_buffer_height

    @property
    def total_summary_buffer_size(self) -> int:
        """Return the total summary buffer size."""

        return self._output_functions.total_summary_buffer_size

    @property
    def summary_output_height_per_variable(self) -> int:
        """Return the summary output height per variable."""

        return self._output_functions.summaries_output_height_per_var

    @property
    def n_summarised_states(self) -> int:
        """Return the number of summarised states."""

        return self._output_functions.n_summarised_states

    @property
    def n_summarised_observables(self) -> int:
        """Return the number of summarised observables."""

        return self._output_functions.n_summarised_observables

    @property
    def summaries_buffer_sizes(self) -> Any:
        """Return the summary buffer size descriptor."""

        return self._output_functions.summaries_buffer_sizes

    @property
    def output_array_heights(self) -> Any:
        """Return the output array height descriptor."""

        return self._output_functions.output_array_heights

    @property
    def summary_legend_per_variable(self) -> Any:
        """Return the summary legend per variable."""

        return self._output_functions.summary_legend_per_variable


__all__ = ["SingleIntegratorRun"]
