"""
Array sizing classes for output buffer and array management.

This module provides classes for calculating and managing the sizes of various
arrays used in CUDA batch solving, including buffers for intermediate
calculations and output arrays for results. All classes inherit from
ArraySizingClass which provides utilities for memory allocation.
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
    from cubie.outputhandling.output_functions import OutputFunctions
    from cubie.odesystems.systems.BaseODE import BaseODE

from abc import ABC

import attrs
from numpy import ceil

from cubie.batchsolving._utils import ensure_nonzero_size


@attrs.define
class ArraySizingClass(ABC):
    """
    Base class for array sizing with memory allocation utilities.

    This abstract base class provides common functionality for all array
    sizing classes, including a method to ensure minimum array sizes for
    memory allocation purposes.

    Notes
    -----
    All derived classes inherit the nonzero property which is useful for
    allocating memory where zero-sized arrays would cause issues.
    """

    @property
    def nonzero(self) -> "ArraySizingClass":
        """
        Return a copy with all sizes having a minimum of one element.

        Returns
        -------
        ArraySizingClass
            Copy of the object with all integer and tuple sizes ensured
            to be at least 1, useful for memory allocation.

        Notes
        -----
        This method creates a new object instance with modified size values
        to prevent zero-sized array allocations which can cause issues in
        CUDA memory management.
        """
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, ensure_nonzero_size(value))
        return new_obj


@attrs.define
class SummariesBufferSizes(ArraySizingClass):
    """
    Buffer sizes for summary metric calculations.

    This class provides a clean interface for accessing summary buffer sizes
    with descriptive names rather than accessing them through more complex
    object hierarchies.

    Attributes
    ----------
    state : int, default 1
        Buffer size required for state variable summaries.
    observables : int, default 1
        Buffer size required for observable variable summaries.
    per_variable : int, default 1
        Buffer size required per individual variable for summaries.

    Notes
    -----
    Most commonly used via the from_output_fns class method which extracts
    the relevant sizes from an OutputFunctions object.
    """

    state: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    per_variable: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_output_fns(
        cls, output_fns: "OutputFunctions"
    ) -> "SummariesBufferSizes":
        """
        Create buffer sizes from an OutputFunctions object.

        Parameters
        ----------
        output_fns : OutputFunctions
            Output functions object containing buffer size information.

        Returns
        -------
        SummariesBufferSizes
            Buffer sizes extracted from the output functions configuration.
        """
        return cls(
            output_fns.state_summaries_buffer_height,
            output_fns.observable_summaries_buffer_height,
            output_fns.summaries_buffer_height_per_var,
        )


@attrs.define
class LoopBufferSizes(ArraySizingClass):
    """
    Buffer sizes for integrator inner loop calculations.

    This class holds the sizes of all buffers used in the main integration
    loop, including system state buffers and summary calculation buffers.

    Attributes
    ----------
    state_summaries : int, default 1
        Buffer size for state summary calculations.
    observable_summaries : int, default 1
        Buffer size for observable summary calculations.
    state : int, default 1
        Buffer size for current state values.
    observables : int, default 1
        Buffer size for current observable values.
    dxdt : int, default 1
        Buffer size for state derivatives.
    parameters : int, default 1
        Buffer size for system parameters.
    drivers : int, default 1
        Buffer size for external driving forces.

    Notes
    -----
    Combines system-based buffer sizes (state, dxdt, parameters) with
    output-derived buffer sizes (summaries) to provide a complete view
    of memory requirements for the integration loop.
    """

    state_summaries: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observable_summaries: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    state: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    dxdt: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    parameters: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    drivers: Optional[int] = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_system_and_output_fns(
        cls,
        system: "BaseODE",
        output_fns: "OutputFunctions",
    ) -> "LoopBufferSizes":
        """
        Create buffer sizes from system and output function objects.

        Parameters
        ----------
        system : BaseODE
            System model containing state and parameter dimensions.
        output_fns : OutputFunctions
            Output functions containing summary buffer requirements.

        Returns
        -------
        LoopBufferSizes
            Combined buffer sizes for the integration loop.
        """
        summary_sizes = SummariesBufferSizes.from_output_fns(output_fns)
        system_sizes = system.sizes
        obj = cls(
            summary_sizes.state,
            summary_sizes.observables,
            system_sizes.states,
            system_sizes.observables,
            system_sizes.states,
            system_sizes.parameters,
            system_sizes.drivers,
        )
        return obj

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "LoopBufferSizes":
        """
        Create buffer sizes from a BatchSolverKernel object.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            Solver instance containing system and output configuration.

        Returns
        -------
        LoopBufferSizes
            Buffer sizes extracted from the solver configuration.
        """
        system_sizes = solver_instance.system_sizes
        summary_sizes = solver_instance.summaries_buffer_sizes
        return cls(
            summary_sizes.state,
            summary_sizes.observables,
            system_sizes.states,
            system_sizes.observables,
            system_sizes.states,
            system_sizes.parameters,
            system_sizes.drivers,
        )


@attrs.define
class OutputArrayHeights(ArraySizingClass):
    """
    Heights of output arrays for different variable types.

    This class specifies the height (number of elements) required for
    output arrays storing results from batch integrations.

    Attributes
    ----------
    state : int, default 1
        Height of state variable output arrays.
    observables : int, default 1
        Height of observable variable output arrays.
    state_summaries : int, default 1
        Height of state summary output arrays.
    observable_summaries : int, default 1
        Height of observable summary output arrays.
    per_variable : int, default 1
        Height per variable for summary outputs.
    """

    state: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    state_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observable_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    per_variable: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_output_fns(
        cls, output_fns: "OutputFunctions"
    ) -> "OutputArrayHeights":
        """
        Create output array heights from an OutputFunctions object.

        Parameters
        ----------
        output_fns : OutputFunctions
            Output functions object containing configuration information.

        Returns
        -------
        OutputArrayHeights
            Array heights calculated from the output configuration.

        Notes
        -----
        The state height includes an extra element if time saving is enabled.
        """
        state = output_fns.n_saved_states + 1 * output_fns.save_time
        observables = output_fns.n_saved_observables
        state_summaries = output_fns.state_summaries_output_height
        observable_summaries = output_fns.observable_summaries_output_height
        per_variable = output_fns.summaries_output_height_per_var
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
            per_variable,
        )
        return obj


@attrs.define
class SingleRunOutputSizes(ArraySizingClass):
    """
    Output array sizes for a single integration run.

    This class provides 2D array sizes (time × variable) for output arrays
    from a single integration run.

    Attributes
    ----------
    state : tuple[int, int], default (1, 1)
        Shape of state output array as (time_samples, n_variables).
    observables : tuple[int, int], default (1, 1)
        Shape of observable output array as (time_samples, n_variables).
    state_summaries : tuple[int, int], default (1, 1)
        Shape of state summary array as (summary_samples, n_summaries).
    observable_summaries : tuple[int, int], default (1, 1)
        Shape of observable summary array as (summary_samples, n_summaries).
    stride_order : tuple[str, ...], default ("time", "variable")
        Order of dimensions in the arrays.
    """

    state: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observables: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    state_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observable_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    stride_order: Tuple[str, ...] = attrs.field(
        default=("time", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["time", "variable"])
        ),
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "SingleRunOutputSizes":
        """
        Create output sizes from a BatchSolverKernel object.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            Solver instance containing timing and output configuration.

        Returns
        -------
        SingleRunOutputSizes
            Output array sizes for a single run.
        """
        heights = solver_instance.output_array_heights
        output_samples = solver_instance.output_length
        summarise_samples = solver_instance.summaries_length

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (
            summarise_samples,
            heights.observable_summaries,
        )
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
        )

        return obj

    @classmethod
    def from_output_fns_and_run_settings(cls, output_fns, run_settings):
        """
        Create output sizes from output functions and run settings.

        Parameters
        ----------
        output_fns : OutputFunctions
            Output functions containing variable configuration.
        run_settings : IntegratorRunSettings
            Run settings containing timing information.

        Returns
        -------
        SingleRunOutputSizes
            Output array sizes for a single run.

        Notes
        -----
        This method is primarily used for testing. In normal operation,
        the from_solver method is preferred.
        """
        heights = OutputArrayHeights.from_output_fns(output_fns)
        output_samples = int(
            ceil(run_settings.duration / run_settings.dt_save)
        )
        summarise_samples = int(
            ceil(run_settings.duration / run_settings.dt_summarise)
        )

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (
            summarise_samples,
            heights.observable_summaries,
        )
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
        )

        return obj


@attrs.define
class BatchInputSizes(ArraySizingClass):
    """
    Input array sizes for batch integration runs.

    This class specifies the sizes of input arrays needed for batch
    processing, including initial conditions, parameters, and forcing terms.

    Attributes
    ----------
    initial_values : tuple[int, int], default (1, 1)
        Shape of initial values array as (n_runs, n_states).
    parameters : tuple[int, int], default (1, 1)
        Shape of parameters array as (n_runs, n_parameters).
    forcing_vectors : tuple[int, int or None], default (1, None)
        Shape of forcing vectors array as (n_drivers, time_steps).
    stride_order : tuple[str, ...], default ("run", "variable")
        Order of dimensions in the input arrays.
    """

    initial_values: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    parameters: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    forcing_vectors: Tuple[int, Optional[int]] = attrs.field(
        default=(1, None), validator=attrs.validators.instance_of(Tuple)
    )

    stride_order: Tuple[str, ...] = attrs.field(
        default=("run", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["run", "variable"])
        ),
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "BatchInputSizes":
        """
        Create input array sizes from a BatchSolverKernel object.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            Solver instance containing batch and system configuration.

        Returns
        -------
        BatchInputSizes
            Input array sizes for the batch run.
        """
        loopBufferSizes = LoopBufferSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        initial_values = (num_runs, loopBufferSizes.state)
        parameters = (num_runs, loopBufferSizes.parameters)
        forcing_vectors = (loopBufferSizes.drivers, None)

        obj = cls(initial_values, parameters, forcing_vectors)
        return obj


@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """
    Output array sizes for batch integration runs.

    This class provides 3D array sizes (time × run × variable) for output
    arrays from batch integration runs.

    Attributes
    ----------
    state : tuple[int, int, int], default (1, 1, 1)
        Shape of state output array as (time_samples, n_runs,
         n_variables).
    observables : tuple[int, int, int], default (1, 1, 1)
        Shape of observable output array as (time_samples, n_runs,
        n_variables).
    state_summaries : tuple[int, int, int], default (1, 1, 1)
        Shape of state summary array as (summary_samples, n_runs,
        n_summaries).
    observable_summaries : tuple[int, int, int], default (1, 1, 1)
        Shape of observable summary array as (summary_samples, n_runs,
        n_summaries).
    stride_order : tuple[str, ...], default ("time", "run", "variable")
        Order of dimensions in the output arrays.
    """

    state: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observables: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    state_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observable_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    stride_order: Tuple[str, ...] = attrs.field(
        default=("time", "run", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["time", "run", "variable"])
        ),
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "BatchOutputSizes":
        """
        Create batch output sizes from a BatchSolverKernel object.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            Solver instance containing batch and output configuration.

        Returns
        -------
        BatchOutputSizes
            Output array sizes for the batch run.

        Notes
        -----
        Constructs 3D array sizes by combining single run sizes with
        the number of runs in the batch.
        """
        single_run_sizes = SingleRunOutputSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        state = (
            single_run_sizes.state[0],
            num_runs,
            single_run_sizes.state[1],
        )
        observables = (
            single_run_sizes.observables[0],
            num_runs,
            single_run_sizes.observables[1],
        )
        state_summaries = (
            single_run_sizes.state_summaries[0],
            num_runs,
            single_run_sizes.state_summaries[1],
        )
        observable_summaries = (
            single_run_sizes.observable_summaries[0],
            num_runs,
            single_run_sizes.observable_summaries[1],
        )
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
        )
        return obj
