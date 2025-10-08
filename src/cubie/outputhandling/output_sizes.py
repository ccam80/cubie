"""Sizing helpers for output buffers and arrays.

The classes in this module compute buffer and array shapes needed for CUDA
batch solving, covering temporary loop storage as well as host-visible output
layouts. Each class inherits from :class:`ArraySizingClass`, which offers a
utility for coercing zero-sized buffers to a minimum of one element for safe
allocation.
"""

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
    from cubie.outputhandling.output_functions import OutputFunctions
    from cubie.odesystems.baseODE import BaseODE
    from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings

from abc import ABC

import attrs
from numpy import ceil

from cubie._utils import ensure_nonzero_size


@attrs.define
class ArraySizingClass(ABC):
    """Base class for output sizing helpers.

    Notes
    -----
    All subclasses inherit the :pyattr:`nonzero` property, which ensures that
    every integer or tuple size has a minimum length of one so host-side
    allocation code can safely request buffers.
    """

    @property
    def nonzero(self) -> "ArraySizingClass":
        """Return a copy with all sizes expanded to at least one element.

        Returns
        -------
        ArraySizingClass
            A new object with every integer and tuple size coerced to a
            minimum of one.

        Notes
        -----
        CUDA allocators cannot handle zero-length buffers, so callers should
        use this property before preallocating device or host memory from
        sizing data.
        """
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, ensure_nonzero_size(value))
        return new_obj


@attrs.define
class SummariesBufferSizes(ArraySizingClass):
    """Buffer heights for summary metric staging buffers.

    Attributes
    ----------
    state : int, default 1
        Number of summary slots required for state variables.
    observables : int, default 1
        Number of summary slots required for observable variables.
    per_variable : int, default 1
        Number of slots to reserve for each tracked variable.

    Notes
    -----
    The :meth:`from_output_fns` constructor is typically used to mirror the
    configuration held by :class:`cubie.outputhandling.output_functions.OutputFunctions`.
    """

    state: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    per_variable: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_output_fns(
        cls, output_fns: "OutputFunctions"
    ) -> "SummariesBufferSizes":
        """Build a sizing instance from configured output functions.

        Parameters
        ----------
        output_fns
            Output function factory carrying summary buffer settings.

        Returns
        -------
        SummariesBufferSizes
            Buffer heights derived from the output configuration.
        """
        return cls(
            output_fns.state_summaries_buffer_height,
            output_fns.observable_summaries_buffer_height,
            output_fns.summaries_buffer_height_per_var,
        )


@attrs.define
class LoopBufferSizes(ArraySizingClass):
    """Staging buffer sizes consumed inside the integrator loop.

    Attributes
    ----------
    state_summaries : int, default 1
        Height of the state summary scratch buffer.
    observable_summaries : int, default 1
        Height of the observable summary scratch buffer.
    state : int, default 1
        Number of state values staged per system evaluation.
    observables : int, default 1
        Number of observable values staged per system evaluation.
    dxdt : int, default 1
        Number of derivative entries staged per step.
    parameters : int, default 1
        Width of the parameter slice requested by the model.
    drivers : int, default 1
        Width of the external driver slice requested by the model.

    Notes
    -----
    The sizing combines state and derivative counts from the ODE system with
    summary staging requirements derived from :class:`SummariesBufferSizes`.
    """

    state_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observable_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    state: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    dxdt: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    parameters: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    drivers: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_system_and_output_fns(
        cls,
        system: "BaseODE",
        output_fns: "OutputFunctions",
    ) -> "LoopBufferSizes":
        """Combine system and output settings into loop buffer sizes.

        Parameters
        ----------
        system
            ODE instance exposing a ``sizes`` attribute with state, observable,
            parameter, and driver counts.
        output_fns
            Output function factory describing required summary buffers.

        Returns
        -------
        LoopBufferSizes
            Aggregated staging dimensions for the integration loop.
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
        """Mirror the buffer sizing defined by a batch solver.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``system_sizes`` and
            ``summaries_buffer_sizes`` attributes.

        Returns
        -------
        LoopBufferSizes
            Staging buffer requirements copied from the solver instance.
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
    """Heights of time-series and summary outputs.

    Attributes
    ----------
    state : int, default 1
        Height of state output arrays, including a slot for time stamps when
        requested.
    observables : int, default 1
        Height of observable output arrays.
    state_summaries : int, default 1
        Height of state summary outputs.
    observable_summaries : int, default 1
        Height of observable summary outputs.
    per_variable : int, default 1
        Height reserved per tracked variable for summary outputs.
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
        """Compute array heights from configured output functions.

        Parameters
        ----------
        output_fns
            Output function factory describing which values are saved and how
            summaries are aggregated.

        Returns
        -------
        OutputArrayHeights
            Array heights derived from the output configuration.

        Notes
        -----
        The state output height reserves an extra row when time saving is
        enabled so that timestamps align with the saved states.
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
    """Output array sizes for a single integration run.

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
        """Transform solver metadata into single-run output shapes.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``output_array_heights``,
            ``output_length``, and ``summaries_length`` attributes.

        Returns
        -------
        SingleRunOutputSizes
            Array shapes for one simulation run.
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
    def from_output_fns_and_run_settings(
        cls,
        output_fns: "OutputFunctions",
        run_settings: "IntegratorRunSettings",
    ) -> "SingleRunOutputSizes":
        """Derive shapes directly from configuration objects.

        Parameters
        ----------
        output_fns
            Output function factory describing saved variables.
        run_settings
            Integration run settings providing durations and save cadences.

        Returns
        -------
        SingleRunOutputSizes
            Array shapes for one simulation run.

        Notes
        -----
        Primarily used by tests; production code prefers
        :meth:`from_solver` to remain aligned with solver metadata.
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
    """Input array sizes for batch integration runs.

    This class specifies the sizes of input arrays needed for batch
    processing, including initial conditions, parameters, and forcing terms.

    Attributes
    ----------
    initial_values : tuple[int, int], default (1, 1)
        Shape of initial values array as (n_runs, n_states).
    parameters : tuple[int, int], default (1, 1)
        Shape of parameters array as (n_runs, n_parameters).
    driver_coefficients : tuple[int or None, int, int or None],
        default (1, 1, 1)
        Shape of the driver coefficient array as
        (num_segments, num_drivers, polynomial_degree).
    stride_order : tuple[str, ...], default ("run", "variable")
        Order of dimensions in the input arrays.
    """

    initial_values: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    parameters: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    driver_coefficients: Tuple[Optional[int], int, Optional[int]] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
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
        """Create batch input shapes based on solver metadata.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``num_runs`` and loop buffer sizes.

        Returns
        -------
        BatchInputSizes
            Input array dimensions for the batch run.
        """
        loopBufferSizes = LoopBufferSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        initial_values = (num_runs, loopBufferSizes.state)
        parameters = (num_runs, loopBufferSizes.parameters)
        driver_coefficients = (None, loopBufferSizes.drivers, None)

        obj = cls(initial_values, parameters, driver_coefficients)
        return obj


@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """Output array sizes for batch integration runs.

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
    status_codes : tuple[int], default (1,)
        Shape of the status code output array indexed by run.
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
    status_codes: Tuple[int] = attrs.field(
        default=(1,), validator=attrs.validators.instance_of(Tuple)
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
        """Lift single-run shapes to batched output arrays.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``num_runs`` and single-run sizing
            helpers.

        Returns
        -------
        BatchOutputSizes
            Output array dimensions for the batch run.

        Notes
        -----
        Builds 3D arrays by pairing the number of runs with single-run heights
        for each data category.
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
        status_codes = (num_runs,)
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
            status_codes,
        )
        return obj
