"""Base classes and shared configuration for integration step factories."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Set, Any, Union, Tuple, Sequence
import warnings

import attrs
import numpy as np
from attrs import validators
import numba

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
    gttype_validator,
)
from cubie.CUDAFactory import CUDAFactory
from cubie.cuda_simsafe import from_dtype as simsafe_dtype

# Define all possible algorithm step parameters across all algorithm types
ALL_ALGORITHM_STEP_PARAMETERS = {
    'algorithm',
    'precision', 'n', 'dxdt_function', 'observables_function',
    'driver_function', 'get_solver_helper_fn', "driver_del_t",
    'dt', 'beta', 'gamma', 'M', 'preconditioner_order', 'krylov_tolerance',
    'max_linear_iters', 'linear_correction_type', 'newton_tolerance',
    'max_newton_iters', 'newton_damping', 'newton_max_backtracks',
    'n_drivers',
}


@attrs.define(frozen=True)
class ButcherTableau:
    """ Generic ``Butcher Tableau``` object.

    Attributes
    ----------
    a
        `a` matrix of the weights of other substages to the current stage
        gradient
    b
        'b' matrix of weights of the stage gradients to the final estimate (
        row 0) and the next-order-up for error calculation (row 1).
    b_hat
        Embedded weights for the higher-order estimate used when calculating
        an error signal.
    c
        'c' vector of the substage times (in proportion of step size)
    order
        Classical order of the accuracy of the method - error grows like O(
        n^order)

    Methods
    -------
    stage_count
        Return the number of stages described by the tableau.
    has_error_estimate
        Returns ``True`` when embedded error weights are supplied.
    typed_rows(rows, numba_precision)
        Returns a given matrix (rows) as precision-typed tuples for each stage.
    """

    a: Tuple[Tuple[float, ...], ...] = attrs.field()
    b: Tuple[float, ...] = attrs.field()
    c: Tuple[float, ...] = attrs.field()
    order: int = attrs.field()
    b_hat: Optional[Tuple[float, ...]] = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        """Validate tableau coefficients after initialisation."""

        stage_count = self.stage_count
        if self.b_hat is not None and len(self.b_hat) != stage_count:
            raise ValueError("b_hat must match the number of stages in b")

    @property
    def d(self) -> Optional[Tuple[float, ...]]:
        """Return coefficients for embedded error estimation."""

        if self.b_hat is None:
            return None
        return tuple(
            b_value - b_hat_value
            for b_value, b_hat_value in zip(self.b, self.b_hat)
        )

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return len(self.b)

    @property
    def has_error_estimate(self) -> bool:
        """Return ``True`` when embedded error weights are supplied."""
        error_coeffs = self.d
        if error_coeffs is None:
            return False
        return any(weight != 0.0 for weight in error_coeffs)

    def typed_rows(
        self,
        rows: Sequence[Sequence[float]],
        numba_precision: type,
    ) -> Tuple[Tuple[float, ...], ...]:
        """Pad and convert tableau rows to the requested precision."""

        typed_rows = []
        for row in rows:
            padded = list(row)
            if len(padded) < self.stage_count:
                padded.extend([0.0] * (self.stage_count - len(padded)))
            typed_rows.append(
                tuple(numba_precision(value) for value in padded)
            )
        return tuple(typed_rows)

    def typed_vector(
        self,
        vector: Sequence[float],
        numba_precision: type,
    ) -> Tuple[float, ...]:
        """Return ``vector`` typed with ``numba_precision``."""

        return tuple(numba_precision(value) for value in vector)

    def error_weights(
        self,
        numba_precision: type,
    ) -> Optional[Tuple[float, ...]]:
        """Return precision-typed weights for the embedded error estimate."""

        if not self.has_error_estimate:
            return None
        error_coeffs = self.d
        return self.typed_vector(error_coeffs, numba_precision)

    def embedded_weights(
        self,
        numba_precision: type,
    ) -> Optional[Tuple[float, ...]]:
        """Return the embedded solution weights typed to ``numba_precision``."""

        if not self.has_error_estimate:
            return None
        return self.typed_vector(self.b_hat, numba_precision)

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align."""

        return bool(self.c
                    and self.c[0] == 0.0 and self.c[-1] == 1.0
                    and self.a[0][0] == 0.0)

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when an accepted step can reuse the start state."""

        return bool(self.c and (self.c[0] == 0.0))


@attrs.define
class StepControlDefaults:
    """Per-algorithm defaults for step controller settings."""

    step_controller: Dict[str, Any] = attrs.field(factory=dict)

    def copy(self) -> "StepControlDefaults":
        """Return a deep-copy of the defaults container."""
        return StepControlDefaults(
            step_controller=dict(self.step_controller),
        )

@attrs.define
class BaseStepConfig(ABC):
    """Configuration shared by explicit and implicit integration steps.

    Parameters
    ----------
    precision
        Numerical precision to apply to device buffers. Supported values are
        ``float16``, ``float32``, and ``float64``.
    n
        Number of state entries advanced by each step call.
    n_drivers
        Number of external driver signals consumed by the step (>= 0).
    dt
        Optional fixed step size supplied by explicit algorithms.
    dxdt_function
        Device function that evaluates the system right-hand side.
    observables_function
        Device function that evaluates the system observables.
    driver_function
        Device function that evaluates driver arrays for a given time.
    get_solver_helper_fn
        Optional callable that returns device helpers required by the
        nonlinear solver construction.
    """

    precision: PrecisionDType = attrs.field(
        default=np.float32,
        converter=precision_converter,
        validator=precision_validator,
    )

    n: int = attrs.field(default=1, validator=getype_validator(int, 1))
    n_drivers: int = attrs.field(default=0, validator=getype_validator(int, 0))
    dt: Optional[float] = attrs.field(
        default=None,
        validator=validators.optional(gttype_validator(float, 0))
    )
    dxdt_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
    )
    observables_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
    )
    driver_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
    )
    get_solver_helper_fn: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(validators.is_callable()),
    )

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype associated with ``precision``."""

        return numba.from_dtype(np.dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for ``precision``."""

        return simsafe_dtype(np.dtype(self.precision))

    @property
    def settings_dict(self) -> Dict[str, object]:
        """Return a mutable view of the configuration state."""

        return {
            "n": self.n,
            "n_drivers": self.n_drivers,
            "precision": self.precision,
        }

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        tableau = getattr(self, "tableau", None)
        if tableau is None:
            return False
        return tableau.first_same_as_last

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when the accepted state seeds the next proposal.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        tableau = getattr(self, "tableau", None)
        if tableau is None:
            return False
        return tableau.can_reuse_accepted_start

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        tableau = getattr(self, "tableau", None)
        if tableau is None:
            return 1
        return tableau.stage_count

@attrs.define
class StepCache:
    """Container for compiled device helpers used by an algorithm step.

    Parameters
    ----------
    step
        Device function that advances the integration state.
    nonlinear_solver
        Optional device function used by implicit methods to perform
        nonlinear solves.
    """

    step: Callable = attrs.field(validator=is_device_validator)
    nonlinear_solver: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
    )

class BaseAlgorithmStep(CUDAFactory):
    """Base class implementing cache and configuration handling for steps.

    The class exposes properties and an ``update`` helper shared by concrete
    explicit and implicit algorithms. Concrete subclasses implement
    ``build`` to compile device helpers and provide metadata about resource
    usage.
    """

    def __init__(self,
                 config: BaseStepConfig,
                 _controller_defaults: StepControlDefaults,
                 ) -> None:
        """Initialise the algorithm step with its configuration object and its
        default runtime settings for collaborators.

        Parameters
        ----------
        config
            Configuration describing the algorithm step.
        _controller_defaults
            Per-algorithm default step controller settings.
        Returns
        -------
        None
            This constructor updates internal configuration state.
        """

        super().__init__()
        self._controller_defaults = _controller_defaults.copy()
        self.setup_compile_settings(config)

    def update(
        self,
        updates_dict: Optional[Dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Apply configuration updates and invalidate caches when needed.

        Parameters
        ----------
        updates_dict
            Mapping of configuration keys to their new values.
        silent
            When ``True``, suppress warnings about inapplicable keys.
        **kwargs
            Additional configuration updates supplied inline.

        Returns
        -------
        set
            Set of configuration keys that were recognized and updated.

        Raises
        ------
        KeyError
            Raised when an unknown key is provided while ``silent`` is False.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised

        # Check if unrecognized parameters are valid algorithm step parameters
        # but not applicable to this specific algorithm
        valid_but_inapplicable = unrecognised & ALL_ALGORITHM_STEP_PARAMETERS
        truly_invalid = unrecognised - ALL_ALGORITHM_STEP_PARAMETERS

        # Mark valid algorithm parameters as recognized to prevent error propagation
        recognised |= valid_but_inapplicable

        if valid_but_inapplicable:
            algorithm_type = self.__class__.__name__
            params_str = ", ".join(sorted(valid_but_inapplicable))
            warnings.warn(
                f"Parameters {{{params_str}}} are not recognized by {algorithm_type}; "
                "updates have been ignored.",
                UserWarning,
                stacklevel=2
            )

        if not silent and truly_invalid:
            raise KeyError(
                f"Unrecognized parameters in update: {truly_invalid}. "
                "These parameters were not updated.",
            )

        return recognised

    @property
    def precision(self) -> PrecisionDType:
        """Return the configured numerical precision."""

        return self.compile_settings.precision

    @property
    def n_drivers(self) -> int:
        """Return the configured number of external drivers."""

        return int(self.compile_settings.n_drivers)

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype used by compiled device helpers."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for the step."""

        return self.compile_settings.simsafe_precision

    @property
    def n(self) -> int:
        """Return the number of state variables advanced per step."""

        return self.compile_settings.n

    @property
    def dt(self) -> Union[float, None]:
        """Return the configured explicit step size."""
        return self.compile_settings.dt

    @property
    def controller_defaults(self) -> StepControlDefaults:
        """Return per-algorithm default settings for controllers, solvers."""
        return self._controller_defaults.copy()

    @property
    @abstractmethod
    def threads_per_step(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multistage(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        raise NotImplementedError

    @property
    def tableau(self) -> Optional[ButcherTableau]:
        """Return the configured tableau when available."""

        return getattr(self.compile_settings, "tableau", None)

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the first and last stages align.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        return self.compile_settings.first_same_as_last

    @property
    def can_reuse_accepted_start(self) -> bool:
        """Return ``True`` when the accepted state seeds the next proposal.

        Returns ``False`` when the algorithm is not tableau-based.
        """

        return self.compile_settings.can_reuse_accepted_start
    @property
    def shared_memory_required(self) -> int:
        """Return the precision-entry count of shared memory required."""

        return self.solver_shared_elements + self.algorithm_shared_elements

    @property
    @abstractmethod
    def local_scratch_required(self) -> int:
        """Return the precision-entry count of local scratch required."""
        raise NotImplementedError

    @property
    def persistent_local_required(self) -> int:
        """Return the persistent local precision-entry requirement."""

        return self.solver_local_elements + self.algorithm_local_elements

    @property
    def solver_shared_elements(self) -> int:
        """Return shared-memory elements consumed by solver infrastructure."""

        return 0

    @property
    def solver_local_elements(self) -> int:
        """Return persistent local elements consumed by solver infrastructure.

        Defaults to zero for algorithms whose solvers do not retain
        thread-local buffers between calls.
        """

        return 0

    @property
    @abstractmethod
    def algorithm_shared_elements(self) -> int:
        """Return shared-memory elements required by the algorithm itself."""
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm_local_elements(self) -> int:
        """Return persistent local elements required by the algorithm."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def order(self) -> int:
        """Return the classical order of accuracy of the algorithm."""
        raise NotImplementedError

    @property
    def step_function(self) -> Callable:
        """Return the cached device function that advances the solution."""
        return self.get_cached_output("step")

    @property
    def nonlinear_solver_function(self) -> Callable:
        """Return the cached nonlinear solver helper."""
        return self.get_cached_output("nonlinear_solver")

    @property
    def settings_dict(self) -> Dict[str, object]:
        """Return the configuration dictionary for the algorithm step."""
        return self.compile_settings.settings_dict

    @property
    def dxdt_function(self) -> Optional[Callable]:
        """Return the compiled device derivative function."""
        return self.compile_settings.dxdt_function

    @property
    def observables_function(self) -> Optional[Callable]:
        """Return the compiled device observables function."""
        return self.compile_settings.observables_function


    @property
    def get_solver_helper_fn(self) -> Optional[Callable]:
        """Return the helper factory used to build solver device functions.

        Returns
        -------
        Callable or None
            Callable that yields device helpers for solver construction when
            available.
        """
        return self.compile_settings.get_solver_helper_fn

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return self.compile_settings.stage_count