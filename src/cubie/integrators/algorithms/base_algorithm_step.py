"""Base classes and shared configuration for integration step factories."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Set
import warnings

import attrs
import numpy as np
from attrs import validators
import numba

from cubie._utils import (
    PrecisionDtype,
    getype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
)
from cubie.CUDAFactory import CUDAFactory
from cubie.cuda_simsafe import from_dtype as simsafe_dtype

# Define all possible algorithm step parameters across all algorithm types
ALL_ALGORITHM_STEP_PARAMETERS = {
    # Base parameters
    'precision', 'n', 'dxdt_function', 'observables_function',
    'get_solver_helper_fn',
    'observables_function',
    # Explicit algorithm parameters
    'dt',
    # Implicit algorithm parameters
    'beta', 'gamma', 'M', 'preconditioner_order', 'linsolve_tolerance',
    'max_linear_iters', 'linear_correction_type', 'nonlinear_tolerance',
    'max_newton_iters', 'newton_damping', 'newton_max_backtracks'
}
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
    dxdt_function
        Device function that evaluates the system right-hand side.
    observables_function
        Device function that evaluates the system observables.
    get_solver_helper_fn
        Optional callable that returns device helpers required by the
        nonlinear solver construction.
    observables_function
        Device function computing system observables.
    """

    precision: PrecisionDtype = attrs.field(
        default=np.float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    n: int = attrs.field(default=1, validator=getype_validator(int, 1))
    dxdt_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator),
    )
    observables_function: Optional[Callable] = attrs.field(
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
            "precision": self.precision,
        }

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

    def __init__(self, config: BaseStepConfig) -> None:
        """Initialise the algorithm step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the algorithm step.

        Returns
        -------
        None
            This constructor updates internal configuration state.
        """

        super().__init__()
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
    def precision(self) -> PrecisionDtype:
        """Return the configured numerical precision."""

        return self.compile_settings.precision

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
    @abstractmethod
    def shared_memory_required(self) -> int:
        """Return the precision-entry count of shared memory required."""
        raise NotImplementedError

    @property
    @abstractmethod
    def local_scratch_required(self) -> int:
        """Return the precision-entry count of local scratch required."""
        raise NotImplementedError

    @property
    @abstractmethod
    def persistent_local_required(self) -> int:
        """Return the persistent local precision-entry requirement."""
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
