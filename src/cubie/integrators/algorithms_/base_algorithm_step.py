"""Base class for inner "step" logic of a numerical integration algorithm.

This module provides the BaseAlgorithmStep class, which serves as the base
class for all inner "step" loops for numerical integration algorithms. It
includes the interface for implementing the inner loop logic only,
returning an integer code that indicates the success or failure of the step."""

from abc import abstractmethod, ABC
from typing import Callable, Optional

import attrs
import numpy as np
from attrs import validators
import numba

from cubie import getype_validator
from cubie._utils import is_device_validator
from cubie.CUDAFactory import CUDAFactory
from cubie.cudasim_utils import from_dtype as simsafe_dtype


@attrs.define
class BaseStepConfig(ABC):
    """Configuration settings for a single integration step.
    """
    precision: type = attrs.field(
        default=np.float32,
        validator=attrs.validators.in_([np.float16, np.float32, np.float64])
    )
    n: int = attrs.field(
        default=1,
        validator=getype_validator(int, 1)
    )
    dxdt_function: Optional[Callable] = attrs.field(
        default=None,
        validator=validators.optional(is_device_validator)
    )

    @property
    def numba_precision(self) -> type:
        """Returns numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Returns simulator safe precision."""
        return simsafe_dtype(self.precision)


@attrs.define
class StepCache:
    step: Callable = attrs.field(validator=is_device_validator)
    nonlinear_solver: Optional[Callable] = attrs.field(
           default=None,
           validator=validators.optional(is_device_validator),
    )

class BaseAlgorithmStep(CUDAFactory):
    """
    Base class for inner "step" logic of a numerical integration algorithm.

    This class provides default update behaviour and properties for a
    unified interface and inherits build and cache logic from CUDAFactory.

    Algorithm algorithms_ handle the "inner" logic of an ODE integration,
    estimating state at some future time given current state and parameters.
    All algorithms_ should return an integer code indicating success or failure.
    """

    def __init__(self,
                 config: BaseStepConfig):
        super().__init__()
        self.setup_compile_settings(config)

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface.

        This method will invalidate the cache if an update is successful.
        Use silent=True when doing bulk updates with other component parameters
        to suppress warnings about unrecognized keys.

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
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised

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
        """Number of precision elements of shared memory needed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def local_scratch_required(self) -> int:
        """How many elements of temporary local memory are required."""
        raise NotImplementedError

    @property
    @abstractmethod
    def persistent_local_required(self) -> int:
        """Number of local elements that must persist between calls"""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError

    @property
    def step_function(self) -> Callable:
        """Get the step function.""

        This function performs a single integration step, returning an
        integer code indicating success or failure.

        Returns
        -------
        Callable
            Device function that performs a single integration step.
        """
        return self.get_cached_output("step")

    @property
    def nonlinear_solver_function(self) -> Callable:
        """Get the nonlinear solver function."""
        return self.get_cached_output("nonlinear_solver")