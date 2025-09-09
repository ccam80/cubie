"""Base class for inner "step" logic of a numerical integration algorithm.

This module provides the BaseAlgorithmStep class, which serves as the base
class for all inner "step" loops for numerical integration algorithms. It
includes the interface for implementing the inner loop logic only,
returning an integer code that indicates the success or failure of the step."""
from abc import abstractmethod
from typing import Callable

from cubie.CUDAFactory import CUDAFactory
import attrs

from cubie._utils import in_attr
from cubie.integrators.steps.BaseStepConfig import BaseStepConfig
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling import LoopBufferSizes


@attrs.define
class StepCache:
    step: Callable = attrs.field()
    nonlinear_solver: Callable = attrs.field()

class BaseAlgorithmStep(CUDAFactory):
    """
    Base class for inner "step" logic of a numerical integration algorithm.

    This class provides default update behaviour and properties for a
    unified interface and inherits build and cache logic from CUDAFactory.

    Algorithm steps handle the "inner" logic of an ODE integration,
    estimating state at some future time given current state and parameters.
    All steps should return an integer code indicating success or failure.
    """

    def __init__(self,
                 buffer_sizes: LoopBufferSizes,
                 system: BaseODE,
                 compile_flags=None):
        config = BaseStepConfig(buffer_sizes = buffer_sizes)
        self.system = system
        self.setup_compile_settings(config)

    @abstractmethod
    def build(self) -> StepCache:
        if self.compile_settings.style == 'implicit':
            nonlinear_solver = self.build_implicit_helpers()
        else:
            nonlinear_solver = None
        step_func = self.build_step(nonlinear_solver)

        return StepCache(nonlinear_solver=nonlinear_solver,
                         step=step_func)

    def build_implicit_helpers(self):
        """Construct the matrix-free solver for implicit methods.

        Constructs a chain of device functions that pieces together the
        matrix-free solvers for implicit methods.

        Returns
        -------
        callable
            Device function that performs the matrix-free solve operation.
        """
        return NotImplementedError("This method isn't provided for explicit "
                                   "methods")

    @abstractmethod
    def build_step(self, nonlinear_solver_fn):
        """Construct the step function as a cuda Device function.

        The function must have the signature:
        step(states, params, drivers, t, dt, temp_mem) -> int32, where the
        return type is an integer code indicating success or failure
        according to SolverRetCodes"""
        #return step_function

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
        for key, value in updates_dict.items():
            if in_attr(key, self.compile_settings.loop_step_config):
                setattr(self.compile_settings, key, value)
                recognised.add(key)

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised
