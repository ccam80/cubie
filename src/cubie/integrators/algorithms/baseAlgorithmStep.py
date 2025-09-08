"""Base class for inner "step" logic of a numerical integration algorithm.

This module provides the BaseAlgorithmStep class, which serves as the base
class for all inner "step" loops for numerical integration algorithms. It
includes the interface for implementing the inner loop logic only,
returning an integer code that indicates the success or failure of the step."""
from abc import abstractmethod
from typing import Callable

from cubie.CUDAFactory import CUDAFactory
import attrs

from cubie.integrators.algorithms.StepConfig import StepConfig
from cubie.integrators.matrix_free_solvers import linear_solver_factory, \
    newton_krylov_solver_factory
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling import LoopBufferSizes


@attrs.define
class StepCache:
    step: Callable = attrs.field()
    nonlinear_solver: Callable = attrs.field()

class BaseAlgorithmStep(CUDAFactory):
    """
    Base class for inner "step" logic of a numerical integration algorithm.

    This class handles building and caching of a step device function, which
    calculates the next state of the system given the current state, current
    time, and step-end time. The step device function modifies the state array
    in-place and returns a success or error code indicating that the step
    converged or failed.
    """

    def __init__(self,
                 buffer_sizes: LoopBufferSizes,
                 system: BaseODE,
                 linear_solver_operator_fn: Callable,
                 preconditioner_fn: Callable,
                 residual_fn: Callable,
                 compile_flags=None):
        config = StepConfig(buffer_sizes = buffer_sizes)
        self.system = system
        self.setup_compile_settings(config)


    def build(self):
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
        beta = self.compile_settings.beta
        gamma = self.compile_settings.gamma
        mass = self.compile_settings.M
        preconditioner_order = self.compile_settings.preconditioner_order
        multistage = self.compile_settings.multistage_residual_fn
        system = self.system
        n = system.sizes.states

        preconditioner = system.get_solver_helper(
                    'neumann_preconditioner',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        if multistage:
            residual = system.get_solver_helper(
                    'stage_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        else:
            residual = system.get_solver_helper(
                    'end_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order,
            )
        operator = system.get_solver_helper(
                'linear_operator',
                beta=beta,
                gamma=gamma,
                mass=mass,
                preconditioner_order=preconditioner_order)

        linsolve_tolerance = self.compile_settings.linsolve_tolerance
        max_linear_iters = self.compile_settings.max_linear_iters
        correction_type = self.compile_settings.linear_correction_type

        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=linsolve_tolerance,
                                              max_iters=max_linear_iters)

        nonlinear_tolerance = self.compile_settings.nonlinear_tolerance
        max_newton_iters = self.compile_settings.max_newton_iters
        newton_damping = self.compile_settings.newton_damping
        newton_max_backtracks = self.compile_settings.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
                residual_function=residual,
                linear_solver=linear_solver,
                n=n,
                tolerance=nonlinear_tolerance,
                max_iters=max_newton_iters,
                damping=newton_damping,
                max_backtracks=newton_max_backtracks)
        #TODO: Is it worth caching the intermediates? They all feed the
        # nonlinear solver, and none are used solo at this point.
        return nonlinear_solver

    @abstractmethod
    def build_step(self, nonlinear_solver_fn):
        """Construct the step function as a cuda Device function.

        The function must have the signature:
        step(states, params, drivers, t, dt, temp_mem) -> int32, where the
        return type is an integer code indicating success or failure
        according to SolverRetCodes"""
        #return step_function