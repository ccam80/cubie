"""Infrastructure for implicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Union

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)

@attrs.define
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for implicit integration steps.

    Parameters
    ----------
    beta
        Implicit integration coefficient applied to the stage derivative.
    gamma
        Implicit integration coefficient applied to the mass matrix product.
    M
        Mass matrix used when evaluating residuals and Jacobian actions.
    preconditioner_order
        Order of the truncated Neumann preconditioner.
    linsolve_tolerance
        Linear solver tolerance used by the Krylov iteration.
    max_linear_iters
        Maximum iterations permitted for the linear solver.
    linear_correction_type
        Identifier controlling the linear correction operator.
    nonlinear_tolerance
        Convergence tolerance for the Newton iteration.
    max_newton_iters
        Maximum iterations permitted for the Newton solver.
    newton_damping
        Damping factor applied within Newton updates.
    newton_max_backtracks
        Maximum number of backtracking steps within Newton updates.
    """

    _beta: float = attrs.field(default=1.0)
    _gamma: float = attrs.field(default=1.0)
    M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
    preconditioner_order: int = attrs.field(default=1)
    _linsolve_tolerance: float = attrs.field(default=1e-3)
    max_linear_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767),
    )
    linear_correction_type: str = attrs.field(default="minimal_residual")

    _nonlinear_tolerance: float = attrs.field(default=1e-3)
    max_newton_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767),
    )
    _newton_damping: float = attrs.field(default=0.5)
    newton_max_backtracks: int = attrs.field(default=10)

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""
        return self.precision(self._gamma)

    @property
    def linsolve_tolerance(self) -> float:
        """Return the linear solver tolerance."""
        return self.precision(self._linsolve_tolerance)

    @property
    def nonlinear_tolerance(self) -> float:
        """Return the nonlinear solver tolerance."""
        return self.precision(self._nonlinear_tolerance)

    @property
    def newton_damping(self) -> float:
        """Return the Newton damping factor."""
        return self.precision(self._newton_damping)

    @property
    def settings_dict(self) -> dict:
        """Return configuration fields as a dictionary."""

        settings_dict = super().settings_dict
        settings_dict.update(
            {
                'beta': self.beta,
                'gamma': self.gamma,
                'M': self.M,
                'preconditioner_order': self.preconditioner_order,
                'linsolve_tolerance': self.linsolve_tolerance,
                'max_linear_iters': self.max_linear_iters,
                'linear_correction_type': self.linear_correction_type,
                'nonlinear_tolerance': self.nonlinear_tolerance,
                'max_newton_iters': self.max_newton_iters,
                'newton_damping': self.newton_damping,
                'newton_max_backtracks': self.newton_max_backtracks,
                'get_solver_helper_fn': self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    def __init__(self, config: ImplicitStepConfig) -> None:
        """Initialise the implicit step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the implicit step.
        """

        super().__init__(config)

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step and nonlinear solver.
        """

        solver_fn = self.build_implicit_helpers()
        config = self.compile_settings
        dxdt_fn = config.dxdt_function
        numba_precision = config.numba_precision
        n = config.n
        observables_function = config.observables_function

        return self.build_step(
            solver_fn,
            dxdt_fn,
            observables_function,
            numba_precision,
            n,
        )

    @abstractmethod
    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        numba_precision: type,
        n: int,
    ) -> StepCache:
        """Build and return the implicit step device function.

        Parameters
        ----------
        solver_fn
            Device nonlinear solver produced by ``build_implicit_helpers``.
        dxdt_fn
            Device derivative function for the ODE system.
        observables_function
            Device observable computation helper.
        numba_precision
            Numba precision for compiled device buffers.
        n
            Dimension of the state vector.

        Returns
        -------
        StepCache
            Container holding the device step implementation.
        """

        raise NotImplementedError

    def build_implicit_helpers(self) -> Callable:
        """Construct the nonlinear solver chain used by implicit methods.

        Returns
        -------
        Callable
            Nonlinear solver function compiled for the configured implicit
            scheme.
        """

        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn
    
        preconditioner = get_fn(
                    'neumann_preconditioner',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        if self.is_multistage:
            residual = get_fn(
                    'stage_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order
            )
        else:
            residual = get_fn(
                    'end_residual',
                    beta=beta,
                    gamma=gamma,
                    mass=mass,
                    preconditioner_order=preconditioner_order,
            )
        operator = get_fn(
                'linear_operator',
                beta=beta,
                gamma=gamma,
                mass=mass,
                preconditioner_order=preconditioner_order)

        linsolve_tolerance = config.linsolve_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=linsolve_tolerance,
                                              max_iters=max_linear_iters)

        nonlinear_tolerance = config.nonlinear_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=nonlinear_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
        )
        return nonlinear_solver

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` to indicate the algorithm is implicit."""
        return True

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""

        return self.compile_settings.beta

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""

        return self.compile_settings.gamma

    @property
    def mass_matrix(self):
        """Return the mass matrix used by the implicit scheme."""

        return self.compile_settings.M

    @property
    def preconditioner_order(self) -> int:
        """Return the order of the Neumann preconditioner."""

        return int(self.compile_settings.preconditioner_order)

    @property
    def linsolve_tolerance(self) -> float:
        """Return the tolerance used for the linear solve."""

        return self.compile_settings.linsolve_tolerance

    @property
    def max_linear_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""

        return int(self.compile_settings.max_linear_iters)

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""

        return self.compile_settings.linear_correction_type

    @property
    def nonlinear_tolerance(self) -> float:
        """Return the Newton solve tolerance."""

        return self.compile_settings.nonlinear_tolerance

    @property
    def max_newton_iters(self) -> int:
        """Return the maximum allowed Newton iterations."""

        return int(self.compile_settings.max_newton_iters)

    @property
    def newton_damping(self) -> float:
        """Return the Newton damping factor."""

        return self.compile_settings.newton_damping

    @property
    def newton_max_backtracks(self) -> int:
        """Return the maximum number of Newton backtracking steps."""

        return int(self.compile_settings.newton_max_backtracks)
