"""Infrastructure for implicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional, Union

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator, gttype_validator
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
    LinearSolverConfig,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
    NewtonKrylovConfig,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache, StepControlDefaults,
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
    krylov_tolerance
        Linear solver tolerance used by the Krylov iteration.
    max_linear_iters
        Maximum iterations permitted for the linear solver.
    linear_correction_type
        Identifier controlling the linear correction operator.
    newton_tolerance
        Convergence tolerance for the Newton iteration.
    max_newton_iters
        Maximum iterations permitted for the Newton solver.
    newton_damping
        Damping factor applied within Newton updates.
    newton_max_backtracks
        Maximum number of backtracking steps within Newton updates.
    """

    _beta: float = attrs.field(
        default=1.0,
        validator=inrangetype_validator(float, 0, 1)
    )
    _gamma: float = attrs.field(
        default=1.0,
        validator=inrangetype_validator(float, 0, 1)
    )
    M: Union[np.ndarray, sp.Matrix] = attrs.field(default=sp.eye(1))
    preconditioner_order: int = attrs.field(
        default=1,
        validator=inrangetype_validator(int, 1, 32)
    )
    _krylov_tolerance: float = attrs.field(
        default=1e-3,
        validator=gttype_validator(float, 0))
    max_linear_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767),
    )
    linear_correction_type: str = attrs.field(default="minimal_residual")

    _newton_tolerance: float = attrs.field(
        default=1e-3,
        validator=gttype_validator(float, 0)
    )
    max_newton_iters: int = attrs.field(
        default=100,
        validator=inrangetype_validator(int, 1, 32767),
    )
    _newton_damping: float = attrs.field(
        default=0.5,
        validator=inrangetype_validator(float, 0, 1)
    )

    newton_max_backtracks: int = attrs.field(
        default=10,
        validator=inrangetype_validator(int, 1, 32767)
    )

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""
        return self.precision(self._gamma)

    @property
    def krylov_tolerance(self) -> float:
        """Return the linear solver tolerance."""
        return self.precision(self._krylov_tolerance)

    @property
    def newton_tolerance(self) -> float:
        """Return the nonlinear solver tolerance."""
        return self.precision(self._newton_tolerance)

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
                'krylov_tolerance': self.krylov_tolerance,
                'max_linear_iters': self.max_linear_iters,
                'linear_correction_type': self.linear_correction_type,
                'newton_tolerance': self.newton_tolerance,
                'max_newton_iters': self.max_newton_iters,
                'newton_damping': self.newton_damping,
                'newton_max_backtracks': self.newton_max_backtracks,
                'get_solver_helper_fn': self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    def __init__(self,
                 config: ImplicitStepConfig,
                 _controller_defaults: StepControlDefaults) -> None:
        """Initialise the implicit step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the implicit step.
        _controller_defaults
           Per-algorithm default runtime collaborators, such as step
           controllers and matrix-free solvers.
        """

        super().__init__(config, _controller_defaults)
        
        # Create LinearSolver instance
        linear_config = LinearSolverConfig(
            precision=config.precision,
            n=config.n,
            correction_type=config.linear_correction_type,
            tolerance=config.krylov_tolerance,
            max_iters=config.max_linear_iters,
        )
        self._linear_solver = LinearSolver(linear_config)
        
        # Create NewtonKrylov instance
        newton_config = NewtonKrylovConfig(
            precision=config.precision,
            n=config.n,
            linear_solver=self._linear_solver,
            tolerance=config.newton_tolerance,
            max_iters=config.max_newton_iters,
            damping=config.newton_damping,
            max_backtracks=config.newton_max_backtracks,
        )
        self._newton_solver = NewtonKrylov(newton_config)

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
        driver_function = config.driver_function
        n_drivers = config.n_drivers

        return self.build_step(
            solver_fn,
            dxdt_fn,
            observables_function,
            driver_function,
            numba_precision,
            n,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
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
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision for compiled device buffers.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

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

        get_fn = config.get_solver_helper_fn
    
        # Get device functions from ODE system
        preconditioner = get_fn(
            'neumann_preconditioner',
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order
        )
        residual = get_fn(
            'stage_residual',
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order
        )
        operator = get_fn(
            'linear_operator',
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order
        )
        
        # Update solvers with device functions
        self._linear_solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
        )
        self._newton_solver.update(
            residual_function=residual,
        )
        
        # Return device function
        return self._newton_solver.device_function

    @property
    def solver_shared_elements(self) -> int:
        """Return shared scratch dedicated to the Newton--Krylov solver."""

        return self._newton_solver.shared_buffer_size

    @property
    def solver_local_elements(self) -> int:
        """Implicit solvers return zero persistent local elements."""

        return 0

    @property
    def algorithm_shared_elements(self) -> int:
        """Implicit base class does not reserve extra shared scratch."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Implicit base class does not reserve persistent locals."""

        return 0

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
    def krylov_tolerance(self) -> float:
        """Return the tolerance used for the linear solve."""

        return self.compile_settings.krylov_tolerance

    @property
    def max_linear_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""

        return int(self.compile_settings.max_linear_iters)

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""

        return self.compile_settings.linear_correction_type

    @property
    def newton_tolerance(self) -> float:
        """Return the Newton solve tolerance."""

        return self.compile_settings.newton_tolerance

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
