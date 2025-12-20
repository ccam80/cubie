"""Infrastructure for implicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional, Union, Set

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator
from cubie.buffer_registry import buffer_registry
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
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

    @property
    def beta(self) -> float:
        """Return the implicit integration beta coefficient."""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """Return the implicit integration gamma coefficient."""
        return self.precision(self._gamma)

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
                'get_solver_helper_fn': self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    def __init__(
        self,
        config: ImplicitStepConfig,
        _controller_defaults: StepControlDefaults,
        solver_type: str = 'newton',
        krylov_tolerance: float = 1e-3,
        max_linear_iters: int = 100,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-3,
        max_newton_iters: int = 100,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 10,
    ) -> None:
        """Initialise the implicit step with its configuration.

        Parameters
        ----------
        config
            Configuration describing the implicit step.
        _controller_defaults
           Per-algorithm default runtime collaborators.
        solver_type
            Type of solver to create: 'newton' or 'linear'.
        krylov_tolerance
            Tolerance used by the linear solver.
        max_linear_iters
            Maximum iterations permitted for the linear solver.
        linear_correction_type
            Identifier for the linear correction strategy.
        newton_tolerance
            Convergence tolerance for the Newton iteration.
        max_newton_iters
            Maximum iterations permitted for the Newton solver.
        newton_damping
            Damping factor applied within Newton updates.
        newton_max_backtracks
            Maximum number of backtracking steps within the Newton solver.
        """
        # Validate solver_type
        if solver_type not in ['newton', 'linear']:
            raise ValueError(
                f"solver_type must be 'newton' or 'linear', got '{solver_type}'"
            )
        
        super().__init__(config, _controller_defaults)
        
        # Create LinearSolver instance with passed parameters
        linear_solver = LinearSolver(
            precision=config.precision,
            n=config.n,
            correction_type=linear_correction_type,
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
        )
        
        # Create solver based on solver_type
        if solver_type == 'newton':
            # Create NewtonKrylov with LinearSolver
            self.solver = NewtonKrylov(
                precision=config.precision,
                n=config.n,
                linear_solver=linear_solver,
                newton_tolerance=newton_tolerance,
                max_newton_iters=max_newton_iters,
                newton_damping=newton_damping,
                newton_max_backtracks=newton_max_backtracks,
            )
        else:  # solver_type == 'linear'
            # Store LinearSolver directly
            self.solver = linear_solver

    def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
        """Update algorithm and owned solver parameters.
        
        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of parameter names to new values.
        silent : bool, default=False
            Suppress warnings for unrecognized parameters.
        **kwargs
            Additional parameters to update.
        
        Returns
        -------
        set[str]
            Names of parameters that were successfully recognized.
        
        Notes
        -----
        Delegates solver parameters to owned solver instance.
        Invalidates step cache only if solver cache was invalidated.
        """
        # Merge updates
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)
        
        if not all_updates:
            return set()
        
        # Delegate to solver with full dict
        recognized = set()
        solver_recognized = self.solver.update(all_updates, silent=True)
        recognized.update(solver_recognized)
        
        # Check if solver cache was invalidated
        if not self.solver.cache_valid:
            self.invalidate_cache()
        
        # Update buffer registry with full dict
        buffer_recognized = buffer_registry.update(
            self, updates_dict=all_updates, silent=True
        )
        recognized.update(buffer_recognized)
        
        # Update algorithm compile settings with full dict
        algo_recognized = self.update_compile_settings(
            updates_dict=all_updates, silent=silent
        )
        recognized.update(algo_recognized)
        
        return recognized

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step and nonlinear solver.
        """
        solver_fn = self.build_implicit_helpers()
        config = self.compile_settings
        
        # Store solver device function reference for cache comparison
        self.update_compile_settings(solver_device_function=solver_fn)
        
        dxdt_fn = config.dxdt_function
        numba_precision = config.numba_precision
        n = config.n
        observables_function = config.observables_function
        driver_function = config.driver_function
        n_drivers = config.n_drivers
        
        # build_step no longer receives solver_fn parameter
        return self.build_step(
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
        
        Notes
        -----
        Subclasses access solver device function via self.solver.device_function.
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
        
        # Update solver with device functions
        # If solver is NewtonKrylov, it will delegate linear params to its linear_solver
        # If solver is LinearSolver, it will recognize linear params directly
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            residual_function=residual,
        )
        
        # Return device function
        return self.solver.device_function

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
        if hasattr(self.solver, 'krylov_tolerance'):
            return self.solver.krylov_tolerance
        # For NewtonKrylov, forward to nested linear_solver
        return self.solver.linear_solver.krylov_tolerance

    @property
    def max_linear_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""
        if hasattr(self.solver, 'max_linear_iters'):
            return int(self.solver.max_linear_iters)
        # For NewtonKrylov, forward to nested linear_solver
        return int(self.solver.linear_solver.max_linear_iters)

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""
        if hasattr(self.solver, 'correction_type'):
            return self.solver.correction_type
        # For NewtonKrylov, forward to nested linear_solver
        return self.solver.linear_solver.correction_type

    @property
    def newton_tolerance(self) -> float:
        """Return the Newton solve tolerance."""
        if hasattr(self.solver, 'newton_tolerance'):
            return self.solver.newton_tolerance
        raise AttributeError(
            f"{type(self.solver).__name__} does not have newton_tolerance"
        )

    @property
    def max_newton_iters(self) -> int:
        """Return the maximum allowed Newton iterations."""
        if hasattr(self.solver, 'max_newton_iters'):
            return int(self.solver.max_newton_iters)
        raise AttributeError(
            f"{type(self.solver).__name__} does not have max_newton_iters"
        )

    @property
    def newton_damping(self) -> float:
        """Return the Newton damping factor."""
        if hasattr(self.solver, 'newton_damping'):
            return self.solver.newton_damping
        raise AttributeError(
            f"{type(self.solver).__name__} does not have newton_damping"
        )

    @property
    def newton_max_backtracks(self) -> int:
        """Return the maximum number of Newton backtracking steps."""
        if hasattr(self.solver, 'newton_max_backtracks'):
            return int(self.solver.newton_max_backtracks)
        raise AttributeError(
            f"{type(self.solver).__name__} does not have newton_max_backtracks"
        )
