"""Infrastructure for implicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional, Union, Set

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator, is_device_validator
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
    solver_function = attrs.field(
        default=None,
        validator=attrs.validators.optional(is_device_validator),
        eq=False
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

    # Parameters accepted by LinearSolver
    _LINEAR_SOLVER_PARAMS = frozenset({
        'linear_correction_type',
        'krylov_tolerance',
        'max_linear_iters',
        'preconditioned_vec_location',
        'temp_location',
    })

    # Parameters accepted by NewtonKrylov
    _NEWTON_KRYLOV_PARAMS = frozenset({
        'newton_tolerance',
        'max_newton_iters',
        'newton_damping',
        'newton_max_backtracks',
        'delta_location',
        'residual_location',
        'residual_temp_location',
        'stage_base_bt_location',
    })

    def __init__(
        self,
        config: ImplicitStepConfig,
        _controller_defaults: StepControlDefaults,
        solver_type: str = "newton",
        **kwargs,
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
        **kwargs
            Optional solver parameters (krylov_tolerance, max_linear_iters,
            newton_tolerance, etc.). None values are ignored and defaults
            from solver config classes are used.
        """
        super().__init__(config, _controller_defaults)

        if solver_type not in ['newton', 'linear']:
            raise ValueError(
                f"solver_type must be 'newton' or 'linear', got '{solver_type}'"
            )

        # Extract kwargs for each solver, filtering None values
        linear_kwargs = {
            k: v for k, v in kwargs.items()
            if k in self._LINEAR_SOLVER_PARAMS and v is not None
        }
        newton_kwargs = {
            k: v for k, v in kwargs.items()
            if k in self._NEWTON_KRYLOV_PARAMS and v is not None
        }

        linear_solver = LinearSolver(
            precision=config.precision,
            n=config.n,
            **linear_kwargs,
        )

        if solver_type == 'newton':
            self.solver = NewtonKrylov(
                precision=config.precision,
                n=config.n,
                linear_solver=linear_solver,
                **newton_kwargs,
            )
        else:
            self.solver = linear_solver

    def register_buffers(self) -> None:
        """ Register buffers with buffer_registry."""
        pass

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
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        recognized = set()

        recognized |= self.solver.update(all_updates, silent=True)

        all_updates["solver_function"] = self.solver.device_function

        recognized |= super().update(all_updates, silent=True)

        return recognized

    def build(self) -> StepCache:
        """Create and cache the device helpers for the implicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step and nonlinear solver.
        """
        config = self.compile_settings
        self.build_implicit_helpers()

        dxdt_fn = config.dxdt_function
        numba_precision = config.numba_precision
        n = config.n
        observables_function = config.observables_function
        driver_function = config.driver_function
        n_drivers = config.n_drivers
        solver_function = config.solver_function

        return self.build_step(
            dxdt_fn,
            observables_function,
            driver_function,
            solver_function,
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
        solver_function: Callable,
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
            Device function for evaluating observables.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        solver_function
            Device function for running internal solver
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

        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            residual_function=residual,
            n = self.compile_settings.n
        )

        self.update_compile_settings(
                solver_function=self.solver.device_function
        )

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
        return self.solver.krylov_tolerance

    @property
    def max_linear_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""
        return int(self.solver.max_linear_iters)

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""
        return self.solver.linear_correction_type

    @property
    def newton_tolerance(self) -> Optional[float]:
        """Return the Newton solve tolerance."""
        return getattr(self.solver, 'newton_tolerance', None)

    @property
    def max_newton_iters(self) -> Optional[int]:
        """Return the maximum allowed Newton iterations."""
        val = getattr(self.solver, 'max_newton_iters', None)
        return int(val) if val is not None else None

    @property
    def newton_damping(self) -> Optional[float]:
        """Return the Newton damping factor."""
        return getattr(self.solver, 'newton_damping', None)

    @property
    def newton_max_backtracks(self) -> Optional[int]:
        """Return the maximum number of Newton backtracking steps."""
        val = getattr(self.solver, 'newton_max_backtracks', None)
        return int(val) if val is not None else None

    @property
    def settings_dict(self) -> dict:
        """Return merged algorithm and solver settings.

        Combines implicit step configuration (beta, gamma, M, etc.)
        with solver settings (Newton and linear solver parameters).

        Returns
        -------
        dict
            Merged configuration dictionary containing:
            - Base step settings (n, n_drivers, precision) from BaseStepConfig
            - Implicit step settings (beta, gamma, M, preconditioner_order,
              get_solver_helper_fn) from ImplicitStepConfig
            - Solver settings (newton_tolerance, krylov_tolerance, etc.)
              from NewtonKrylov or LinearSolver
            - All buffer location parameters from solver hierarchy
        """
        settings = super().settings_dict
        settings.update(self.solver.settings_dict)
        return settings
