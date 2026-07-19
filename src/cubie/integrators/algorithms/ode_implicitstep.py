"""Infrastructure for implicit integration step implementations.

Published Classes
-----------------
:class:`ImplicitStepConfig`
    Configuration container extending :class:`BaseStepConfig` with
    implicit-specific fields (beta, gamma, preconditioner order).

:class:`ODEImplicitStep`
    Abstract base for implicit algorithms. Owns a
    :class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    or
    :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    instance and delegates solver parameter updates.

See Also
--------
:class:`~cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep`
    Parent factory class.
:class:`~cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep`
    Explicit counterpart.
:class:`~cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
    Nonlinear solver consumed by implicit steps.
"""

from abc import abstractmethod
from typing import Callable, Optional, Union, Set

from attrs import define, field, validators
from numpy import ndarray

from cubie._utils import inrangetype_validator, is_device_validator
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
    StepControlDefaults,
)


@define
class ImplicitStepConfig(BaseStepConfig):
    """Configuration settings for implicit integration steps.

    Parameters
    ----------
    beta
        Implicit integration coefficient applied to the stage derivative.
    gamma
        Implicit integration coefficient applied to the mass matrix product.
    preconditioner_order
        Order of the truncated Neumann preconditioner.

    Notes
    -----
    The mass matrix is not an algorithm parameter: it belongs to the
    ODE system, and mass-consuming solver helpers read it from the
    system when generated through ``get_solver_helper_fn``.
    """

    _beta: float = field(
        default=1.0, validator=inrangetype_validator(float, 0, 1)
    )
    _gamma: float = field(
        default=1.0, validator=inrangetype_validator(float, 0, 1)
    )
    preconditioner_order: int = field(
        default=2, validator=inrangetype_validator(int, 1, 32)
    )
    preconditioner_type: Union[str, list] = field(
        default="neumann",
    )
    solver_function = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )

    @property
    def solver_n(self) -> int:
        """Return the nonlinear solver's vector length."""
        return self.n

    @property
    def preconditioner_is_chained(self) -> bool:
        """Return whether the preconditioner resolves to a chain.

        Single strings and one-element lists resolve to a bare
        preconditioner; two-element lists compose as ``P1(P0(v))``
        with the chained (``chain_scratch``-carrying) signature.
        """
        return (
            isinstance(self.preconditioner_type, (list, tuple))
            and len(self.preconditioner_type) == 2
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
                "beta": self.beta,
                "gamma": self.gamma,
                "preconditioner_order": self.preconditioner_order,
                "preconditioner_type": self.preconditioner_type,
                "get_solver_helper_fn": self.get_solver_helper_fn,
            }
        )
        return settings_dict


class ODEImplicitStep(BaseAlgorithmStep):
    """Base helper for implicit integration algorithms."""

    # Union of parameters accepted by all linear solver types.
    # Params not applicable to the chosen solver are silently
    # ignored during construction.
    _LINEAR_SOLVER_PARAMS = frozenset(
        {
            "linear_correction_type",
            "krylov_atol",
            "krylov_rtol",
            "krylov_max_iters",
            # MR buffer locations
            "preconditioned_vec_location",
            "temp_location",
            # BiCGSTAB buffer locations
            "r0_hat_location",
            "p_location",
            "v_location",
            "tmp_location",
            "s_hat_location",
        }
    )

    # Parameters accepted by NewtonKrylov
    _NEWTON_KRYLOV_PARAMS = frozenset(
        {
            "newton_atol",
            "newton_rtol",
            "newton_max_iters",
            "newton_damping",
            "newton_max_backtracks",
            "delta_location",
            "residual_location",
            "residual_temp_location",
            "stage_base_bt_location",
        }
    )

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
            Optional solver parameters (krylov_atol, krylov_max_iters,
            newton_rtol, etc.). None values are ignored and defaults
            from solver config classes are used. ``newton_norm``
            supplies a :class:`CorrectionNorm` for Newton solves;
            when absent the solver builds its default.
        """
        super().__init__(config, _controller_defaults)

        if solver_type not in ["newton", "linear"]:
            raise ValueError(
                f"solver_type must be 'newton' or 'linear', got '{solver_type}'"
            )

        newton_norm = kwargs.pop("newton_norm", None)

        # Extract kwargs for each solver, filtering None values
        linear_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._LINEAR_SOLVER_PARAMS and v is not None
        }
        newton_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._NEWTON_KRYLOV_PARAMS and v is not None
        }

        correction_type = linear_kwargs.pop(
            "linear_correction_type", "minimal_residual"
        )
        solver_n = config.solver_n

        if correction_type == "bicgstab":
            linear_solver = BiCGSTABSolver(
                precision=config.precision,
                n=solver_n,
                **linear_kwargs,
            )
        else:
            linear_solver = MRLinearSolver(
                precision=config.precision,
                n=solver_n,
                linear_correction_type=correction_type,
                **linear_kwargs,
            )

        if solver_type == "newton":
            self.solver = NewtonKrylov(
                precision=config.precision,
                n=solver_n,
                linear_solver=linear_solver,
                norm=newton_norm,
                **newton_kwargs,
            )
        else:
            self.solver = linear_solver

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
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

        evaluate_f = config.evaluate_f
        numba_precision = config.numba_precision
        n = config.n
        evaluate_observables = config.evaluate_observables
        evaluate_driver_at_t = config.evaluate_driver_at_t
        n_drivers = config.n_drivers
        solver_function = config.solver_function

        return self.build_step(
            evaluate_f,
            evaluate_observables,
            evaluate_driver_at_t,
            solver_function,
            numba_precision,
            n,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build and return the implicit step device function.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating the ODE right-hand side f(t, y).
        evaluate_observables
            Device function for evaluating observables.
        evaluate_driver_at_t
            Optional device function evaluating drivers at arbitrary times.
        solver_function
            Device function for running internal solver.
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

    def build_implicit_helpers(self) -> None:
        """Construct the nonlinear solver chain used by implicit methods.

        Populates the owned solver with operator, preconditioner, and
        residual device functions, then stores the compiled solver
        function in compile settings.
        """

        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        preconditioner_order = config.preconditioner_order

        get_fn = config.get_solver_helper_fn

        # Get device functions from ODE system
        preconditioner = get_fn(
            "preconditioner",
            preconditioner_type=config.preconditioner_type,
            solver_beta=beta,
            solver_gamma=gamma,
            preconditioner_order=preconditioner_order,
        )
        residual = get_fn(
            "stage_residual",
            solver_beta=beta,
            solver_gamma=gamma,
            preconditioner_order=preconditioner_order,
        )
        operator = get_fn(
            "linear_operator",
            solver_beta=beta,
            solver_gamma=gamma,
            preconditioner_order=preconditioner_order,
        )

        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            preconditioner_is_chained=(
                config.preconditioner_is_chained
            ),
            residual_function=residual,
            n=config.solver_n,
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
    def preconditioner_order(self) -> int:
        """Return the order of the Neumann preconditioner."""

        return int(self.compile_settings.preconditioner_order)

    @property
    def preconditioner_type(self) -> Union[str, list]:
        """Return the type of preconditioner used by the linear solver."""
        return self.compile_settings.preconditioner_type

    @property
    def krylov_atol(self) -> ndarray:
        """Return the absolute tolerance array for linear solve."""
        return self.solver.krylov_atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return the relative tolerance array for linear solve."""
        return self.solver.krylov_rtol

    @property
    def krylov_max_iters(self) -> int:
        """Return the maximum number of linear iterations allowed."""
        return int(self.solver.krylov_max_iters)

    @property
    def linear_correction_type(self) -> str:
        """Return the linear correction strategy identifier."""
        return self.solver.linear_correction_type

    @property
    def newton_atol(self) -> Optional[ndarray]:
        """Return the Newton absolute tolerance array."""
        return getattr(self.solver, "newton_atol", None)

    @property
    def newton_rtol(self) -> Optional[ndarray]:
        """Return the Newton relative tolerance array."""
        return getattr(self.solver, "newton_rtol", None)

    @property
    def newton_max_iters(self) -> Optional[int]:
        """Return the maximum allowed Newton iterations."""
        val = getattr(self.solver, "newton_max_iters", None)
        return int(val) if val is not None else None

    @property
    def newton_damping(self) -> Optional[float]:
        """Return the Newton damping factor."""
        return getattr(self.solver, "newton_damping", None)

    @property
    def newton_max_backtracks(self) -> Optional[int]:
        """Return the maximum number of Newton backtracking steps."""
        val = getattr(self.solver, "newton_max_backtracks", None)
        return int(val) if val is not None else None

    @property
    def settings_dict(self) -> dict:
        """Return merged algorithm and solver settings.

        Combines implicit step configuration (beta, gamma, etc.)
        with solver settings (Newton and linear solver parameters).

        Returns
        -------
        dict
            Merged configuration dictionary containing:
            - Base step settings (n, n_drivers, precision) from BaseStepConfig
            - Implicit step settings (beta, gamma, preconditioner_order,
              get_solver_helper_fn) from ImplicitStepConfig
            - Solver settings (newton_atol, krylov_rtol, etc.)
              from NewtonKrylov or LinearSolverBase
            - All buffer location parameters from solver hierarchy
        """
        settings = super().settings_dict
        settings.update(self.solver.settings_dict)
        return settings
