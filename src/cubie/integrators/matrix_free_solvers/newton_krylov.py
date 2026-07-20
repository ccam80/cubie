"""Newton--Krylov solver factories for matrix-free integrators.

This module wraps a linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver_base` to build
damped Newton iterations suitable for CUDA device execution.

Published Classes
-----------------
:class:`NewtonKrylovConfig`
    Attrs configuration for the Newton--Krylov solver factory.

:class:`NewtonKrylovCache`
    Cache container holding the compiled Newton--Krylov device
    function.

:class:`NewtonKrylov`
    CUDAFactory subclass that compiles a damped Newton--Krylov
    solver wrapping a :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`.

See Also
--------
:class:`~cubie.integrators.matrix_free_solvers.base_solver.MatrixFreeSolver`
    Parent factory providing norm and tolerance management.
:class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
    Inner linear solver used for the Newton correction equation.
:mod:`cubie.integrators.algorithms.ode_implicitstep`
    Implicit step base class that creates :class:`NewtonKrylov`
    instances.
"""

from math import sqrt as math_sqrt
from typing import Callable, Optional, Set, Dict, Any

from attrs import define, field, validators
from cubie.cuda_simsafe import cuda, int32
from numpy import finfo as np_finfo
from numpy import int32 as np_int32
from numpy import ndarray

from cubie._utils import (
    PrecisionDType,
    build_config,
    inrangetype_validator,
    is_device_validator,
)
from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDADispatcherCache
from cubie.cuda_simsafe import (
    activemask,
    all_sync,
    selp,
    any_sync,
)
from cubie.result_codes import CUBIE_RESULT_CODES

from cubie.integrators.matrix_free_solvers.linear_solver_base import (
    LinearSolverBase,
)
from cubie.integrators.norms import CorrectionNorm, DIRKCorrectionNorm


@define
class NewtonKrylovConfig(MatrixFreeSolverConfig):
    """Configuration for NewtonKrylov solver compilation.

    Attributes
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of state vectors.
    max_iters : int
        Maximum solver iterations permitted.
    norm_device_function : Optional[Callable]
        Compiled norm function for convergence checks.
    residual_function : Optional[Callable]
        Device function evaluating residuals.
    linear_solver_function : Optional[Callable]
        Device function for solving linear systems.
    newton_damping : float
        Step shrink factor for backtracking.
    newton_max_backtracks : int
        Extra damping attempts. Zero disables backtracking.
    delta_location : str
        Memory location for delta buffer.
    residual_location : str
        Memory location for residual buffer.
    residual_temp_location : str
        Memory location for residual_temp buffer.
    stage_base_bt_location : str
        Memory location for stage_base_bt buffer.
    krylov_iters_local_location : str
        Memory location for the single-element Krylov iteration
        counter buffer.

    Notes
    -----
    Tolerance arrays (newton_atol, newton_rtol) are managed by the solver's
    norm factory and accessed via NewtonKrylov.newton_atol/newton_rtol
    properties.
    """

    residual_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    linear_solver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    correction_norm_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    _newton_damping: float = field(
        default=0.5, validator=inrangetype_validator(float, 0, 1)
    )
    newton_max_backtracks: int = field(
        default=0, validator=inrangetype_validator(int, 0, 32767)
    )
    delta_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    residual_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    residual_temp_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_base_bt_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    krylov_iters_local_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def newton_damping(self) -> float:
        """Return damping factor in configured precision."""
        return self.precision(self._newton_damping)

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return Newton-Krylov configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary. Note: newton_atol and newton_rtol
            are not included here; access them via solver.newton_atol
            and solver.newton_rtol properties which delegate to the
            norm factory.
        """
        return {
            "newton_max_iters": self.max_iters,
            "newton_damping": self.newton_damping,
            "newton_max_backtracks": self.newton_max_backtracks,
            "delta_location": self.delta_location,
            "residual_location": self.residual_location,
            "residual_temp_location": self.residual_temp_location,
            "stage_base_bt_location": self.stage_base_bt_location,
            "krylov_iters_local_location": self.krylov_iters_local_location,
        }


@define
class NewtonKrylovCache(CUDADispatcherCache):
    """Cache container for NewtonKrylov outputs.

    Attributes
    ----------
    newton_krylov_solver : Callable
        Compiled CUDA device function for Newton-Krylov solving.
    """

    newton_krylov_solver: Callable = field(validator=is_device_validator)


class NewtonKrylov(MatrixFreeSolver):
    """Factory for Newton--Krylov solver device functions.

    Uses a matrix-free linear solver for each Newton correction.

    Parameters
    ----------
    precision : PrecisionDType
        Numerical precision for computations.
    n : int
        Size of state vectors.
    linear_solver : LinearSolverBase
        :class:`~cubie.integrators.matrix_free_solvers.linear_solver_base.LinearSolverBase`
        instance for solving linear systems.
    **kwargs
        Forwarded to :class:`NewtonKrylovConfig` and the norm
        factory. Includes prefixed tolerance parameters
        (``newton_atol``, ``newton_rtol``).

    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        linear_solver: LinearSolverBase,
        norm: Optional[CorrectionNorm] = None,
        **kwargs,
    ) -> None:
        """Initialize NewtonKrylov with parameters.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        n : int
            Size of state vectors.
        linear_solver : LinearSolverBase
            Inner linear solver.
        norm : CorrectionNorm, optional
            Correction norm. Defaults to DIRK scaling.
        **kwargs
            Newton solver settings.
        """

        if norm is None:
            norm = DIRKCorrectionNorm(
                precision=precision,
                n=n,
                instance_label="newton",
                **kwargs,
            )
        super().__init__(
            precision=precision,
            solver_type="newton",
            n=n,
            norm=norm,
            **kwargs,
        )

        config = build_config(
            NewtonKrylovConfig,
            required={
                "precision": precision,
                "n": n,
                "norm_device_function": self.norm.device_function,
                "correction_norm_function": (
                    self.norm.correction_device_function
                ),
            },
            instance_label="newton",
            **kwargs,
        )

        self.linear_solver = linear_solver
        self.setup_compile_settings(config)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        # Register buffers with buffer_registry
        config = self.compile_settings
        precision = config.precision
        backtracking_size = config.n if config.newton_max_backtracks else 0

        buffer_registry.register(
            "delta", self, config.n, config.delta_location, precision=precision
        )
        buffer_registry.register(
            "residual",
            self,
            config.n,
            config.residual_location,
            precision=precision,
        )
        buffer_registry.register(
            "residual_temp",
            self,
            backtracking_size,
            config.residual_temp_location,
            precision=precision,
        )
        buffer_registry.register(
            "stage_base_bt",
            self,
            backtracking_size,
            config.stage_base_bt_location,
            precision=precision,
        )
        buffer_registry.register(
            "krylov_iters_local",
            self,
            1,
            config.krylov_iters_local_location,
            precision=np_int32,
        )
        # Warm-started contraction estimate carried between solves.
        buffer_registry.register(
            "prev_theta",
            self,
            1,
            "local",
            persistent=True,
            precision=precision,
        )
        # Record the linear solver as a child at registration time so
        # clear_parent cascades reach it before this solver has built;
        # build refreshes the same named registration with real sizes.
        buffer_registry.register_child(
            self, self.linear_solver, name="linear_solver"
        )

    def build(self) -> NewtonKrylovCache:
        """Compile the Newton solver.

        Returns
        -------
        NewtonKrylovCache
            Compiled device function.
        """
        config = self.compile_settings

        # Extract parameters from config
        residual_function = config.residual_function
        linear_solver_fn = config.linear_solver_function
        scaled_norm_fn = config.norm_device_function
        correction_norm_fn = config.correction_norm_function

        n = config.n
        max_iters = int32(config.max_iters)
        use_backtracking = config.newton_max_backtracks > 0
        newton_damping = config.newton_damping
        # Include the full trial before the configured damped trials.
        max_backtracks = int32(config.newton_max_backtracks + 1)

        numba_precision = config.numba_precision
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        max_newton_iters_exceeded = int32(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
        )
        newton_backtracking_failed = int32(
            CUBIE_RESULT_CODES.NEWTON_BACKTRACKING_NO_SUITABLE_STEP
        )
        newton_divergence = int32(CUBIE_RESULT_CODES.NEWTON_DIVERGENCE)
        typed_one = numba_precision(1.0)
        typed_damping = numba_precision(newton_damping)
        typed_huge = numba_precision(float(np_finfo(config.precision).max))
        # Convergence and divergence constants follow OrdinaryDiffEq's
        # NLNewton: accept when eta * ||dz|| < kappa, bail out when the
        # contraction estimate exceeds two, and settle ties at the
        # floating-point stagnation limit by the update norm alone.
        kappa = numba_precision(0.01)
        first_iteration_bound = numba_precision(1.0e-5)
        theta_decay = numba_precision(0.3)
        theta_divergence_bound = numba_precision(2.0)
        stagnation_eps = numba_precision(
            100.0 * math_sqrt(float(np_finfo(config.precision).eps))
        )
        n_val = int32(n)

        # Get allocators from buffer_registry
        get_alloc = buffer_registry.get_allocator
        alloc_delta = get_alloc("delta", self)
        alloc_residual = get_alloc("residual", self)
        alloc_residual_temp = get_alloc("residual_temp", self)
        alloc_stage_base_bt = get_alloc("stage_base_bt", self)
        alloc_krylov_iters_local = get_alloc("krylov_iters_local", self)
        alloc_prev_theta = get_alloc("prev_theta", self)

        # Get child allocators for linear solver. Named registration
        # keeps re-registration idempotent if the linear solver
        # instance is ever replaced.
        alloc_lin_shared, alloc_lin_persistent = (
            buffer_registry.get_child_allocators(
                self, self.linear_solver, name="linear_solver"
            )
        )

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def newton_krylov_solver(
            stage_increment,
            parameters,
            drivers,
            t,
            h,
            a_ij,
            base_state,
            step_start,
            shared_scratch,
            persistent_scratch,
            counters,
        ):
            """Solve the nonlinear system."""

            # Allocate buffers from registry
            delta = alloc_delta(shared_scratch, persistent_scratch)
            residual = alloc_residual(shared_scratch, persistent_scratch)
            residual_temp = alloc_residual_temp(
                shared_scratch, persistent_scratch
            )
            stage_base_bt = alloc_stage_base_bt(
                shared_scratch, persistent_scratch
            )
            prev_theta_store = alloc_prev_theta(
                shared_scratch, persistent_scratch
            )
            lin_shared = alloc_lin_shared(shared_scratch, persistent_scratch)
            lin_persistent = alloc_lin_persistent(
                shared_scratch, persistent_scratch
            )
            krylov_iters_local = alloc_krylov_iters_local(
                shared_scratch, persistent_scratch
            )

            # Warm-started contraction estimate. Only values in the
            # valid band are trusted: converged solves store estimates
            # at or below one, so anything else marks a fresh or
            # uninitialised scratch buffer and starts conservatively.
            stored_theta = prev_theta_store[0]
            theta_valid = (stored_theta > typed_zero) & (
                stored_theta <= typed_one
            )
            prev_theta = selp(theta_valid, stored_theta, typed_one)

            # RMS norm of the previous accepted full-step correction;
            # zero marks unavailable contraction history.
            ndz_prev = typed_zero

            converged = False
            failed = False
            final_status = success

            # Track the latest active iteration's status signals.
            last_lin_status = success
            last_backtrack_failed = False

            iters_count = int32(0)
            total_krylov_iters = int32(0)
            iteration = int32(0)
            mask = activemask()
            for _ in range(max_iters):
                if all_sync(mask, converged | failed):
                    break
                iteration += int32(1)
                active = (not converged) & (not failed)

                residual_function(
                    stage_increment,
                    parameters,
                    drivers,
                    t,
                    h,
                    a_ij,
                    base_state,
                    residual,
                )
                if use_backtracking:
                    norm2_residual = scaled_norm_fn(
                        residual, stage_increment
                    )
                for i in range(n_val):
                    residual[i] = -residual[i]
                    delta[i] = typed_zero

                krylov_iters_local[0] = int32(0)
                lin_status = linear_solver_fn(
                    stage_increment,
                    parameters,
                    drivers,
                    base_state,
                    t,
                    h,
                    a_ij,
                    residual,
                    delta,
                    lin_shared,
                    lin_persistent,
                    krylov_iters_local,
                )

                total_krylov_iters += selp(
                    active, krylov_iters_local[0], int32(0)
                )
                last_lin_status = selp(active, lin_status, last_lin_status)
                iters_count = selp(
                    active, int32(iters_count + int32(1)), iters_count
                )

                norm2_dz = correction_norm_fn(
                    delta,
                    stage_increment,
                    base_state,
                    step_start,
                    a_ij,
                )
                ndz = numba_precision(math_sqrt(norm2_dz))

                # A failed linear solve yields no usable correction:
                # nothing commits and no contraction evidence accrues.
                judged = active & (lin_status == success)
                history = ndz_prev > typed_zero
                ndz_prev_safe = selp(history, ndz_prev, typed_one)
                theta = selp(
                    history,
                    max(theta_decay * prev_theta, ndz / ndz_prev_safe),
                    prev_theta,
                )
                small_first_step = (iteration == int32(1)) & (
                    ndz < first_iteration_bound
                )
                eta_accept = (theta < typed_one) & (
                    theta * ndz < kappa * (typed_one - theta)
                )

                if use_backtracking:
                    # Full-step acceptance by the update-error bound;
                    # damped trials fall back to residual descent.
                    accept_update = judged & (
                        eta_accept | small_first_step
                    )
                    for i in range(n_val):
                        stage_base_bt[i] = stage_increment[i]
                        stage_increment[i] = selp(
                            accept_update,
                            stage_base_bt[i] + delta[i],
                            stage_increment[i],
                        )
                    converged = converged | accept_update
                    prev_theta = selp(judged & history, theta, prev_theta)
                    searching = active & (not converged)
                    accepted_alpha = selp(
                        accept_update, typed_one, typed_zero
                    )
                    ndz_next = selp(accept_update, ndz, typed_zero)
                    alpha = typed_one

                    for _ in range(max_backtracks):
                        if not any_sync(mask, searching):
                            break

                        for i in range(n_val):
                            stage_increment[i] = (
                                stage_base_bt[i] + alpha * delta[i]
                            )

                        residual_function(
                            stage_increment,
                            parameters,
                            drivers,
                            t,
                            h,
                            a_ij,
                            base_state,
                            residual_temp,
                        )
                        norm2_new = scaled_norm_fn(
                            residual_temp, stage_increment
                        )

                        accept_trial = searching & (
                            norm2_new < norm2_residual
                        )
                        converged = converged | (
                            accept_trial & (norm2_new <= typed_one)
                        )
                        accepted_alpha = selp(
                            accept_trial, alpha, accepted_alpha
                        )
                        searching = searching & (not accept_trial)
                        ndz_next = selp(
                            accept_trial
                            & (alpha == typed_one)
                            & (lin_status == success),
                            ndz,
                            ndz_next,
                        )
                        for i in range(n_val):
                            stage_increment[i] = (
                                stage_base_bt[i]
                                + accepted_alpha * delta[i]
                            )

                        alpha *= typed_damping

                    ndz_prev = ndz_next
                    last_backtrack_failed = searching
                    for i in range(n_val):
                        stage_increment[i] = selp(
                            searching,
                            stage_base_bt[i],
                            stage_increment[i],
                        )
                else:
                    # Divergence and stagnation are judged before the
                    # commit, mirroring OrdinaryDiffEq's NLNewton.
                    nonfinite = not (norm2_dz <= typed_huge)
                    stagnant = (
                        judged
                        & history
                        & (abs(theta - typed_one) <= stagnation_eps)
                    )
                    diverging = judged & (
                        (history & (theta > theta_divergence_bound))
                        | nonfinite
                    )
                    converged_stagnant = (
                        stagnant & (ndz <= typed_one) & (not diverging)
                    )
                    failed_now = diverging | (
                        stagnant & (ndz > typed_one)
                    )
                    failed = failed | failed_now

                    commit = (
                        judged
                        & (not failed_now)
                        & (not converged_stagnant)
                    )
                    for i in range(n_val):
                        stage_increment[i] = selp(
                            commit,
                            stage_increment[i] + delta[i],
                            stage_increment[i],
                        )
                    converged = (
                        converged
                        | converged_stagnant
                        | (commit & (eta_accept | small_first_step))
                    )
                    ndz_prev = selp(commit, ndz, typed_zero)
                    prev_theta = selp(judged & history, theta, prev_theta)

            # Persist contraction history for the next solve; a failed
            # solve resets it to the conservative estimate.
            prev_theta_store[0] = selp(converged, prev_theta, typed_one)

            final_status = selp(
                (not converged) & (not failed),
                int32(final_status | max_newton_iters_exceeded),
                final_status,
            )
            final_status = selp(
                (not converged) & failed,
                int32(final_status | newton_divergence),
                final_status,
            )
            final_status = selp(
                (not converged) & last_backtrack_failed,
                int32(final_status | newton_backtracking_failed),
                final_status,
            )
            final_status = selp(
                (not converged) & (last_lin_status != success),
                int32(final_status | last_lin_status),
                final_status,
            )

            counters[0] = iters_count
            counters[1] = total_krylov_iters

            return final_status

        # no cover: end
        return NewtonKrylovCache(newton_krylov_solver=newton_krylov_solver)

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Set[str]:
        """Update compile settings and invalidate cache if changed.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of settings to update.
        silent : bool, default False
            If True, suppress warnings about unrecognized keys.
        **kwargs
            Additional settings as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        recognized = set()

        # Forward krylov-prefixed params to linear solver
        recognized |= self.linear_solver.update(all_updates, silent=True)
        # Add linear_solver_function to updates for compile settings
        all_updates["linear_solver_function"] = (
            self.linear_solver.device_function
        )
        # Update the norm before reading its compiled functions.
        recognized |= super().update(all_updates, silent=True)
        self.update_compile_settings(
            {
                "correction_norm_function": (
                    self.norm.correction_device_function
                )
            },
            silent=True,
        )

        # Buffer locations handled by registry
        recognized |= buffer_registry.update(
            self, updates_dict=all_updates, silent=True
        )
        self.register_buffers()

        return recognized

    @property
    def device_function(self) -> Callable:
        """Return cached Newton-Krylov solver device function."""
        return self.get_cached_output("newton_krylov_solver")

    @property
    def newton_atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.norm.atol

    @property
    def newton_rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.norm.rtol

    @property
    def newton_max_iters(self) -> int:
        """Return maximum Newton iterations."""
        return self.max_iters

    @property
    def newton_damping(self) -> float:
        """Return damping factor."""
        return self.compile_settings.newton_damping

    @property
    def newton_max_backtracks(self) -> int:
        """Return the number of damped trials."""
        return self.compile_settings.newton_max_backtracks

    @property
    def krylov_atol(self) -> ndarray:
        """Return the Krylov absolute tolerance array from nested linear solver."""
        return self.linear_solver.atol

    @property
    def krylov_rtol(self) -> ndarray:
        """Return the Krylov relative tolerance array from nested linear solver."""
        return self.linear_solver.rtol

    @property
    def krylov_max_iters(self) -> int:
        """Return max linear iterations from nested linear solver."""
        return self.linear_solver.max_iters

    @property
    def linear_correction_type(self) -> str:
        """Return correction type from nested linear solver."""
        return self.linear_solver.linear_correction_type

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return merged Newton and linear solver configuration.

        Combines Newton-level settings from compile_settings with
        linear solver settings from nested linear_solver instance,
        plus tolerance arrays from the norm factory.

        Returns
        -------
        dict
            Merged configuration dictionary containing both Newton
            parameters, linear solver parameters, and tolerance arrays.
        """
        combined = dict(self.linear_solver.settings_dict)
        combined.update(self.compile_settings.settings_dict)
        combined["newton_atol"] = self.newton_atol
        combined["newton_rtol"] = self.newton_rtol
        return combined
