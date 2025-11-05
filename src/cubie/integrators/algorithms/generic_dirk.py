"""
Diagonally implicit Runge--Kutta integration step implementation.

This module implements configurable diagonally implicit Runge--Kutta (DIRK)
methods for CUDA-accelerated stiff ODE integration. DIRK methods have
nonzero diagonal entries in the Butcher tableau, requiring solution of
nonlinear systems at each stage via Newton--Krylov iteration with matrix-free
linear solvers.

The implementation supports arbitrary DIRK tableaus including SDIRK (Singly
Diagonally Implicit) and ESDIRK (Explicit first stage SDIRK) methods,
embedded error estimates for adaptive step control, and optional FSAL-like
optimizations for compatible tableaus.

Classes
-------
DIRKStepConfig
    Configuration attrs class for DIRK integrators.
DIRKStep
    Factory producing compiled CUDA device functions for DIRK methods.

Notes
-----
Each implicit stage is solved using a Newton--Krylov solver with Neumann
series preconditioning and backtracking line search. The shared memory
layout partitions space for stage accumulators and solver scratch buffers.

To avoid warp divergence when deciding whether to reuse cached increments
from the previous accepted step, the implementation uses warp-vote
intrinsics that ensure all threads in a warp make the same decision
(fix for issue #149).
"""

from typing import Callable, Optional

import attrs
import numpy as np
from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)


DIRK_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.6,
        "kd": 0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)
@attrs.define
class DIRKStepConfig(ImplicitStepConfig):
    """
    Configuration describing the DIRK integrator.

    Parameters
    ----------
    tableau
        Diagonally implicit Runge--Kutta tableau defining the method's
        coefficients and error estimate. Defaults to
        :data:`DEFAULT_DIRK_TABLEAU`.

    Attributes
    ----------
    tableau : DIRKTableau
        Tableau specifying stage coefficients, diagonal elements, and
        embedded error weights for adaptive control.
    """

    tableau: DIRKTableau = attrs.field(
        default=DEFAULT_DIRK_TABLEAU,
    )


class DIRKStep(ODEImplicitStep):
    """
    Diagonally implicit Runge--Kutta step with an embedded error estimate.

    This class compiles CUDA device functions for diagonally implicit
    Runge--Kutta methods, suitable for stiff ODEs. Each implicit stage
    is solved using Newton--Krylov iteration with matrix-free linear
    solvers and optional preconditioning. The implementation supports
    embedded error estimates for adaptive step control and FSAL
    optimization for certain tableaus.

    DIRK methods have nonzero diagonal entries in the Butcher tableau,
    requiring solution of nonlinear systems at each stage. The diagonal
    structure allows sequential stage solving rather than simultaneous
    solution of all stages.

    Notes
    -----
    The FSAL optimization caches the final stage increment when the
    tableau's structure permits reuse on the next accepted step. To
    avoid warp divergence on GPUs, the implementation uses warp-vote
    intrinsics (``activemask`` and ``all_sync``) to ensure all threads
    in a warp make the same caching decision based on whether all systems
    accepted the previous step (issue #149).

    The Newton--Krylov solver expects caller-supplied residual functions,
    linear operators, and preconditioners obtained through the
    ``get_solver_helper_fn`` callable.

    Examples
    --------
    >>> from cubie.integrators.algorithms import DIRKStep
    >>> from cubie.integrators.algorithms.generic_dirk_tableaus import SDIRK23
    >>> step = DIRKStep(
    ...     precision=np.float64,
    ...     n=3,
    ...     dt=0.01,
    ...     tableau=SDIRK23,
    ...     newton_tolerance=1e-6,
    ...     max_newton_iters=100
    ... )
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 2,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        newton_tolerance: float = 1e-6,
        max_newton_iters: int = 100,
        newton_damping: float = 0.5,
        newton_max_backtracks: int = 8,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
        n_drivers: int = 0,
    ) -> None:
        """
        Initialise the DIRK step configuration.

        Parameters
        ----------
        precision
            Numpy dtype for floating-point operations (float16/32/64).
        n
            Number of state variables in the ODE system.
        dt
            Initial time step size. If ``None``, must be set before solving.
        dxdt_function
            CUDA device function computing the time derivative.
        observables_function
            CUDA device function computing observable quantities.
        driver_function
            Optional CUDA device function for time-varying forcing terms.
        get_solver_helper_fn
            Callable returning solver helper functions (residual, operator,
            preconditioner) for the Newton--Krylov iteration.
        preconditioner_order
            Order of the Neumann series preconditioner. Defaults to 2.
        krylov_tolerance
            Convergence tolerance for the GMRES-like linear solver.
            Defaults to 1e-6.
        max_linear_iters
            Maximum iterations for the linear solver. Defaults to 200.
        linear_correction_type
            Type of Krylov correction ("minimal_residual" or "full_orthog").
            Defaults to "minimal_residual".
        newton_tolerance
            Convergence tolerance for Newton iteration. Defaults to 1e-6.
        max_newton_iters
            Maximum Newton iterations per stage. Defaults to 100.
        newton_damping
            Damping factor for backtracking line search. Defaults to 0.5.
        newton_max_backtracks
            Maximum backtracking steps in line search. Defaults to 8.
        tableau
            Diagonally implicit Runge--Kutta tableau describing the
            coefficients used by the integrator. Defaults to
            :data:`DEFAULT_DIRK_TABLEAU`.
        n_drivers
            Number of driver (forcing) terms. Defaults to 0.

        Notes
        -----
        The tableau determines the method's order, stability properties,
        and whether embedded error weights are available for adaptive
        step control. SDIRK (Singly Diagonally Implicit) and ESDIRK
        (Explicit first stage SDIRK) tableaus are common choices for
        stiff problems.

        The Newton--Krylov solver relies on matrix-free operators,
        avoiding explicit Jacobian storage and enabling solution of
        large-scale systems.
        """

        mass = np.eye(n, dtype=precision)
        config = DIRKStepConfig(
            precision=precision,
            n=n,
            n_drivers=n_drivers,
            dt=dt,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            get_solver_helper_fn=get_solver_helper_fn,
            preconditioner_order=preconditioner_order,
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            newton_tolerance=newton_tolerance,
            max_newton_iters=max_newton_iters,
            newton_damping=newton_damping,
            newton_max_backtracks=newton_max_backtracks,
            tableau=tableau,
            beta=1.0,
            gamma=1.0,
            M=mass,
        )
        self._cached_auxiliary_count = 0
        super().__init__(config, DIRK_DEFAULTS)

    def build_implicit_helpers(
        self,
    ) -> Callable:
        """
        Construct the nonlinear solver chain used by implicit methods.

        Returns
        -------
        Callable
            Compiled CUDA device function implementing Newton--Krylov
            iteration for solving implicit stage equations.

        Notes
        -----
        This method assembles the following components:

        1. Neumann series preconditioner for the linear system
        2. Nonlinear stage residual function
        3. Matrix-free linear operator for Jacobian-vector products
        4. GMRES-like linear solver using the operator and preconditioner
        5. Newton--Krylov solver combining nonlinear iteration with
           backtracking line search

        All components are obtained through the ``get_solver_helper_fn``
        callable, which must return compiled CUDA device functions
        matching the expected signatures.
        """

        precision = self.precision
        config = self.compile_settings
        beta = config.beta
        gamma = config.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        residual = get_fn(
            "stage_residual",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        operator = get_fn(
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(
            operator,
            n=n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )

        newton_tolerance = config.newton_tolerance
        max_newton_iters = config.max_newton_iters
        newton_damping = config.newton_damping
        newton_max_backtracks = config.newton_max_backtracks

        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
        )

        return nonlinear_solver

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """
        Compile the DIRK device step.

        Parameters
        ----------
        solver_fn
            Compiled Newton--Krylov solver device function.
        dxdt_fn
            Compiled CUDA device function for time derivatives.
        observables_function
            Compiled CUDA device function for observables.
        driver_function
            Optional compiled CUDA device function for drivers.
        numba_precision
            Numba dtype (e.g., numba.float64) matching the precision.
        n
            Number of state variables.
        dt
            Time step size (may be ``None`` for runtime determination).
        n_drivers
            Number of driver variables.

        Returns
        -------
        StepCache
            Container holding the compiled step device function and the
            nonlinear solver.

        Notes
        -----
        The compiled step function expects the following signature:

        .. code-block:: python

            step(state, proposed_state, parameters, driver_coeffs,
                 drivers_buffer, proposed_drivers, observables,
                 proposed_observables, error, dt_scalar, time_scalar,
                 first_step_flag, accepted_flag, shared, persistent_local)

        The shared memory layout partitions space for:

        1. Stage accumulators: ``(stage_count - 1) * n`` entries
        2. Solver scratch: Workspace for Newton--Krylov iteration

        Within solver scratch, the first ``n`` entries serve as stage RHS
        storage, and the second ``n`` entries cache the final stage
        increment for FSAL reuse. When FSAL is active and not the first
        step, all threads in a warp vote on whether the previous step was
        accepted. Only when all threads accepted does the warp reuse the
        cached increment, avoiding divergence (issue #149).

        The step function returns an int32 status code, with the upper 16
        bits encoding the total Newton iteration count across all stages.
        """

        config = self.compile_settings
        tableau = config.tableau
        nonlinear_solver = solver_fn
        stage_count = tableau.stage_count

        # Compile-time toggles
        has_driver_function = driver_function is not None
        has_error = self.is_adaptive
        multistage = stage_count > 1
        first_same_as_last = self.first_same_as_last
        can_reuse_accepted_start = self.can_reuse_accepted_start

        stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
        diagonal_coeffs = tableau.diagonal(numba_precision)
        stage_implicit = tuple(coeff != numba_precision(0.0)
                          for coeff in diagonal_coeffs)
        accumulator_length = max(stage_count - 1, 0) * n
        solver_shared_elements = self.solver_shared_elements

        # Shared memory indices
        acc_start = 0
        acc_end = accumulator_length
        solver_start = acc_end
        solver_end = acc_end + solver_shared_elements


        # no cover: start
        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :, :],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision,
                numba_precision,
                int16,
                int16,
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
        ):
            # ----------------------------------------------------------- #
            # Shared and local buffer guide:
            # stage_accumulator: size (stage_count-1) * n, shared memory.
            #   Default behaviour:
            #       - Stores accumulated explicit contributions for successors.
            #       - Slice k feeds the base state for stage k+1.
            #   Reuse:
            #       - stage_base: first slice (size n)
            #           - Holds the working state during the current stage.
            #           - New data lands only after the prior stage has finished.
            # solver_scratch: size solver_shared_elements, shared memory.
            #   Default behaviour:
            #       - Provides workspace for the Newton iteration helpers.
            #   Reuse:
            #       - stage_rhs: first slice (size n)
            #           - Carries the Newton residual and then the stage rhs.
            #           - Once a stage closes we reuse it for the next residual,
            #             so no live data remains.
            #       - increment_cache: second slice (size n)
            #           - Receives the accepted increment at step end for FSAL.
            #           - Solver stops touching it once convergence is reached.
            #       - eval_state: third slice (size n)
            #           - Stores base_state + a_ij * stage_increment.
            #           - Reserved for Newton Jacobian evaluations.
            # stage_increment: size n, per-thread local memory.
            #   Default behaviour:
            #       - Starts as the Newton guess and finishes as the step.
            #       - Copied into increment_cache once the stage closes.
            # proposed_state: size n, global memory.
            #   Default behaviour:
            #       - Carries the running solution with each stage update.
            #       - Only updated after a stage converges, keeping data stable.
            # proposed_drivers / proposed_observables: size n each, global.
            #   Default behaviour:
            #       - Refresh to the stage time before rhs or residual work.
            #       - Later stages reuse only the newest values, so no clashes.
            # ----------------------------------------------------------- #
            stage_increment = cuda.local.array(n, numba_precision)

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[acc_start:acc_end]
            solver_scratch = shared[solver_start:solver_end]
            stage_rhs = solver_scratch[:n]
            increment_cache = solver_scratch[n:2*n]

            #Alias stage base onto first stage accumulator - lifetimes disjoint
            if multistage:
                stage_base = stage_accumulator[:n]
            else:
                stage_base = cuda.local.array(n, numba_precision)

            for idx in range(n):
                if has_error:
                    error[idx] = typed_zero
                stage_increment[idx] = increment_cache[idx] # cache spent

            status_code = int32(0)
            # --------------------------------------------------------------- #
            #            Stage 0: may reuse cached values                     #
            # --------------------------------------------------------------- #

            first_step = first_step_flag != int16(0)
            # Warp-vote to avoid divergence on FSAL cache decision (issue #149)
            if first_same_as_last and not first_step and multistage:
                mask = activemask()
                all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
                use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            stage_time = current_time + dt_value * stage_time_fractions[0]
            diagonal_coeff = diagonal_coeffs[0]

            for idx in range(n):
                stage_base[idx] = state[idx]
                proposed_state[idx] = typed_zero

            # Only caching achievable is reusing rhs for FSAL
            if use_cached_rhs:
                # RHS is aliased onto solver scratch cache at step-end
                pass

            else:
                if can_reuse_accepted_start:
                    for idx in range(drivers_buffer.shape[0]):
                        # Use step-start driver values
                        proposed_drivers[idx] = drivers_buffer[idx]

                else:
                    if has_driver_function:
                        driver_function(
                            stage_time,
                            driver_coeffs,
                            proposed_drivers,
                        )

                if stage_implicit[0]:
                    status_code |= nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_value,
                        diagonal_coeffs[0],
                        stage_base,
                        solver_scratch,
                    )
                    for idx in range(n):
                        stage_base[idx] += (
                            diagonal_coeff * stage_increment[idx]
                        )

                # Get obs->dxdt from stage_base
                observables_function(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

            solution_weight = solution_weights[0]
            error_weight = error_weights[0]
            for idx in range(n):
                rhs_value = stage_rhs[idx]
                proposed_state[idx] += solution_weight * rhs_value
                if has_error:
                    error[idx] += error_weight * rhs_value
                            
            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all qtys                    #
            # --------------------------------------------------------------- #

            for stage_idx in range(1, stage_count):
                prev_idx = stage_idx - 1
                successor_range = stage_count - stage_idx
                stage_time = (
                        current_time + dt_value * stage_time_fractions[stage_idx]
                )

                # Fill accumulators with previous step's contributions
                for successor_offset in range(successor_range):
                    successor_idx = stage_idx + successor_offset
                    base = (successor_idx - 1) * n
                    for idx in range(n):
                        state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                        contribution = state_coeff * stage_rhs[idx] * dt_value
                        stage_accumulator[base + idx] += contribution

                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                # Grab a view of the completed accumulator slice, add state
                stage_base = stage_accumulator[(stage_idx-1) * n:stage_idx * n]
                for idx in range(n):
                    stage_base[idx] += state[idx]

                diagonal_coeff = diagonal_coeffs[stage_idx]

                if stage_implicit[stage_idx]:
                    status_code |= nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        stage_time,
                        dt_value,
                        diagonal_coeffs[stage_idx],
                        stage_base,
                        solver_scratch,
                    )

                    for idx in range(n):
                        stage_base[idx] += diagonal_coeff * stage_increment[idx]

                observables_function(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                dxdt_fn(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    proposed_state[idx] += solution_weight * rhs_value
                    if has_error:
                        error[idx] += error_weight * rhs_value

            # --------------------------------------------------------------- #

            for idx in range(n):
                proposed_state[idx] *= dt_value
                proposed_state[idx] += state[idx]
                if has_error:
                    error[idx] *= dt_value

            if has_driver_function:
                driver_function(
                    end_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            # RHS auto-cached through aliasing to solver scratch
            for idx in range(n):
                increment_cache[idx] = stage_increment[idx]

            return status_code
        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def is_multistage(self) -> bool:
        """
        Return ``True`` as the method has multiple stages.

        Returns
        -------
        bool
            ``True`` when stage count exceeds 1.
        """
        return self.tableau.stage_count > 1


    @property
    def is_adaptive(self) -> bool:
        """
        Return ``True`` because an embedded error estimate is produced.

        Returns
        -------
        bool
            ``True`` when the tableau provides embedded error weights for
            adaptive step size control.
        """
        return self.tableau.has_error_estimate

    @property
    def cached_auxiliary_count(self) -> int:
        """
        Return the number of cached auxiliary entries for the JVP.

        Returns
        -------
        int
            Number of auxiliary variables cached by solver helpers.

        Notes
        -----
        Lazily builds implicit helpers so as not to return an errant
        ``None``.
        """
        if self._cached_auxiliary_count is None:
            self.build_implicit_helpers()
        return self._cached_auxiliary_count

    @property
    def shared_memory_required(self) -> int:
        """
        Return the number of precision entries required in shared memory.

        Returns
        -------
        int
            Total shared memory requirement in floating-point entries,
            comprising stage accumulators, solver scratch space, and
            cached auxiliary variables.
        """

        tableau = self.tableau
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
        return (accumulator_span
            + self.solver_shared_elements
            + self.cached_auxiliary_count
        )

    @property
    def local_scratch_required(self) -> int:
        """
        Return the number of local precision entries required.

        Returns
        -------
        int
            Number of per-thread local memory entries. Equals ``2 * n``
            to hold stage increment and temporary state variables.
        """
        return 2 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """
        Return the number of persistent local entries required.

        Returns
        -------
        int
            Always returns 0 as DIRK steps do not require persistent
            local storage across invocations.
        """
        return 0

    @property
    def is_implicit(self) -> bool:
        """
        Return ``True`` because the method solves nonlinear systems.

        Returns
        -------
        bool
            Always ``True`` for DIRK methods.
        """
        return True

    @property
    def order(self) -> int:
        """
        Return the classical order of accuracy.

        Returns
        -------
        int
            The order of the method as specified by the tableau.
        """
        return self.tableau.order


    @property
    def threads_per_step(self) -> int:
        """
        Return the number of CUDA threads that advance one state.

        Returns
        -------
        int
            Always returns 1 as each system is integrated by a single thread.
        """

        return 1
