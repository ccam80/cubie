from numba import cuda

from cubie.integrators.matrix_free_solvers import (linear_solver_factory,
    newton_krylov_solver_factory)
from cubie.integrators.algorithms_.base_algorithm_step import BaseAlgorithmStep


class ODEImplicitStep(BaseAlgorithmStep):
    def __init__(self):
        pass

    def build_step(self):
        @cuda.jit
        def explicit_step():
            pass

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