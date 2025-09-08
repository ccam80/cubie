from numba import cuda, from_dtype, int32

from cubie.integrators.loops.BaseIntegratorLoop import BaseIntegratorLoop


class ODEFixedStepLoop(BaseIntegratorLoop):
    def __init__(self):
        pass

    def build_loop(self,
                   precision,
                   step_fn,
                   save_state_func,
                   update_summaries_func,
                   save_summaries_func,):

        """
        Build the CUDA device function for the Euler integration loop.

        This method constructs a numba-compiled CUDA device function that
        implements the Euler integration algorithm with output handling.

        Parameters
        ----------
        precision : type
            Numerical precision type for the integration.
        dxdt_function : callable
            Function that computes time derivatives.
        save_state_func : callable
            Function for saving state values.
        update_summaries_func : callable
            Function for updating summary statistics.
        save_summaries_func : callable
            Function for saving summary statistics.

        Returns
        -------
        callable
            Compiled CUDA device function for Euler integration.

        Notes
        -----
        The generated function handles memory allocation, state initialization,
        the main integration loop, and output generation according to the
        configured step sizes and buffer requirements.
        """

        save_steps, summarise_steps, step_size = (
            self.compile_settings.fixed_steps
        )

        sizes = self.compile_settings.buffer_sizes
        flags = self.compile_settings.compile_flags
        save_observables_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_observables_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state

        state_buffer_size = sizes.state
        observables_buffer_size = sizes.observables
        dxdt_buffer_size = sizes.dxdt
        parameter_buffer_size = sizes.nonzero.parameters
        parameters_actual = sizes.parameters
        drivers_buffer_size = sizes.drivers
        state_summary_buffer_size = sizes.state_summaries
        observables_summary_buffer_size = sizes.observable_summaries

        # Generate indices into shared memory as compile-time constants
        dxdt_start_index = state_buffer_size
        observables_start_index = dxdt_start_index + dxdt_buffer_size
        parameters_start_index = (
            observables_start_index + observables_buffer_size
        )
        drivers_start_index = (
            parameters_start_index + parameter_buffer_size
        )
        state_summaries_start_index = (
            drivers_start_index + drivers_buffer_size
        )
        observable_summaries_start_index = (
            state_summaries_start_index + state_summary_buffer_size
        )
        end_index = (
            observable_summaries_start_index
            + observables_summary_buffer_size
        )

        numba_precision = from_dtype(precision)

        # no cover: start
        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                int32,
                int32,
            ),
            device=True,
            inline=True,
        )
        def euler_loop(
            inits,
            parameters,
            forcing_vec,
            shared_memory,
            state_output,
            observables_output,
            state_summaries_output,
            observables_summaries_output,
            output_length,
            warmup_samples=0,
        ):
            """
            CUDA device function implementing the Euler integration loop.

            This function performs the actual Euler integration on the GPU,
            handling state updates, output saving, and summary calculations.

            Parameters
            ----------
            inits : array_like
                Initial values for the state variables.
            parameters : array_like
                System parameters for the integration.
            forcing_vec : array_like, shape (n_steps, n_drivers)
                External forcing/driving values over time.
            shared_memory : array_like
                Shared GPU memory buffer for intermediate calculations.
            state_output : array_like, shape (n_samples, n_states)
                Output array for saving state trajectories.
            observables_output : array_like, shape (n_samples, n_observables)
                Output array for saving observable trajectories.
            state_summaries_output : array_like, shape (n_summaries, n_states)
                Output array for saving state summaries.
            observables_summaries_output : array_like, shape (n_summaries, n_observables)
                Output array for saving observable summaries.
            output_length : int
                Number of output samples to generate.
            warmup_samples : int, default=0
                Number of initial samples to skip (for transient removal).

            Notes
            -----
            This function is compiled as a CUDA device function and runs
            entirely on the GPU. It manages shared memory allocation,
            performs the Euler stepping, and handles all output generation
            according to the configured sampling intervals.
            """

            # Allocate shared memory slices

            state_buffer = shared_memory[:dxdt_start_index]
            dxdt = shared_memory[
                dxdt_start_index:observables_start_index
            ]
            observables_buffer = shared_memory[
                observables_start_index:drivers_start_index
            ]
            parameters_buffer = shared_memory[
                parameters_start_index:drivers_start_index
            ]
            drivers = shared_memory[
                drivers_start_index:state_summaries_start_index
            ]
            state_summary_buffer = shared_memory[
                state_summaries_start_index:observable_summaries_start_index
            ]
            observable_summary_buffer = shared_memory[
                observable_summaries_start_index:end_index
            ]

            driver_length = forcing_vec.shape[0]

            # Initialise/Assign values to allocated memory
            shared_memory[:end_index] = numba_precision(
                0.0
            )  # initialise all shared memory before adding values
            for i in range(state_buffer_size):
                state_buffer[i] = inits[i]

            for i in range(parameters_actual):
                parameters_buffer[i] = parameters[i]

            # Loop through output samples, one iteration per output sample
            for i in range(warmup_samples + output_length):
                # Euler loop - internal step size <= outout step size
                for j in range(save_steps):
                    for k in range(drivers_buffer_size):
                        drivers[k] = forcing_vec[
                            (i * save_steps + j) % driver_length, k
                        ]

                    step_fn()

                    # Start saving after the requested settling time has passed.
                if i > (warmup_samples - 1):
                    output_sample = i - warmup_samples
                    save_state_func(
                        state_buffer,
                        observables_buffer,
                        state_output[
                            output_sample * save_state_bool, :
                        ],
                        observables_output[
                            output_sample * save_observables_bool, :
                        ],
                        output_sample,
                    )
                    update_summaries_func(
                        state_buffer,
                        observables_buffer,
                        state_summary_buffer,
                        observable_summary_buffer,
                        output_sample,
                    )

                    if (i + 1) % summarise_steps == 0:
                        summary_sample = (
                            output_sample + 1
                        ) // summarise_steps - 1
                        save_summaries_func(
                            state_summary_buffer,
                            observable_summary_buffer,
                            state_summaries_output[
                                summary_sample * summarise_state_bool,
                                :,
                            ],
                            observables_summaries_output[
                                summary_sample
                                * summarise_observables_bool,
                                :,
                            ],
                            summarise_steps,
                        )

        return euler_loop
        # no cover: stop