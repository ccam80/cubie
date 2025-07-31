import pytest
import numpy as np
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester

from CuMC.ForwardSim.integrators.algorithms.euler import Euler
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes


class TestEuler(LoopAlgorithmTester):
    """Testing class for the Euler algorithm. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    @pytest.mark.parametrize("system_override", [None, "ThreeChamber", "Decays1_100", "genericODE"])
    def test_loop_function_builds(self, built_loop_function):
        pass

    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides, output_functions,
                                                       loop_under_test, expected_summary_buffer_size,
                                                       ):
        pass

    @pytest.fixture(scope="class")
    def algorithm_class(self):
        return Euler

    @pytest.fixture(scope="function")
    def expected_answer(self, system, loop_compile_settings, run_settings, solver, inputs, precision):
        inits = inputs['initial_values']
        params = inputs['parameters']
        driver_vec = inputs['forcing_vectors']
        dt = loop_compile_settings['dt_min']
        output_dt = loop_compile_settings['dt_save']
        warmup = solver.warmup
        duration = solver.duration
        saved_observable_indices = loop_compile_settings['saved_observable_indices']
        saved_state_indices = loop_compile_settings['saved_state_indices']
        save_time = "time" in loop_compile_settings['output_functions']

        state_output, observables_output = self._cpu_euler_loop(system, inits, params, driver_vec, dt, output_dt,
                                                                warmup, duration, saved_observable_indices, saved_state_indices,
                                                                save_time,
                                                                )

        return state_output, observables_output

    def _cpu_euler_loop(self,
                        system,
                        inits,
                        params,
                        driver_vec,
                        dt,
                        output_dt,
                        warmup,
                        duration,
                        saved_observable_indices,
                        saved_state_indices,
                        save_time,
                        ):
        """A simple CPU implementation of the Euler loop for testing."""
        t = 0.0
        save_every = int(round(output_dt / dt))
        output_length = int(duration / output_dt)
        warmup_samples = int(warmup / output_dt)
        n_saved_states = len(saved_state_indices)
        n_saved_observables = len(saved_observable_indices)
        total_samples = int((duration + warmup) / output_dt)

        state_output = np.zeros((output_length, n_saved_states + save_time * 1), dtype=inits.dtype)
        observables_output = np.zeros((output_length, n_saved_observables), dtype=inits.dtype)
        state = inits.copy()

        for i in range(total_samples):
            for j in range(save_every):
                drivers = driver_vec[:, (i * save_every + j) % len(driver_vec)]
                t += dt
                dx, observables = system.correct_answer_python(state, params, drivers)
                state += dx * dt
            if i > (warmup_samples - 1):
                state_output[i - warmup_samples, :n_saved_states] = state[saved_state_indices]
                observables_output[i - warmup_samples, :] = observables[saved_observable_indices]
                if save_time:
                    state_output[i - warmup_samples, -1] = i - warmup_samples

        return state_output, observables_output

    @pytest.mark.nocudasim
    @pytest.mark.parametrize("loop_compile_settings_overrides, inputs_override, solver_settings_override, "
                             "precision_override",
                             [({'output_functions':  ["state", "observables", "time"], 'saved_state_indices': [0, 1, 2],
                                "saved_observable_indices": [0, 1]
                                },
                               {},
                               {'duration': 0.1, 'warmup': 0.0},
                               {}
                               ),
                              ({'output_functions': ["state", "observables", "mean", "max", "rms", "peaks[3]"]},
                               {},
                               {'duration': 1.0, 'warmup': 1.0},
                               {'precision': np.float64}
                               ),

                              ({'output_functions': ["state", "observables", "mean", "max", "rms", "peaks[3]"]},
                               {},
                               {'duration': 5.0, 'warmup': 1.0},
                               {'precision': np.float32}
                               ),
                              ],
                             ids=['no_summaries', 'all_summaries_64', 'all_summaries_long_32'],
                             indirect=True,
                             )
    def test_loop(self, loop_test_kernel, outputs, inputs, precision, output_functions,
                  loop_under_test, expected_answer, expected_summaries, solver,
                  ):
        super().test_loop(loop_test_kernel, outputs, inputs, precision, output_functions,
                          loop_under_test, expected_answer, expected_summaries, solver,
                          )

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{},
                              {'dt_min':            0.002,
                               'dt_max':            0.02,
                               'dt_save':           0.02,
                               'dt_summarise':      0.2,
                               'atol':              1.0e-7,
                               'rtol':              1.0e-5,
                               'saved_state_indices':      [0, 1, 2],
                               'saved_observable_indices': [0, 2],
                               'output_functions':  ["state", "peaks[3]"],
                               }
                              ],
                             ids=['no_change', 'change_all'],
                             indirect=True,
                             )
    def test_loop_function_builds(self, built_loop_function):
        super().test_loop_function_builds(built_loop_function)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{},
                              {'dt_min':            0.002,
                               'dt_max':            0.02,
                               'dt_save':           0.02,
                               'dt_summarise':      0.2,
                               'atol':              1.0e-7,
                               'rtol':              1.0e-5,
                               'saved_state_indices':      [0, 1, 2],
                               'saved_observable_indices': [0, 2],
                               'output_functions':  ["state", "peaks[3]"],
                               }
                              ],
                             ids=['no_change', 'change_all'],
                             indirect=True,
                             )
    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides, output_functions,
                                                       loop_under_test, expected_summary_buffer_size,
                                                       ):
        super().test_loop_compile_settings_passed_successfully(loop_compile_settings_overrides, output_functions,
                                                               loop_under_test, expected_summary_buffer_size,
                                                               )

    @pytest.fixture()
    def expected_loop_shared_memory(self, system, output_functions):
        """Calculate the expected shared memory size for the Euler algorithm."""
        sizes = LoopBufferSizes.from_system_and_output_fns(system, output_functions)
        loop_shared_memory = (sizes.state + sizes.dxdt + sizes.observables + sizes.drivers +
                              sizes.state_summaries + sizes.observable_summaries)
        return loop_shared_memory