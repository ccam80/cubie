import pytest
import numpy as np
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester

from CuMC.ForwardSim.integrators.algorithms.euler import Euler


@pytest.mark.parametrize("system_override", [None, "ThreeChamber", "Decays1_100", "genericODE"])
class TestEuler(LoopAlgorithmTester):
    """Testing class for the Euler algorithm. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    #overrides to debug the other tests
    def test_loop_function_builds(self, built_loop_function):
        pass

    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides,
                                                       loop_under_test, expected_summary_temp_memory,
                                                       ):
        pass

    @pytest.fixture(scope="class")
    def algorithm_class(self):
        return Euler

    @pytest.fixture(scope="function")
    def expected_answer(self, system, loop_compile_settings, run_settings, inputs, precision):
        inits = inputs['initial_values']
        params = inputs['parameters']
        driver_vec = inputs['forcing_vectors']
        dt = loop_compile_settings['dt_min']
        output_dt = loop_compile_settings['dt_save']
        warmup = run_settings['warmup']
        duration = run_settings['duration']
        saved_observables = loop_compile_settings['saved_observables']
        saved_states = loop_compile_settings['saved_states']
        save_time = "time" in loop_compile_settings['output_functions']

        state_output, observables_output = self._cpu_euler_loop(system, inits, params, driver_vec, dt, output_dt,
                                                                warmup, duration, saved_observables, saved_states,
                                                                save_time
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
                        saved_observables,
                        saved_states,
                        save_time,
                        ):
        """A simple CPU implementation of the Euler loop for testing."""
        t = 0.0
        save_every = int(round(output_dt / dt))
        output_length = int(duration / output_dt)
        warmup_samples = int(warmup / output_dt)
        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)
        total_samples = int((duration + warmup) / output_dt)

        state_output = np.zeros((n_saved_states + save_time * 1, output_length), dtype=inits.dtype)
        observables_output = np.zeros((n_saved_observables, output_length), dtype=inits.dtype)
        state = inits.copy()

        for i in range(total_samples):
            for j in range(save_every):
                drivers = driver_vec[:, (i * save_every + j) % len(driver_vec)]
                t += dt
                dx, observables = system.correct_answer_python(state, params, drivers)
                state += dx * dt
            if i > (warmup_samples - 1):
                state_output[:, i - warmup_samples] = state[saved_states]
                observables_output[:, i - warmup_samples] = observables[saved_observables]
                if save_time:
                    state_output[-1, i - warmup_samples] = i - warmup_samples

        return state_output, observables_output

    @pytest.mark.parametrize("loop_compile_settings_overrides, inputs_override, run_settings_override",
                             [({'output_functions': ["state", "observables"], 'saved_states': [0, 1, 2]}, {}, {}),
                              ({'output_functions':  ["state", "observables", "mean"],
                                'saved_states':      [0, 1],
                                'saved_observables': [0, 1, 2]
                                },
                               {},
                               {}
                               ),
                              ({}, {'initial_values': np.array([1.0, 2.0, 3.0])}, {}),
                              ({}, {}, {'duration': 5.0, 'warmup': 1.0}),
                              ({'output_functions': ["state", "observables", "mean", "max", "rms", "peaks"],
                                'n_peaks':          3, 'saved_states': [0, 1, 2]
                                }, {}, {}
                               )],
                             ids=['state_and_observables_empty_obs_list', 'state_observables_and_mean',
                                  'custom_initial_values', 'custom_run_settings', "all_summaries"],
                             indirect=True,
                             )
    @pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=['float32', 'float64'])
    def test_loop(self, loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                  loop_under_test, expected_answer,
                  ):
        super().test_loop(loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                          loop_under_test, expected_answer,
                          )

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states":      [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions':  ["state", "observables"]
                               },
                              {'dt_min':            0.002,
                               'dt_max':            0.02,
                               'dt_save':           0.02,
                               'dt_summarise':      0.2,
                               'atol':              1.0e-7,
                               'rtol':              1.0e-5,
                               'saved_states':      [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions':  ["state", "peaks"],
                               'n_peaks':           2
                               }
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True,
                             )
    def test_loop_function_builds(self, built_loop_function):
        super().test_loop_function_builds(built_loop_function)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states":      [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions':  ["state", "observables"]
                               },
                              {'dt_min':            0.002,
                               'dt_max':            0.02,
                               'dt_save':           0.02,
                               'dt_summarise':      0.2,
                               'atol':              1.0e-7,
                               'rtol':              1.0e-5,
                               'saved_states':      [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions':  ["state", "peaks"],
                               'n_peaks':           2
                               }
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True,
                             )
    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides,
                                                       loop_under_test, expected_summary_temp_memory,
                                                       ):
        super().test_loop_compile_settings_passed_successfully(loop_compile_settings_overrides,
                                                               loop_under_test, expected_summary_temp_memory,
                                                               )

    @pytest.fixture()
    def expected_loop_shared_memory(self, system):
        """Calculate the expected shared memory size for the Euler algorithm."""
        n_states = system.num_states
        n_obs = system.num_observables
        n_drivers = system.num_drivers
        return n_states + n_states + n_obs + n_drivers