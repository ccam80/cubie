import pytest
import numpy as np

from CuMC.ForwardSim.integrators.algorithms.genericIntegratorAlgorithm import GenericIntegratorAlgorithm
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester


class TestGenericLoopAlgorithm(LoopAlgorithmTester):
    """Test class for the GenericIntegratorAlgorithm."""

    @pytest.fixture(scope="module")
    def algorithm_class(self):
        return GenericIntegratorAlgorithm

    @pytest.fixture(scope="function")
    def expected_answer(self, system, loop_compile_settings, run_settings, solver, inputs, precision):
        save_state = "state" in loop_compile_settings['output_functions']
        save_observables = "observables" in loop_compile_settings['output_functions']
        n_states_total = inputs['initial_values'].shape[0]
        n_samples = solver.output_length
        save_time = "time" in loop_compile_settings['output_functions']

        if save_state:
            saved_states = np.asarray(loop_compile_settings['saved_states'])
            n_saved_states = len(saved_states)
            expected_state_output = np.zeros((n_samples, n_saved_states + 1 * save_time), dtype=precision)
            expected_state_output[:, :n_saved_states] = inputs['initial_values'][saved_states][np.newaxis, :]
            if save_time:
                expected_state_output[:, -1] = np.arange(n_samples, dtype=precision)
        else:
            expected_state_output = np.zeros((1, 1), dtype=precision)

        if save_observables:

            saved_observables = np.asarray(loop_compile_settings['saved_observables'])
            n_saved_observables = len(saved_observables)
            expected_observables = np.zeros((n_samples, n_saved_observables), dtype=precision)
            expected_observables[:, :] = inputs['initial_values'][np.newaxis,
                (saved_observables % n_states_total)]
        else:
            expected_observables = np.zeros((1, 1), dtype=precision)

        return expected_state_output, expected_observables

    @pytest.mark.nocudasim
    @pytest.mark.parametrize("loop_compile_settings_overrides, inputs_override, solver_settings_override, "
                             "precision_override",
                             [({'output_functions':  ["state", "observables", "time"], 'saved_states': [0, 1, 2],
                                "saved_observables": [0, 1]
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
                               'saved_states':      [0, 1, 2],
                               'saved_observables': [0, 2],
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
                               'saved_states':      [0, 1, 2],
                               'saved_observables': [0, 2],
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

    @pytest.fixture(scope='function')
    def expected_loop_shared_memory(self, system, output_functions):
        """Override with your loops expected shared memory usage (not including summary memory, this is tested in
        test_output_functions.py)"""
        return 0