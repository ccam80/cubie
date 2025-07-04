import pytest
import numpy as np

from CuMC.ForwardSim.integrators.algorithms.genericIntegratorAlgorithm import GenericIntegratorAlgorithm
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester
from tests._utils import calculate_expected_summaries

# This class doesn't use anything from the systems, so don't bother parametrizing it with one.
class TestGenericLoopAlgorithm(LoopAlgorithmTester):
    """Test class for the GenericIntegratorAlgorithm."""

    @pytest.fixture(scope="module")
    def algorithm_class(self):
        return GenericIntegratorAlgorithm

    @pytest.fixture(scope="function")
    def expected_answer(self, system, loop_compile_settings, run_settings, inputs, precision):
        """OVERRIDE THIS FIXTURE with a python version of what your integrator loop should return - a scipy integrator
        or homebrew CPU loop should be able to provide an answer that matches the output of a given loop to within
        floating-point precision."""
        save_state = "state" in loop_compile_settings['output_functions']
        save_observables = "observables" in loop_compile_settings['output_functions']
        n_states_total = inputs['initial_values'].shape[0]
        n_samples = int(run_settings['duration'] / loop_compile_settings['dt_save'])

        if save_state:
            saved_states = np.asarray(loop_compile_settings['saved_states'])
            n_saved_states = len(saved_states)
            expected_state_output = np.zeros((n_saved_states, n_samples), dtype=precision)
            expected_state_output[:, :] = inputs['initial_values'][saved_states][:, np.newaxis]
        else:
            expected_state_output = np.zeros((0, n_samples), dtype=precision)

        if save_observables:

            saved_observables = np.asarray(loop_compile_settings['saved_observables'])
            n_saved_observables = len(saved_observables)
            expected_observables = np.zeros((n_saved_observables, n_samples), dtype=precision)
            expected_observables[:, :] = inputs['initial_values'][
                (saved_observables % n_states_total), np.newaxis]
        else:
            expected_observables = np.zeros((0, n_samples), dtype=precision)

        return expected_state_output, expected_observables

    @pytest.mark.parametrize("loop_compile_settings_overrides, inputs_override, run_settings_override",
                             [({'output_functions': ["state", "observables"], 'saved_states': [0, 1, 2]}, {}, {}),
                              ({'output_functions': ["state", "observables", "mean"],
                                'saved_states': [0, 1],
                                'saved_observables': [0, 1, 2]},
                               {},
                               {}),
                              ({}, {'initial_values': np.array([1.0, 2.0, 3.0])}, {}),
                              ({}, {}, {'duration': 5.0, 'warmup': 2.0}),
                              ({'output_functions': ["state", "observables", "mean", "max", "rms", "peaks"],
                                'n_peaks': 3, 'saved_states': [0, 1, 2]}, {}, {})],
                             ids=['state_and_observables_empty_obs_list', 'state_observables_and_mean',
                                  'custom_initial_values', 'custom_run_settings', "all_summaries"],
                             indirect=True)
    def test_loop(self, loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                  loop_under_test, expected_answer):
        super().test_loop(loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                          loop_under_test, expected_answer)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states": [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions': ["state", "observables"]},
                              {'dt_min': 0.002,
                               'dt_max': 0.02,
                               'dt_save': 0.02,
                               'dt_summarise': 0.2,
                               'atol': 1.0e-7,
                               'rtol': 1.0e-5,
                               'saved_states': [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions': ["state", "peaks"],
                               'n_peaks': 2}
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True)
    def test_loop_function_builds(self, built_loop_function):
        super().test_loop_function_builds(built_loop_function)


    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states": [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions': ["state", "observables"]},
                              {'dt_min': 0.002,
                               'dt_max': 0.02,
                               'dt_save': 0.02,
                               'dt_summarise': 0.2,
                               'atol': 1.0e-7,
                               'rtol': 1.0e-5,
                               'saved_states': [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions': ["state", "peaks"],
                               'n_peaks': 2}
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True)
    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides,
                                                       loop_under_test, expected_summary_temp_memory):
        super().test_loop_compile_settings_passed_successfully(loop_compile_settings_overrides,
                                                               loop_under_test, expected_summary_temp_memory)


    @pytest.fixture(scope='function')
    def expected_loop_shared_memory(self, system):
        """Override with your loops expected shared memory usage (not including summary memory, this is tested in
        test_output_functions.py)"""
        return 0

    def _time_to_fixed_steps(self):
        """Fixed-step helper function: Convert the time-based compile_settings to sample-based compile_settings,
        which are used by fixed-step loop functions. Sanity-check values and warn the user if they don't work.

        Returns:
            save_every_samples (int): The number of internal loop steps between saves.
            summarise_every_samples (int): The number of output samples between summary metric calculations.
            step_size (float): The internal time step size used in the loop (dt_min, by default).

        Raises:
            ValueError: If the user tries to save more often than they step, or summarise more often than they save.
            UserWarning: If the output rate or summary rate aren't an integer divisor of the internal loop frequency,
                update these values to be the actual time interval caused by stepping an integer number of steps. Warn
                the user that results aren't what they asked for.
        """

        dt_min = self.loop_parameters['dt_min']
        dt_max = self.loop_parameters['dt_max']
        dt_save = self.loop_parameters['dt_save']
        dt_summarise = self.loop_parameters['dt_summarise']

        check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)

        # Update the actual save and summary intervals, which will differ from what was ordered if they are not
        # a multiple of the loop step size.
        save_every_samples, summarise_every_samples, actual_dt_save, actual_dt_summarise = convert_times_to_fixed_steps(
            dt_min, dt_save, dt_summarise)

        # Update parameters if they differ from requested values and warn the user
        if actual_dt_save != dt_save:
            self.loop_parameters['dt_save'] = actual_dt_save
            warn(
                f"dt_save was set to {actual_dt_save}s, because it is not a multiple of dt_min ({dt_min}s). "
                f"dt_save can only save a value after an integer number of steps in a fixed-step integrator",
                UserWarning)

        if actual_dt_summarise != dt_summarise:
            self.loop_parameters['dt_summarise'] = actual_dt_summarise
            warn(
                f"dt_summarise was set to {actual_dt_summarise}s, because it is not a multiple of dt_save ({actual_dt_save}s). "
                f"dt_summarise can only save a value after an integer number of steps in a fixed-step integrator",
                UserWarning)

        return save_every_samples, summarise_every_samples, dt_min


