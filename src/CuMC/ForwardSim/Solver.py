import numpy as np


class Solver:
    """
    User-facing class for batch-solving systems of ODEs. Accepts and sanitises user-world inputs, and passes them to
    GPU-world integrating functions. This class instantiates and owns a SolverKernel, which in turn interfaces
    distributes parameter and initial value sets to a groupd of SingleIntegratorRun device functions that perform
    each integration. The only part of this machine that the user must configure themself before using is the system
    model, which contains the ODEs to be solved.
    """


    def _clarify_None_output_functions(self, n_saved_observables):
        """Clarify the output functions to be used in the loop, if None is specified, set to default values."""
        # TODO: add empty list check
        if self.compile_settings['output_functions'] is None:
            self.compile_settings['output_functions'] = ['state']
            if n_saved_observables > 0:
                self.compile_settings['output_functions'].append('observables')

    def _get_saved_values(self, n_states):
        """Sanitise empty lists and None values - statse default to all, observables default to none."""

        #TODO: add a routine to handle saved_state or saved_observables being given as strings - figure out at which level this should
        # happen and whether it can just call one of them fancy systemvalues functions.
        saved_states = self.compile_settings['saved_states']
        saved_observables = self.compile_settings['saved_observables']

        # If no saved states specified, assume all states are saved.
        if saved_states is None:
            saved_states = np.arange(n_states, dtype=np.int16)
        n_saved_states = len(saved_states)

        # On the other hand, if no observables are specified, assume no observables are saved.
        if saved_observables is None:
            saved_observables = []
        n_saved_observables = len(saved_observables)

        return saved_states, saved_observables, n_saved_states, n_saved_observables

    def get_set_at_output_index(self, inits_sets, params_sets, idx):
        """ Returns the initial values and parameters that correspond to a particular (idx) slice of the output arrays.
        """
        num_inits = inits_sets.shape[0]
        init_index = idx % num_inits
        param_index = idx // num_inits
        return inits_sets[init_index, :], params_sets[param_index, :]

