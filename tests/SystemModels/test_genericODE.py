import pytest
from tests.SystemModels.SystemTester import SystemTester
from CuMC.SystemModels.genericODE import genericODE

testsets=[
        # instantiate_settings is a tuple of (precision, state_names, parameter_names, observable_names, constants, num_drivers)
        # input_data is a tuple of (state, parameters, drivers)

        ((np.float32, ["x0"], ["p"], ["o"], {}, 1),
         (np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), np.asarray([3.0], dtype=np.float32)),
         "Single state, param, observable, no constants, 1 driver"),
        ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1"], {'c1': 2.0}, 2),
         (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64), np.asarray([4.2, 1.8], dtype=np.float64)),
         "Two states, two params, one observable, one constant, 2 drivers")
    ]

@pytest.mark.parametrize("instantiate_settings, input_data, test_name",
                         testsets,
                         ids=[testset[2] for testset in testsets])
class TestGenericODE(SystemTester):
    """Example subclass using genericODE as the system under test."""

    @pytest.fixture(scope="class", autouse=True)
    def system_class(self):
        return genericODE

    def correct_answer(self, instantiate_settings, input_data):
        """Override to produce custom expected output."""
        precision, s, pars, obs, cdict, n_drv = instantiate_settings
        state, params, drivers = input_data

        n_states = len(s)
        n_obs = len(obs)
        n_params = len(pars)
        n_constants = len(cdict)
        n_drivers = n_drv

        state_output = np.zeros(n_states, dtype=precision)
        observables = np.zeros(n_obs, dtype=precision)

        for i in range(n_states):
            if n_params > 0:
                param = pars[i % n_params]
            else:
                param = precision(0.0)
            state_output[i] = state[i] + params[i % n_params]
        for i in range(n_obs):
            if n_drivers > 0:
                driver = drivers[i % n_drivers]
            else:
                driver = precision(0.0)
            if n_constants > 0:
                constant_keys = list(cdict.keys())
                constant = self.system_instance.constants[i % n_constants]
            else:
                constant = precision(0.0)
            observables[i] = driver + constant

        return state_output, observables