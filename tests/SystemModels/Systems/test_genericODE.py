import pytest
from tests.SystemModels.SystemTester import SystemTester
from CuMC.SystemModels.Systems.GenericODE import GenericODE
import numpy as np

#These are hard-coded in the generalODE class as it does not have a fixed set of parameters, states, or observables.
# For a custom class, there are helper functions to generate tests sets in tests.SystemsModels._utils
testsets = [
    # instantiate_settings is a tuple of (precision, state_names, parameter_names, observable_names, constants, num_drivers)
    # input_data is a tuple of (state, parameters, drivers)

    ((np.float32, ["x0"], ["p"], ["o"], {}, 1),
     (np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), np.asarray([3.0], dtype=np.float32)),
     "Single state, param, observable, no constants, 1 driver"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1"], {'c1': 2.0}, 2),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8], dtype=np.float64)
      ),
     "Two states, two params, one observable, one constant, 2 drivers"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1", "o2"], {'c1': 2.0}, 1),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2], dtype=np.float64)
      ),
     "Two states, two params, two observables, one constant, 1 driver"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1"], {'c1': 2.0, 'c2': 3.0}, 2),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8], dtype=np.float64)
      ),
     "Two states, two params, one observables, two constants, 2 drivers"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1"], {'c1': 2.0, 'c2': 3.0}, 1),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2], dtype=np.float64)
      ),
     "Two states, two params, one observables, two constants, 1 drivers"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1", "o2"], {'c1': 2.0, 'c2': 3.0}, 2),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8], dtype=np.float64)
      ),
     "Two states, two params, two observables, two constants, 2 drivers"
     ),
    ((np.float64, ["x1", "x2", "x3", "x4", "x5", "x6"], ["p1", "p2"], ["o1", "o2", "o3", "o4", "o5", "o6", ],
      {'c1': 8.0}, 6
      ),
     (np.asarray([1.0, 0.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8, 4.6, 1.7, 4.1, 2.3], dtype=np.float64)
      ),
     "More states and obs (slots) than params and constants (values)"
     ),

    ((np.float32, ["x0"], ["p"], ["o"], {}, 1),
     (np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32),
      np.asarray([3.0], dtype=np.float32)
      ),
     "Single state, param, observable, no constants, 1 driver"
     ),
    ((np.float64, ["x1", "x2"], ["p1", "p2"], ["o1"], {'c1': 2.0}, 2),
     (np.asarray([1.0, 0.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8], dtype=np.float64)
      ),
     "Two states, two params, one observable, one constant, 2 drivers"
     ),
    ((np.float64, ["x1", "x2", "x3", "x4", "x5", "x6"], ["p1", "p2"], ["o1", "o2", "o3", "o4", "o5", "o6", ],
      {'c1': 8.0}, 6
      ),
     (np.asarray([1.0, 0.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float64), np.asarray([0.5, 5.5], dtype=np.float64),
      np.asarray([4.2, 1.8, 4.6, 1.7, 4.1, 2.3], dtype=np.float64)
      ),
     "More states and obs (slots) than params and constants (values)"
     )
    ]


@pytest.mark.parametrize("instantiate_settings, input_data, test_name",
                         testsets,
                         ids=[testset[2] for testset in testsets],
                         )
class TestGenericODE(SystemTester):
    """Example subclass using GenericODE as the system under test."""

    @pytest.fixture(scope="class", autouse=True)
    def system_class(self):
        return GenericODE