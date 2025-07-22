import pytest
from SystemModels.Systems.SystemTester import SystemTester
from tests.SystemModels._utils import generate_system_tests
from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel

testsets = generate_system_tests(ThreeChamberModel, (-6, 6))

@pytest.mark.parametrize("instantiate_settings, input_data, test_name",
                         testsets,
                         ids=[testset[2] for testset in testsets])
class TestThreeCM(SystemTester):
    """Testing class for the Three Chamber model. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    @pytest.fixture(scope="class", autouse=True)
    def system_class(self):
        return ThreeChamberModel

