import pytest
from tests.odesystems.systems.SystemTester import SystemTester
from tests.odesystems._utils import generate_system_tests
from cubie.odesystems.systems.threeCM import ThreeChamberModel

testsets = generate_system_tests(ThreeChamberModel, (-6, 6))


@pytest.mark.parametrize(
    "instantiate_settings, input_data, test_name",
    testsets,
    ids=[testset[2] for testset in testsets],
)
class TestThreeCM(SystemTester):
    """Testing class for the Three Chamber model. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    @pytest.fixture(scope="class", autouse=True)
    def system_class(self):
        return ThreeChamberModel

    def test_constants_edit(
        self, system_class, instantiate_settings, input_data, test_name
    ):
        """No constants in ThreeCM, so this test should pass without errors."""
        assert True
