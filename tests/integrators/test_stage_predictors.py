"""Unit tests for the dense stage predictor factory."""

import numpy as np
import pytest

from cubie import create_ODE_system, solve_ivp
from cubie.integrators.algorithms.base_algorithm_step import (
    ButcherTableau,
)
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRK_TABLEAU_REGISTRY,
    RADAU_IIA_5_TABLEAU,
)
from cubie.integrators.stage_predictors import (
    DenseStagePredictor,
    dense_predictor_matrix,
    dense_predictor_ratio_coefficients,
    tableau_supports_dense_prediction,
)


REPEATED_NODE_TABLEAU = ButcherTableau(
    a=((0.5, 0.0), (0.0, 0.5)),
    b=(0.5, 0.5),
    c=(0.5, 0.5),
    order=1,
)

ZERO_NODE_SINGULAR_TABLEAU = ButcherTableau(
    a=((0.0, 0.0), (0.5, 0.5)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
)


def test_repeated_node_tableau_is_rejected():
    """Coincident stage times pin the curve twice; no curve exists."""
    assert not tableau_supports_dense_prediction(REPEATED_NODE_TABLEAU)
    with pytest.raises(ValueError, match="distinct"):
        dense_predictor_matrix(REPEATED_NODE_TABLEAU)


def test_zero_node_singular_tableau_is_supported():
    """Derivative samples need neither nonzero nodes nor invertible A."""
    assert tableau_supports_dense_prediction(
        ZERO_NODE_SINGULAR_TABLEAU
    )


NEAR_DUPLICATE_NODE_TABLEAU = ButcherTableau(
    a=((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)),
    b=(0.4, 0.3, 0.3),
    c=(0.5, 0.5 + 1e-9, 1.0),
    order=1,
)


def test_near_duplicate_node_tableau_is_rejected():
    """Nearly coincident nodes amplify sample error into garbage."""
    assert not tableau_supports_dense_prediction(
        NEAR_DUPLICATE_NODE_TABLEAU
    )
    with pytest.raises(ValueError, match="amplifies"):
        dense_predictor_matrix(NEAR_DUPLICATE_NODE_TABLEAU)


def test_registry_tableaus_support_expectations():
    """Every registry tableau lands on the expected side of the gate."""
    firk_support = {
        name: tableau_supports_dense_prediction(tableau)
        for name, tableau in FIRK_TABLEAU_REGISTRY.items()
    }
    assert firk_support == {
        "firk_gauss_legendre_2": True,
        "firk_gauss_legendre_4": True,
        "radau_iia_5": True,
        "radau": True,
    }
    dirk_support = {
        name: tableau_supports_dense_prediction(tableau)
        for name, tableau in DIRK_TABLEAU_REGISTRY.items()
    }
    assert dirk_support == {
        "implicit_midpoint": True,
        "trapezoidal_dirk": True,
        "ode23t": True,
        "kvaerno3": False,
        "kvaerno5": False,
        "lobatto_iiic_3": True,
        "sdirk_2_2": True,
        "l_stable_dirk_3": True,
        "l_stable_sdirk_4": True,
    }


@pytest.mark.parametrize(
    "tableau",
    [
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        DIRK_TABLEAU_REGISTRY["lobatto_iiic_3"],
        RADAU_IIA_5_TABLEAU,
    ],
)
@pytest.mark.parametrize("step_ratio", [0.5, 1.0, 1.7])
def test_dense_predictor_reads_derivative_samples_ahead(
    tableau, step_ratio
):
    """The matrix equals an independent derivative-sample fit."""
    stage_nodes = np.asarray(tableau.c)
    increments = np.arange(1, tableau.stage_count + 1, dtype=float)
    polynomial = np.polynomial.polynomial.polyfit(
        stage_nodes, increments, tableau.stage_count - 1
    )
    expected = step_ratio * np.polynomial.polynomial.polyval(
        1.0 + step_ratio * stage_nodes, polynomial
    )
    predictor = dense_predictor_matrix(tableau, step_ratio)
    np.testing.assert_allclose(predictor @ increments, expected)


@pytest.mark.parametrize(
    "tableau",
    [DEFAULT_FIRK_TABLEAU, RADAU_IIA_5_TABLEAU],
)
@pytest.mark.parametrize("step_ratio", [0.5, 1.0, 1.7])
def test_dense_predictor_extrapolates_stage_curve(tableau, step_ratio):
    """For collocation tableaus the read-ahead extrapolates the
    state collocation polynomial, checked by an independent fit."""
    stage_matrix = np.asarray(tableau.a)
    stage_nodes = np.asarray(tableau.c)
    increments = np.arange(1, tableau.stage_count + 1, dtype=float)
    old_stage_values = stage_matrix @ increments

    polynomial = np.polynomial.polynomial.polyfit(
        np.concatenate(([0.0], stage_nodes)),
        np.concatenate(([0.0], old_stage_values)),
        tableau.stage_count,
    )
    shifted_stage_values = np.polynomial.polynomial.polyval(
        1.0 + step_ratio * stage_nodes, polynomial
    ) - np.polynomial.polynomial.polyval(1.0, polynomial)
    expected = np.linalg.solve(stage_matrix, shifted_stage_values)

    predictor = dense_predictor_matrix(tableau, step_ratio)
    np.testing.assert_allclose(predictor @ increments, expected)


@pytest.mark.parametrize(
    "tableau",
    [
        DEFAULT_FIRK_TABLEAU,
        RADAU_IIA_5_TABLEAU,
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        DIRK_TABLEAU_REGISTRY["lobatto_iiic_3"],
    ],
)
@pytest.mark.parametrize("step_ratio", [0.65, 1.3])
def test_ratio_coefficients_reconstruct_matrix(tableau, step_ratio):
    """Summing the power stack reproduces the matrix at any ratio."""
    coefficients = dense_predictor_ratio_coefficients(tableau)
    reconstructed = np.zeros_like(coefficients[0])
    for power_index in range(tableau.stage_count):
        reconstructed += (
            coefficients[power_index] * step_ratio ** (power_index + 1)
        )
    direct = dense_predictor_matrix(tableau, step_ratio)
    scale = np.max(np.abs(direct))
    assert np.max(np.abs(reconstructed - direct)) < 1e-12 * scale


def test_radau_dense_predictor_coefficients_are_pinned():
    expected = np.asarray(
        [
            [0.19107136, -0.89141154, 1.7003402],
            [1.5580782, -5.5244045, 4.9663267],
            [3.2735543, -10.606888, 8.333333],
        ],
        dtype=np.float32,
    )

    actual = np.asarray(
        dense_predictor_matrix(RADAU_IIA_5_TABLEAU),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_predictor_update_recognises_parameters(precision):
    """The predictor participates in the standard update flow."""
    predictor = DenseStagePredictor(
        precision=precision,
        n=3,
        tableau=DEFAULT_FIRK_TABLEAU,
    )
    recognised = predictor.update(
        previous_step_size_location="shared", n=5
    )
    assert recognised == {"previous_step_size_location", "n"}
    assert callable(predictor.device_function)


ITERATION_CASES = [
    pytest.param(
        "firk",
        {"step_controller": "fixed", "dt": 0.005},
        id="firk-fixed",
    ),
    pytest.param(
        "l_stable_dirk_3",
        {"step_controller": "fixed", "dt": 0.005},
        id="dirk-fixed",
    ),
    pytest.param(
        "radau",
        {
            "step_controller": "gustafsson",
            "dt_min": 1e-6,
            "dt_max": 0.02,
            "atol": 1e-6,
            "rtol": 1e-6,
        },
        id="firk-adaptive",
    ),
]


@pytest.mark.parametrize("method,controller_settings", ITERATION_CASES)
def test_dense_prediction_reduces_newton_iterations(
    precision, method, controller_settings
):
    """A live predictor strictly beats carried increments.

    Guards against the predictor silently compiling to a no-op
    (e.g. its persistent step-size scalar missing from the loop's
    layout): converged results cannot distinguish a dead predictor,
    but the Newton iteration count can.
    """
    system = create_ODE_system(
        "\n".join(
            (
                "dx = sigma*(y - x)",
                "dy = x*(rho - z) - y",
                "dz = x*y - b_const*z",
            )
        ),
        states={"x": 1.05, "y": 0.97, "z": 1.02},
        parameters={"sigma": 10.0, "rho": 28.0},
        constants={"b_const": 8.0 / 3.0},
        precision=precision,
        name="dense_prediction_iteration_probe",
    )
    common = {
        "y0": {
            "x": np.array([1.05]),
            "y": np.array([0.97]),
            "z": np.array([1.02]),
        },
        "grid_type": "verbatim",
        "method": method,
        "duration": 0.1,
        "save_every": 0.025,
        "output_types": ["state", "iteration_counters"],
    }
    common.update(controller_settings)

    def newton_total(result):
        assert not np.any(result.status_codes)
        counters = np.asarray(result.iteration_counters)
        return int(counters.sum(axis=0).T[0, 0])

    predicted = newton_total(solve_ivp(system, **common))
    carried = newton_total(
        solve_ivp(system, attempt_dense_prediction=False, **common)
    )
    assert predicted < carried
