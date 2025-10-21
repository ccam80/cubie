import numpy as np
import pytest

from cubie.integrators.array_interpolator import ArrayInterpolator
from tests.integrators.cpu_reference.cpu_utils import DriverEvaluator


@pytest.mark.parametrize(
    "precision, atol, rtol",
    [
        (np.float32, 5e-6, 5e-6),
        (np.float64, 1e-12, 1e-12),
    ],
)
def test_cpu_driver_evaluator_matches_gpu_time_alignment(
    precision, atol, rtol
):
    dt = precision(0.125)
    warmup = precision(0.35)
    sample_count = 12
    sample_times = warmup + dt * np.arange(sample_count, dtype=precision)
    driver_values = np.cos(sample_times * precision(1.3))
    driver_values += np.sin(sample_times * precision(0.7))

    driver_dict = {
        "t0": warmup,
        "dt": dt,
        "wrap": False,
        "order": 3,
        "driver": driver_values.astype(precision),
    }

    interpolator = ArrayInterpolator(precision=precision, input_dict=driver_dict)
    evaluator = DriverEvaluator(
        coefficients=interpolator.coefficients,
        dt=precision(interpolator.dt),
        t0=precision(interpolator.t0),
        wrap=interpolator.wrap,
        precision=precision,
        boundary_condition=interpolator.boundary_condition,
    )

    query_times = np.linspace(
        warmup - dt,
        warmup + dt * precision(sample_count - 1) + dt,
        num=40,
        dtype=precision,
    )
    gpu_values = interpolator.get_interpolated(query_times)
    cpu_values = np.stack(
        [evaluator.evaluate(float(time)) for time in query_times]
    )

    np.testing.assert_allclose(cpu_values, gpu_values, rtol=rtol, atol=atol)
