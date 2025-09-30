"""Plot CUDA driver array interpolation against SciPy references.

The example builds a richly featured driver array with several harmonic and
modulated components, compiles the :class:`~cubie.integrators.driver_array.
DriverArray` device function, and then compares GPU-evaluated samples against
SciPy's :class:`scipy.interpolate.CubicSpline`. The script requires SciPy,
NumPy, Matplotlib, and a CUDA-capable device (or Numba's CUDA simulator).
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from scipy.interpolate import CubicSpline

from cubie.integrators.driver_array import DriverArray


def build_wiggly_driver(
    num_samples: int = 512, duration: float = 32.0
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a long driver array with rapidly changing structure.

    Parameters
    ----------
    num_samples : int, optional
        Number of temporal samples to generate.
    duration : float, optional
        Total duration of the driver timeline in seconds.

    Returns
    -------
    tuple of numpy.ndarray
        Pair of time samples and driver values.
    """

    times = np.linspace(0.0, duration, num_samples, dtype=np.float64)
    base = np.sin(0.6 * times) + 0.2 * np.sin(4.2 * times + 0.7)
    chirp = 0.35 * np.sin(0.15 * times ** 1.4)
    ripples = 0.5 * np.sin(21.0 * times) + 0.05 * np.cos(39.0 * times)
    envelope = 1.0 + 0.3 * np.sin(0.25 * times)
    values = envelope * (base + chirp + ripples)
    return times, values.astype(np.float64)


def evaluate_on_device(
    device_fn: Callable[[float, np.ndarray, np.ndarray], None],
    coefficients: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate the driver array device function on supplied samples.

    Parameters
    ----------
    device_fn : callable
        CUDA device function produced by :class:`DriverArray`.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients returned by ``build``.
    query_times : numpy.ndarray
        Times at which to evaluate the driver array.

    Returns
    -------
    numpy.ndarray
        Device-evaluated driver values for each query time.
    """

    out_host = np.empty((query_times.size, coefficients.shape[1]))

    @cuda.jit
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.device_array_like(out_host)
    threads_per_block = 128
    blocks = (query_times.size + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_coeffs, d_out)
    return d_out.copy_to_host(out_host)


def main() -> None:
    """Generate plots comparing CUDA and SciPy spline interpolation."""
    precision = np.float64
    times, samples = build_wiggly_driver()
    driver = DriverArray(
        precision=precision,
        drivers_dict = {
            'wiggler': samples,
            'time': times
        },
        order=3,
        wrap=False
    )
    cache = driver.build()

    dense_times = np.linspace(times[0], times[-1], 8192, dtype=np.float64)
    device_values = evaluate_on_device(
        cache.device_function, cache.coefficients, dense_times
    )

    spline = CubicSpline(times, samples)
    scipy_values = spline(dense_times)

    plt.figure(figsize=(12, 6))
    plt.plot(times, samples, label="samples", alpha=0.5, linewidth=1.0)
    plt.plot(dense_times, device_values[:, 0], label="CUDA", linewidth=1.5)
    plt.plot(dense_times, scipy_values, label="SciPy", linewidth=1.0)
    plt.title("Driver array interpolation comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Driver value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
