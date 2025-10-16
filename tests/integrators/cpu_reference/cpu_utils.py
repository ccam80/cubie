"""Shared utilities for CPU reference integrator implementations."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from cubie.integrators import IntegratorReturnCodes


Array = NDArray[np.floating]
STATUS_MASK = 0xFFFF


def _ensure_array(vector: Sequence[float] | Array, dtype: np.dtype) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.atleast_1d(vector).astype(dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


@dataclass
class StepResult:
    """Container describing the outcome of a single integration step."""

    state: Array
    observables: Array
    error: Array
    status: int = 0
    niters: int = 0


def _encode_solver_status(converged: bool, niters: int) -> int:
    """Return a solver status word with the Newton iteration count encoded."""

    base_code = (
        IntegratorReturnCodes.SUCCESS
        if converged
        else IntegratorReturnCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
    )
    iter_count = max(0, min(int(niters), STATUS_MASK))
    return (iter_count << 16) | (int(base_code) & STATUS_MASK)


class DriverEvaluator:
    """Evaluate spline coefficients to recover driver samples on the CPU."""

    def __init__(
        self,
        coefficients: Optional[Array],
        dt: float,
        t0: float,
        wrap: bool,
        precision: np.dtype,
        *,
        boundary_condition: Optional[str] = "not-a-knot",
    ) -> None:
        coeffs = (
            np.zeros((0, 0, 1), dtype=precision)
            if coefficients is None
            else np.array(coefficients, dtype=precision, copy=True)
        )
        if coeffs.ndim != 3:
            raise ValueError(
                "Driver coefficients must have shape (segments, drivers, "
                "order + 1)."
            )
        self.precision = precision
        self.coefficients = coeffs
        self.dt = precision(dt)
        self.t0 = precision(t0)
        self.wrap = bool(wrap)
        self.boundary_condition = boundary_condition
        self._segments = coeffs.shape[0]
        self._width = coeffs.shape[1]
        self._zero = np.zeros(self._width, dtype=precision)
        pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        self._evaluation_start = self.t0 - (self.dt if pad_clamped else 0.0)

    def evaluate(self, time: float) -> Array:
        """Return driver values interpolated at ``time``."""

        precision = self.precision

        if self._width == 0 or self._segments == 0 or self.dt == 0.0:
            return self._zero.copy()

        inv_res = 1.0 / self.dt
        scaled = (time - self._evaluation_start) * inv_res
        scaled_floor = math.floor(scaled)
        idx = int(scaled_floor)

        if self.wrap and self._segments > 0:
            segment = idx % self.coefficients.shape[0]
            if segment < 0:
                segment += self.coefficients.shape[0]
            tau = scaled - scaled_floor
            in_range = True
        else:
            in_range = 0.0 <= scaled <= precision(self.coefficients.shape[0])
            segment = idx if idx >= 0 else 0
            if segment >= self.coefficients.shape[0]:
                segment = self.coefficients.shape[0] - 1
            tau = scaled - precision(segment)

        values = self._zero.copy()
        for driver_idx in range(self._width):
            segment_coeffs = self.coefficients[segment, driver_idx]
            acc = precision(0.0)
            for coeff in reversed(segment_coeffs):
                acc = acc * tau + precision(coeff)
            if self.wrap or in_range:
                values[driver_idx] = acc
            else:
                values = self._zero.copy()
        return values

    def __call__(self, time: float) -> Array:
        """Alias for :meth:`evaluate` so instances are callable."""

        return self.evaluate(time)

    def with_coefficients(self, coefficients: Optional[Array]) -> "DriverEvaluator":
        """Return a new evaluator with ``coefficients`` but shared timing."""

        return DriverEvaluator(
            coefficients=coefficients,
            dt=self.dt,
            t0=self.t0,
            wrap=self.wrap,
            precision=self.precision,
            boundary_condition=self.boundary_condition,
        )


__all__ = [
    "Array",
    "DriverEvaluator",
    "STATUS_MASK",
    "StepResult",
    "_encode_solver_status",
    "_ensure_array",
]
