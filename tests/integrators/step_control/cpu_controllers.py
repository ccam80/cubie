"""Lightweight CPU implementations of adaptive step controllers.

The real controllers live in :mod:`cubie.integrators.step_control` and
provide GPU device functions.  These helpers mirror the gain logic using
pure Python and NumPy so that unit tests can assert behaviour without
depending on implementation details of the CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class StepResult:
    """Container for the outcome of a controller step.

    Parameters
    ----------
    dt
        Proposed step size after applying gain and clamping.
    accepted
        Integer acceptance flag (``1`` for accept, ``0`` for reject).
    local_mem
        Updated controller local memory as a NumPy array.
    """

    dt: float
    accepted: int
    local_mem: NDArray[np.floating]


class CPUAdaptiveController:
    """Common utilities shared by the CPU controller implementations."""

    def __init__(
        self,
        *,
        precision: type,
        dt_min: float,
        dt_max: float,
        atol: NDArray[np.floating],
        rtol: NDArray[np.floating],
        order: int,
        safety: float,
        min_gain: float,
        max_gain: float,
    ) -> None:
        self.precision = precision
        self.atol = np.asarray(atol, dtype=precision)
        self.rtol = np.asarray(rtol, dtype=precision)
        self.order = float(order)
        self.safety = float(safety)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.n_states = int(self.atol.shape[0])

    def _prepare_state(
        self,
        error: Sequence[float],
        state: Optional[Sequence[float]],
        state_prev: Optional[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays representing the controller state.

        Parameters
        ----------
        error, state, state_prev
            Error estimates and state vectors.  Missing values default to
            zeros, matching the GPU implementation used in tests.

        Returns
        -------
        tuple of numpy.ndarray
            Arrays with dtype equal to the configured precision.
        """

        err = np.asarray(error, dtype=self.precision)
        if err.shape[0] != self.n_states:
            msg = (
                "Error vector length does not match controller state count. "
                f"Expected {self.n_states}, received {err.shape[0]}."
            )
            raise ValueError(msg)

        if state is None:
            state_arr = np.zeros_like(err)
        else:
            state_arr = np.asarray(state, dtype=self.precision)

        if state_prev is None:
            state_prev_arr = np.zeros_like(err)
        else:
            state_prev_arr = np.asarray(state_prev, dtype=self.precision)

        return err, state_arr, state_prev_arr

    def _compute_norm(
        self,
        error: np.ndarray,
        state: np.ndarray,
        state_prev: np.ndarray,
    ) -> tuple[float, int]:
        """Calculate the squared error norm and acceptance flag."""

        tol = self.atol + self.rtol * np.maximum(
            np.abs(state), np.abs(state_prev)
        )
        ratio = (tol * tol) / (error * error)
        nrm2 = float(np.sum(ratio))
        accept = int((nrm2 / self.n_states) >= 1.0)
        return nrm2, accept

    def _apply_gain(self, dt: float, gain: float) -> tuple[float, float]:
        """Apply ``gain`` to ``dt`` and clamp to controller bounds."""

        dt_raw = float(dt) * float(gain)
        dt_clamped = min(self.dt_max, max(self.dt_min, dt_raw))
        return dt_raw, dt_clamped

    def _clamp_gain(self, gain: float) -> float:
        """Clamp the gain to controller limits."""

        return min(self.max_gain, max(self.min_gain, float(gain)))

    def _as_array(self, values: Iterable[float]) -> NDArray[np.floating]:
        """Return ``values`` as an array in the configured precision."""

        return np.asarray(list(values), dtype=self.precision)


class CPUAdaptiveIController(CPUAdaptiveController):
    """Integral controller mirroring :class:`AdaptiveIController`."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self._exponent = 1.0 / (2.0 * (1.0 + self.order))

    def step(
        self,
        dt: float,
        error: Sequence[float],
        *,
        state: Optional[Sequence[float]] = None,
        state_prev: Optional[Sequence[float]] = None,
        local_mem: Optional[Sequence[float]] = None,
    ) -> StepResult:
        """Execute a single controller step on the CPU."""

        del local_mem  # No local memory for the integral controller.
        err, state_arr, state_prev_arr = self._prepare_state(
            error, state, state_prev
        )
        nrm2, accepted = self._compute_norm(err, state_arr, state_prev_arr)
        gain_tmp = self.safety * (nrm2 ** self._exponent)
        gain = self._clamp_gain(gain_tmp)
        _, dt_new = self._apply_gain(dt, gain)
        return StepResult(dt=dt_new, accepted=accepted, local_mem=self._as_array([]))


class CPUAdaptivePIController(CPUAdaptiveController):
    """Proportional–integral CPU controller matching the CUDA version."""

    def __init__(self, *, kp: float, ki: float, **kwargs: float) -> None:
        super().__init__(**kwargs)
        denom = 2.0 * (1.0 + self.order)
        self._kp_exponent = kp / denom
        self._ki_exponent = ki / denom

    def step(
        self,
        dt: float,
        error: Sequence[float],
        *,
        state: Optional[Sequence[float]] = None,
        state_prev: Optional[Sequence[float]] = None,
        local_mem: Optional[Sequence[float]] = None,
    ) -> StepResult:
        """Execute the proportional–integral control step."""

        if local_mem is None:
            raise ValueError("PI controller requires previous error history.")

        err_prev = float(local_mem[0])
        err, state_arr, state_prev_arr = self._prepare_state(
            error, state, state_prev
        )
        nrm2, accepted = self._compute_norm(err, state_arr, state_prev_arr)
        pgain = nrm2 ** self._kp_exponent
        igain = err_prev ** self._ki_exponent
        gain_tmp = self.safety * pgain * igain
        gain = self._clamp_gain(gain_tmp)
        _, dt_new = self._apply_gain(dt, gain)
        updated = self._as_array([nrm2])
        return StepResult(dt=dt_new, accepted=accepted, local_mem=updated)


class CPUAdaptivePIDController(CPUAdaptiveController):
    """Proportional–integral–derivative CPU controller."""

    def __init__(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        **kwargs: float,
    ) -> None:
        super().__init__(**kwargs)
        denom = 2.0 * (1.0 + self.order)
        self._kp_exponent = kp / denom
        self._ki_exponent = ki / denom
        self._kd_exponent = kd / denom

    def step(
        self,
        dt: float,
        error: Sequence[float],
        *,
        state: Optional[Sequence[float]] = None,
        state_prev: Optional[Sequence[float]] = None,
        local_mem: Optional[Sequence[float]] = None,
    ) -> StepResult:
        """Execute the PID control step."""

        if local_mem is None or len(local_mem) != 2:
            raise ValueError("PID controller requires two error history values.")

        err_prev = float(local_mem[0])
        err_prev2 = float(local_mem[1])
        err, state_arr, state_prev_arr = self._prepare_state(
            error, state, state_prev
        )
        nrm2, accepted = self._compute_norm(err, state_arr, state_prev_arr)
        pgain = nrm2 ** self._kp_exponent
        igain = err_prev ** self._ki_exponent
        dgain = err_prev2 ** self._kd_exponent
        gain_tmp = self.safety * pgain * igain * dgain
        gain = self._clamp_gain(gain_tmp)
        _, dt_new = self._apply_gain(dt, gain)
        updated = self._as_array([nrm2, err_prev])
        return StepResult(dt=dt_new, accepted=accepted, local_mem=updated)


class CPUGustafssonController(CPUAdaptiveController):
    """CPU implementation of the Gustafsson predictive controller."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(**kwargs)
        self._exponent = 1.0 / (2.0 * (1.0 + self.order))

    def step(
        self,
        dt: float,
        error: Sequence[float],
        *,
        state: Optional[Sequence[float]] = None,
        state_prev: Optional[Sequence[float]] = None,
        local_mem: Optional[Sequence[float]] = None,
    ) -> StepResult:
        """Execute the Gustafsson control step."""

        if local_mem is None or len(local_mem) != 2:
            raise ValueError(
                "Gustafsson controller requires previous step and error values."
            )

        dt_prev = float(local_mem[0])
        err_prev = float(local_mem[1])
        err, state_arr, state_prev_arr = self._prepare_state(
            error, state, state_prev
        )
        nrm2, accepted = self._compute_norm(err, state_arr, state_prev_arr)
        gain_basic = self.safety * (nrm2 ** self._exponent)
        gain = gain_basic
        new_dt_prev = dt_prev
        new_err_prev = err_prev
        if accepted and dt_prev > 0.0 and err_prev > 0.0:
            ratio = nrm2 / err_prev
            gain_gus = self.safety * (dt / dt_prev) * (ratio ** self._exponent)
            gain = min(gain_gus, gain_basic)
            new_dt_prev = dt
            new_err_prev = max(nrm2, 1e-4)
        gain = self._clamp_gain(gain)
        _, dt_new = self._apply_gain(dt, gain)
        updated = self._as_array([new_dt_prev, new_err_prev])
        return StepResult(dt=dt_new, accepted=accepted, local_mem=updated)

