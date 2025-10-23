"""CPU reference implementations of step controllers."""

import numpy as np

from cubie._utils import PrecisionDType

from .cpu_utils import Array


class CPUAdaptiveController:
    """Simple adaptive step controller mirroring GPU heuristics."""

    def __init__(
        self,
        *,
        kind: str,
        dt: float,
        dt_min: float,
        dt_max: float,
        atol: float,
        rtol: float,
        order: int,
        precision: PrecisionDType,
        kp: float = 1 / 18,
        ki: float = 1 / 9,
        kd: float = 1 / 18,
        gamma: float = 0.9,
        safety: float = 0.9,
        min_gain: float = 0.5,
        max_gain: float = 2.0,
        max_newton_iters: int = 0,
        deadband_min: float = 1.0,
        deadband_max: float = 1.2,
    ) -> None:
        self.kind = kind.lower()
        self.dt_min = precision(dt_min)
        self.dt_max = precision(dt_max)
        if kind == "fixed":
            self.dt0 = precision(dt)
        else:
            self.dt0 = precision(np.sqrt(dt_min * dt_max))
        self.dt = self.dt0
        self.atol = precision(atol)
        self.rtol = precision(rtol)
        self.order = order
        self.precision = precision
        self.safety = precision(safety)
        self.min_gain = precision(min_gain)
        self.max_gain = precision(max_gain)
        self.kp = precision(kp)
        self.ki = precision(ki)
        self.kd = precision(kd)
        self.gamma = precision(gamma)
        self.max_newton_iters = int(max_newton_iters)
        self.deadband_min = precision(deadband_min)
        self.deadband_max = precision(deadband_max)
        self.unity_gain = precision(1.0)
        self._deadband_disabled = (
            (self.deadband_min == self.unity_gain)
            and (self.deadband_max == self.unity_gain)
        )
        zero = precision(0.0)
        self._history = [zero, zero]
        self._step_count = 0
        self._convergence_failed = False
        self._rejections_at_dt_min = 0
        self._prev_nrm2 = zero
        self._prev_prev_nrm2 = zero
        self._prev_dt = zero

    @property
    def is_adaptive(self) -> bool:
        return self.kind != "fixed"

    @property
    def prev_dt(self) -> np.floating:
        """Return the previous step size used by the controller."""

        return self._prev_dt

    def error_norm(self, state_prev: Array, state_new: Array, error: Array) -> float:
        error = np.maximum(np.abs(error), 1e-12)
        scale = self.atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new)
        )
        nrm2 = np.sum((scale * scale) / (error * error))
        norm = nrm2 / len(error)
        return self.precision(norm)

    def propose_dt(
        self,
        error_vector: Array,
        prev_state: Array,
        new_state: Array,
        niters: int = 0,
    ) -> bool:
        self._step_count += 1
        if not self.is_adaptive:
            return True
        errornorm = self.error_norm(
            state_prev=prev_state,
            state_new=new_state,
            error=error_vector,
        )

        accept = errornorm >= self.precision(1.0)

        current_dt = self.dt
        gain = self._gain(
            errornorm=errornorm,
            accept=accept,
            niters=niters,
            current_dt=current_dt,
        )

        unclamped_dt = self.precision(current_dt * gain)
        new_dt = min(self.dt_max, max(self.dt_min, unclamped_dt))
        self.dt = new_dt
        self._prev_dt = current_dt
        self._prev_prev_nrm2 = self._prev_nrm2
        self._prev_nrm2 = errornorm

        if unclamped_dt < self.dt_min:
            raise ValueError(
                f"dt < dt_min: {unclamped_dt} < {self.dt_min} exceeded"
            )

        return accept

    def _gain(
        self,
        *,
        errornorm: float,
        accept: bool,
        niters: int,
        current_dt: float,
    ) -> float:
        precision = self.precision
        expo_fraction = precision(
            precision(1.0)
            / (precision(2) * (precision(self.order) + precision(1)))
        )
        kp_exp = precision(self.kp * expo_fraction)
        ki_exp = precision(self.ki * expo_fraction)
        kd_exp = precision(self.kd * expo_fraction)

        if self.kind == "i":
            exponent = expo_fraction
            gain = self.safety * precision(errornorm ** exponent)

        elif self.kind == "pi":
            prev = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            gain = (
                self.safety
                * precision(errornorm ** kp_exp)
                * precision(prev ** ki_exp)
            )

        elif self.kind == "pid":
            prev_nrm2 = (
                self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            )
            prev_prev = (
                self._prev_prev_nrm2
                if self._prev_prev_nrm2 > 0.0
                else prev_nrm2
            )
            gain = (
                self.safety
                * precision(errornorm ** kp_exp)
                * precision(prev_nrm2 ** ki_exp)
                * precision(prev_prev ** kd_exp)
            )

        elif self.kind == "gustafsson":
            one = precision(1.0)
            two = precision(2.0)
            niters_eff = precision(max(niters, 1))
            M = self.max_newton_iters
            dt_prev = max(precision(1e-16), self._prev_dt)
            nrm2_prev = max(precision(1e-16), self._prev_nrm2)
            fac = min(
                self.gamma,
                ((one + two * M) * self.gamma) / (niters_eff + two * M),
            )
            gain_basic = precision(
                self.safety * fac * (errornorm ** expo_fraction)
            )

            use_gus = (
                accept and (self._prev_dt > 0.0) and (self._prev_nrm2 > 0.0)
            )
            if use_gus:
                ratio = (errornorm * errornorm) / nrm2_prev
                gain_gus = (
                    self.safety
                    * (current_dt / dt_prev)
                    * precision(ratio ** expo_fraction)
                    * self.gamma
                )
                gain = gain_gus if gain_gus < gain_basic else gain_basic
            else:
                gain = gain_basic
        else:
            gain = precision(1.0)

        gain = min(self.max_gain, max(self.min_gain, gain))
        if not self._deadband_disabled:
            if self.deadband_min <= gain <= self.deadband_max:
                gain = self.unity_gain
        return precision(gain)


__all__ = ["CPUAdaptiveController"]
