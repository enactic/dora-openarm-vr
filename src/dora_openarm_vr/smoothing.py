# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


_IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def _slerp_quat(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        res = q1 + alpha * (q2 - q1)
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q1 + s1 * q2


def _quat_error(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Return q_to * inverse(q_from) in wxyz order."""
    w_from, v_from = q_from[0], q_from[1:]
    w_to, v_to = q_to[0], q_to[1:]
    error = np.concatenate(
        (
            [w_to * w_from + np.dot(v_to, v_from)],
            -w_to * v_from + w_from * v_to - np.cross(v_to, v_from),
        )
    )
    return error / np.linalg.norm(error)


def _limit_scale(step: float, maximum: float) -> float:
    return min(1.0, maximum / step) if maximum > 0.0 and step > 0.0 else 1.0


class OneEuroPoseSmoother:
    """1 Euro Filter applied to position (adaptive cutoff) and rotation (SLERP)."""

    def __init__(
        self,
        min_cutoff: float = 10.0,
        beta: float = 0.8,
        d_cutoff: float = 1.0,
        lag_cutoff: float = float("inf"),
        max_linear_speed: float = 0.0,
        max_angular_speed: float = 0.0,
    ):
        if np.isnan(lag_cutoff) or lag_cutoff <= 0.0:
            raise ValueError("lag_cutoff must be positive")
        if not np.isfinite(max_linear_speed) or max_linear_speed < 0.0:
            raise ValueError("max_linear_speed must be finite and non-negative")
        if not np.isfinite(max_angular_speed) or max_angular_speed < 0.0:
            raise ValueError("max_angular_speed must be finite and non-negative")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.lag_cutoff = lag_cutoff
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.p_lag_prev = None
        self.q_lag_prev = None
        self.t_prev = None

    @property
    def lag(self) -> np.ndarray | None:
        if self.p_lag_prev is None or self.q_lag_prev is None:
            return None
        return np.concatenate([self.p_lag_prev, self.q_lag_prev]).astype(np.float32)

    def reset(self) -> None:
        """Clear all filter state."""
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.p_lag_prev = None
        self.q_lag_prev = None
        self.t_prev = None

    def suspend(self, t: float) -> None:
        """Pause updates while preserving the last filtered pose."""
        self.dp_prev = np.zeros(3)
        self.p_lag_prev = None
        self.q_lag_prev = None
        self.t_prev = t

    def smooth(self, t: float, target_pose: np.ndarray | None) -> np.ndarray | None:
        if target_pose is None:
            return None

        t_p = target_pose[0:3]
        t_q = target_pose[3:7]

        if self.t_prev is None or self.p_prev is None:
            self.p_prev = t_p.copy()
            self.q_prev = t_q.copy()
            self.p_lag_prev = np.zeros(3)
            self.q_lag_prev = _IDENTITY_QUATERNION.copy()
            self.t_prev = t
            return target_pose.copy()

        dt = t - self.t_prev
        if dt <= 0.0:
            return target_pose.copy()

        def get_alpha(dt: float, cutoff: float) -> float:
            tau = 1.0 / (2 * np.pi * cutoff)
            return 1.0 / (1.0 + tau / dt)

        p_error = t_p - self.p_prev
        q_error = _quat_error(self.q_prev, t_q)

        if self.p_lag_prev is None:
            self.p_lag_prev = p_error
            self.q_lag_prev = _slerp_quat(_IDENTITY_QUATERNION, q_error, 1.0)

        alpha_lag = get_alpha(dt, self.lag_cutoff)
        p_lag = alpha_lag * p_error + (1.0 - alpha_lag) * self.p_lag_prev
        q_lag = _slerp_quat(
            self.q_lag_prev,
            q_error,
            alpha_lag,
        )

        dp_raw = p_error / dt
        alpha_d = get_alpha(dt, self.d_cutoff)
        dp_filtered = alpha_d * dp_raw + (1.0 - alpha_d) * self.dp_prev

        speed = np.linalg.norm(dp_filtered)
        cutoff_p = self.min_cutoff + self.beta * speed

        alpha_p = get_alpha(dt, cutoff_p)
        position_step = alpha_p * p_error
        position_step *= _limit_scale(
            np.linalg.norm(position_step), self.max_linear_speed * dt
        )
        p_hat = self.p_prev + position_step

        q_error_angle = 2.0 * np.arccos(np.clip(abs(q_error[0]), 0.0, 1.0))
        rotation_alpha = alpha_p * _limit_scale(
            alpha_p * q_error_angle,
            self.max_angular_speed * dt,
        )
        q_hat = _slerp_quat(self.q_prev, t_q, rotation_alpha)

        self.p_prev = p_hat
        self.q_prev = q_hat
        self.dp_prev = dp_filtered
        self.p_lag_prev = p_lag
        self.q_lag_prev = q_lag
        self.t_prev = t

        return np.array(
            [p_hat[0], p_hat[1], p_hat[2], q_hat[0], q_hat[1], q_hat[2], q_hat[3]],
            dtype=np.float32,
        )
