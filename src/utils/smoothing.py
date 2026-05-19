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


class OneEuroPoseSmoother:
    """1 Euro Filter applied to position (adaptive cutoff) and rotation (SLERP)."""

    def __init__(self, min_cutoff: float = 10.0, beta: float = 0.8, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.t_prev = None

    def reset(self) -> None:
        """Clear state so next sample is treated as a fresh start (call on INVALID→valid transition)."""
        self.p_prev = None
        self.q_prev = None
        self.dp_prev = np.zeros(3)
        self.t_prev = None

    def smooth(self, t: float, target_pose: np.ndarray | None) -> np.ndarray | None:
        if target_pose is None:
            return None

        t_p = target_pose[0:3]
        t_q = target_pose[3:7]

        if self.t_prev is None or self.p_prev is None:
            self.p_prev = t_p.copy()
            self.q_prev = t_q.copy()
            self.t_prev = t
            return target_pose.copy()

        dt = t - self.t_prev
        if dt <= 0.0:
            return target_pose.copy()

        def get_alpha(dt: float, cutoff: float) -> float:
            tau = 1.0 / (2 * np.pi * cutoff)
            return 1.0 / (1.0 + tau / dt)

        dp_raw = (t_p - self.p_prev) / dt
        alpha_d = get_alpha(dt, self.d_cutoff)
        dp_filtered = alpha_d * dp_raw + (1.0 - alpha_d) * self.dp_prev

        speed = np.linalg.norm(dp_filtered)
        cutoff_p = self.min_cutoff + self.beta * speed

        alpha_p = get_alpha(dt, cutoff_p)
        p_hat = self.p_prev + alpha_p * (t_p - self.p_prev)
        q_hat = _slerp_quat(self.q_prev, t_q, alpha_p)

        self.p_prev = p_hat
        self.q_prev = q_hat
        self.dp_prev = dp_filtered
        self.t_prev = t

        return np.array(
            [p_hat[0], p_hat[1], p_hat[2], q_hat[0], q_hat[1], q_hat[2], q_hat[3]],
            dtype=np.float32,
        )
