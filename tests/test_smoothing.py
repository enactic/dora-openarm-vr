# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from dora_openarm_vr.smoothing import OneEuroPoseSmoother


def _pose(x: float, angle: float) -> np.ndarray:
    return np.array([x, 0.0, 0.0, np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


class OneEuroPoseSmootherTest(unittest.TestCase):
    def test_invalid_limits_are_rejected(self) -> None:
        invalid_options = (
            {"lag_cutoff": 0.0},
            {"lag_cutoff": np.nan},
            {"max_linear_speed": -1.0},
            {"max_angular_speed": np.inf},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                OneEuroPoseSmoother(**options)

    def test_pose_step_is_rate_limited(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            lag_cutoff=10.0,
            max_linear_speed=1.0,
            max_angular_speed=2.0,
        )
        initial = _pose(0.0, 0.0)
        target = _pose(1.0, np.pi)
        smoother.smooth(0.0, initial)

        output = smoother.smooth(0.01, target)

        self.assertLessEqual(np.linalg.norm(output[:3] - initial[:3]), 0.01 + 1e-7)
        self.assertLessEqual(_quaternion_angle(output[3:7], initial[3:7]), 0.02 + 2e-6)

    def test_suspend_recovery_preserves_output_and_initializes_lag(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            lag_cutoff=10.0,
            max_linear_speed=1.0,
            max_angular_speed=2.0,
        )
        smoother.smooth(0.0, _pose(0.0, 0.0))
        before_suspend = smoother.smooth(0.01, _pose(0.2, 0.4))
        smoother.suspend(0.02)
        recovered_target = _pose(1.0, np.pi)

        recovered_output = smoother.smooth(0.024, recovered_target)

        self.assertLessEqual(
            np.linalg.norm(recovered_output[:3] - before_suspend[:3]), 0.004 + 1e-7
        )
        self.assertFalse(np.allclose(recovered_output, recovered_target))
        np.testing.assert_allclose(
            smoother.lag[:3],
            recovered_target[:3] - before_suspend[:3],
        )
        self.assertAlmostEqual(
            _quaternion_angle(
                smoother.lag[3:],
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            _quaternion_angle(recovered_target[3:7], before_suspend[3:7]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
