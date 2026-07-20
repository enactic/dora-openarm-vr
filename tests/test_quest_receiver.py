"""Tests for Quest controller mappings."""

import unittest

import numpy as np

from dora_openarm_vr import quest_receiver


class QuestReceiverTest(unittest.TestCase):
    def test_calibrated_trigger_mapping_is_preserved(self) -> None:
        self.assertAlmostEqual(
            quest_receiver._map_trigger_to_gripper(0.0, "right"),
            np.deg2rad(-45.0),
        )
        self.assertAlmostEqual(
            quest_receiver._map_trigger_to_gripper(1.0, "right"), np.deg2rad(8.0)
        )
        self.assertAlmostEqual(
            quest_receiver._map_trigger_to_gripper(0.0, "left"), np.deg2rad(45.0)
        )
        self.assertAlmostEqual(
            quest_receiver._map_trigger_to_gripper(1.0, "left"), np.deg2rad(-8.0)
        )


if __name__ == "__main__":
    unittest.main()
