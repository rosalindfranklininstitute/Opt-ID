# Copyright 2017 Diamond Light Source
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


# Utility imports
import unittest

# Test imports
import optid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class PoseTest(unittest.TestCase):
    """
    Test device Pose class.
    """

    def test_constructor(self):

        optid.device.Pose(gap=0, phase=0)
        optid.device.Pose(gap=1, phase=1)

    def test_constructor_negative_gap_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', optid.device.Pose,
                               gap=-1, phase=0)

    def test_constructor_negative_phase_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', optid.device.Pose,
                               gap=0, phase=-1)

    def test_repr(self):

        self.assertEqual(str(optid.device.Pose(gap=0, phase=0)), 'Pose(gap=0, phase=0)')
        self.assertEqual(str(optid.device.Pose(gap=1, phase=1)), 'Pose(gap=1, phase=1)')

    def test_eq(self):

        self.assertTrue(optid.device.Pose(gap=0, phase=0) == optid.device.Pose(gap=0, phase=0))
        self.assertTrue(optid.device.Pose(gap=1, phase=2) == optid.device.Pose(gap=1, phase=2))
        self.assertFalse(optid.device.Pose(gap=0, phase=0) == optid.device.Pose(gap=1, phase=0))
        self.assertFalse(optid.device.Pose(gap=0, phase=0) == optid.device.Pose(gap=0, phase=1))
        self.assertFalse(optid.device.Pose(gap=0, phase=0) == 'None')

    def test_hash(self):

        self.assertIsNotNone(hash(optid.device.Pose(gap=0, phase=0)))
