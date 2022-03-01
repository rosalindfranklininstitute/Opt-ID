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
import numpy as np

# Test imports
import optid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class MaterialTest(unittest.TestCase):
    """
    Test Material class.
    """

    def test_to_radia_list(self):

        material = optid.material.Material()

        self.assertIsNone(material.to_radia([0, 0, 0]))

    def test_to_radia_array(self):

        material = optid.material.Material()

        self.assertIsNone(material.to_radia(np.array([0, 0, 0], dtype=np.float32)))

    def test_to_radia_bad_type_raises_exception(self):

        material = optid.material.Material()

        self.assertRaisesRegex(TypeError, '.*', material.to_radia,
                               vector=np.array([0, 0, 0], dtype=np.int32))

    def test_to_radia_bad_shape_raises_exception(self):

        material = optid.material.Material()

        self.assertRaisesRegex(ValueError, '.*', material.to_radia,
                               vector=np.array([0, 0, 0, 0], dtype=np.float32))
