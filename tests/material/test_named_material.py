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


class NamedMaterialTest(unittest.TestCase):
    """
    Test Material class.
    """

    def test_constructor(self):

        optid.material.NamedMaterial(name='Sm2Co17')

    def test_constructor_bad_name_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', optid.material.NamedMaterial,
                               name='')

    def test_constructor_bad_remanent_magnetization_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', optid.material.NamedMaterial,
                               name='Sm2Co17', remanent_magnetization=-1)

    def test_to_radia(self):

        material = optid.material.NamedMaterial(name='Sm2Co17')

        self.assertIsNotNone(material.to_radia([0, 0, 0]))

    def test_to_radia_remanent_magnetization(self):

        material = optid.material.NamedMaterial(name='Sm2Co17', remanent_magnetization=1)

        self.assertIsNotNone(material.to_radia([0, 0, 0]))

