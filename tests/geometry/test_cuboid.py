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
from beartype.roar import BeartypeException
import sys
import unittest
import numpy as np

# Test imports
import optid
from optid.geometry import Cuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class CuboidTest(unittest.TestCase):
    """
    Test Cuboid class.
    """

    ####################################################################################################################

    def test_constructor_shape_array(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        geometry = Cuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        # polyhedra = [
        #     [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
        #     [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
        #     [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
        #     [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
        #     [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
        #     [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        # self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_shape_list(self):

        shape = [1, 1, 1]

        geometry = Cuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        # polyhedra = [
        #     [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
        #     [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
        #     [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
        #     [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
        #     [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
        #     [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        # self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_shape_tuple(self):

        shape = (1, 1, 1)

        geometry = Cuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        # polyhedra = [
        #     [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
        #     [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
        #     [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
        #     [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
        #     [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
        #     [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        # self.assertEqual(geometry.polyhedra, polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shape_type_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', Cuboid,
                               shape=None)

    def test_constructor_bad_shape_shape_raises_exception(self):

        shape = np.ones((2,), dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', Cuboid,
                               shape=shape)

    def test_constructor_bad_shape_array_type_raises_exception(self):

        shape = np.ones((3,), dtype=np.int32)

        self.assertRaisesRegex(TypeError, '.*', Cuboid,
                               shape=shape)

    def test_constructor_bad_shape_raises_exception(self):

        shape = np.array([-1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', Cuboid,
                               shape=shape)
