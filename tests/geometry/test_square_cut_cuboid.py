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
from optid.geometry import SquareCutCuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class SquareCutCuboidTest(unittest.TestCase):
    """
    Test SquareCutCuboid class.
    """

    ####################################################################################################################

    def test_constructor_shape_array(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        geometry = SquareCutCuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_shape_list(self):

        shape = [1, 1, 1]

        geometry = SquareCutCuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_shape_tuple(self):

        shape = (1, 1, 1)

        geometry = SquareCutCuboid(shape=shape)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shape_type_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', SquareCutCuboid,
                               shape=None)

    def test_constructor_bad_shape_shape_raises_exception(self):

        shape = np.ones((2,), dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape)

    def test_constructor_bad_shape_array_type_raises_exception(self):

        shape = np.ones((3,), dtype=np.int32)

        self.assertRaisesRegex(TypeError, '.*', SquareCutCuboid,
                               shape=shape)

    def test_constructor_bad_shape_raises_exception(self):

        shape = np.array([-1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape)

    def test_constructor_cuts_array_scalar(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = np.zeros((), dtype=np.float32)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_array_xz(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = np.zeros((2,), dtype=np.float32)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_array_scalar_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = np.zeros((4,), dtype=np.float32)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_array_xz_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = np.zeros((4, 2), dtype=np.float32)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_scalar(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = 0

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_tuple_xz(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = (0, 0)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_tuple_scalar_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = (0, 0, 0, 0)

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_tuple_xz_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0, 0), (0, 0), (0, 0))

        geometry = SquareCutCuboid(shape=shape, cuts=cuts)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_bl(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0.2, 0.1), (0, 0), (0, 0), (0, 0))

        geometry = SquareCutCuboid(shape=shape, cuts=cuts, nobisect=True)

        vertices = np.array([
            [-0.30000001192092896, -0.5, -0.5], [-0.30000001192092896, -0.4000000059604645, -0.5],
            [-0.5, -0.4000000059604645, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5], [-0.30000001192092896, -0.5, 0.5], [-0.30000001192092896, -0.4000000059604645, 0.5],
            [-0.5, -0.4000000059604645, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
            [-0.4333333373069763, -0.09999999403953552, 0.018803969025611877]],
            dtype=np.float32)

        polyhedra = [
            [[3, 8, 12], [8, 12, 9], [12, 9, 3], [9, 3, 8]],
            [[1, 6, 11], [6, 11, 7], [11, 7, 1], [7, 1, 6]],
            [[4, 12, 10], [12, 10, 9], [10, 9, 4], [9, 4, 12]],
            [[1, 6, 0], [6, 0, 11], [0, 11, 1], [11, 1, 6]],
            [[5, 12, 1], [12, 1, 11], [1, 11, 5], [11, 5, 12]],
            [[4, 12, 3], [12, 3, 1], [3, 1, 4], [1, 4, 12]],
            [[12, 5, 4], [5, 4, 10], [4, 10, 12], [10, 12, 5]],
            [[11, 12, 1], [12, 1, 7], [1, 7, 11], [7, 11, 12]],
            [[5, 12, 11], [12, 11, 10], [11, 10, 5], [10, 5, 12]],
            [[12, 10, 9], [10, 9, 7], [9, 7, 12], [7, 12, 10]],
            [[7, 2, 8], [2, 8, 12], [8, 12, 7], [12, 7, 2]],
            [[9, 7, 8], [7, 8, 12], [8, 12, 9], [12, 9, 7]],
            [[5, 12, 4], [12, 4, 1], [4, 1, 5], [1, 5, 12]],
            [[0, 1, 11], [1, 11, 5], [11, 5, 0], [5, 0, 1]],
            [[11, 12, 7], [12, 7, 10], [7, 10, 11], [10, 11, 12]],
            [[1, 2, 7], [2, 7, 12], [7, 12, 1], [12, 1, 2]],
            [[3, 8, 2], [8, 2, 12], [2, 12, 3], [12, 3, 8]],
            [[12, 4, 3], [4, 3, 9], [3, 9, 12], [9, 12, 4]],
            [[2, 1, 3], [1, 3, 12], [3, 12, 2], [12, 2, 1]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_tl(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0.2, 0.1), (0, 0), (0, 0))

        geometry = SquareCutCuboid(shape=shape, cuts=cuts, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.4000000059604645, -0.5], [-0.30000001192092896, 0.4000000059604645, -0.5],
            [-0.30000001192092896, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.4000000059604645, 0.5], [-0.30000001192092896, 0.4000000059604645, 0.5],
            [-0.30000001192092896, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
            [0.23333333432674408, 0.13333332538604736, -0.05661701038479805]],
            dtype=np.float32)

        polyhedra = [
            [[1, 12, 2], [12, 2, 7], [2, 7, 1], [7, 1, 12]],
            [[12, 9, 10], [9, 10, 4], [10, 4, 12], [4, 12, 9]],
            [[7, 12, 8], [12, 8, 6], [8, 6, 7], [6, 7, 12]],
            [[12, 6, 11], [6, 11, 8], [11, 8, 12], [8, 12, 6]],
            [[12, 3, 2], [3, 2, 8], [2, 8, 12], [8, 12, 3]],
            [[12, 0, 2], [0, 2, 5], [2, 5, 12], [5, 12, 0]],
            [[1, 12, 0], [12, 0, 2], [0, 2, 1], [2, 1, 12]],
            [[12, 9, 8], [9, 8, 10], [8, 10, 12], [10, 12, 9]],
            [[1, 12, 7], [12, 7, 6], [7, 6, 1], [6, 1, 12]],
            [[12, 1, 0], [1, 0, 6], [0, 6, 12], [6, 12, 1]],
            [[9, 12, 3], [12, 3, 4], [3, 4, 9], [4, 9, 12]],
            [[12, 9, 3], [9, 3, 8], [3, 8, 12], [8, 12, 9]],
            [[11, 8, 10], [8, 10, 12], [10, 12, 11], [12, 11, 8]],
            [[12, 0, 5], [0, 5, 11], [5, 11, 12], [11, 12, 0]],
            [[5, 10, 12], [10, 12, 11], [12, 11, 5], [11, 5, 10]],
            [[12, 6, 0], [6, 0, 11], [0, 11, 12], [11, 12, 6]],
            [[12, 3, 4], [3, 4, 2], [4, 2, 12], [2, 12, 3]],
            [[4, 2, 5], [2, 5, 12], [5, 12, 4], [12, 4, 2]],
            [[5, 10, 4], [10, 4, 12], [4, 12, 5], [12, 5, 10]],
            [[7, 12, 2], [12, 2, 8], [2, 8, 7], [8, 7, 12]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_tr(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0, 0), (0.2, 0.1), (0, 0))

        geometry = SquareCutCuboid(shape=shape, cuts=cuts, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.30000001192092896, 0.5, -0.5],
            [0.30000001192092896, 0.4000000059604645, -0.5], [0.5, 0.4000000059604645, -0.5],
            [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.30000001192092896, 0.5, 0.5],
            [0.30000001192092896, 0.4000000059604645, 0.5], [0.5, 0.4000000059604645, 0.5], [0.5, -0.5, 0.5],
            [0.03333333507180214, 0.46666666865348816, 0.01981320045888424]],
            dtype=np.float32)

        polyhedra = [
            [[5, 12, 3], [12, 3, 0], [3, 0, 5], [0, 5, 12]],
            [[2, 7, 12], [7, 12, 8], [12, 8, 2], [8, 2, 7]],
            [[8, 3, 9], [3, 9, 12], [9, 12, 8], [12, 8, 3]],
            [[6, 12, 0], [12, 0, 1], [0, 1, 6], [1, 6, 12]],
            [[11, 12, 9], [12, 9, 5], [9, 5, 11], [5, 11, 12]],
            [[6, 12, 7], [12, 7, 9], [7, 9, 6], [9, 6, 12]],
            [[5, 10, 9], [10, 9, 11], [9, 11, 5], [11, 5, 10]],
            [[7, 8, 9], [8, 9, 12], [9, 12, 7], [12, 7, 8]],
            [[5, 12, 9], [12, 9, 3], [9, 3, 5], [3, 5, 12]],
            [[12, 6, 7], [6, 7, 1], [7, 1, 12], [1, 12, 6]],
            [[5, 10, 4], [10, 4, 9], [4, 9, 5], [9, 5, 10]],
            [[12, 11, 6], [11, 6, 0], [6, 0, 12], [0, 12, 11]],
            [[11, 12, 5], [12, 5, 0], [5, 0, 11], [0, 11, 12]],
            [[12, 0, 1], [0, 1, 3], [1, 3, 12], [3, 12, 0]],
            [[9, 4, 5], [4, 5, 3], [5, 3, 9], [3, 9, 4]],
            [[11, 12, 6], [12, 6, 9], [6, 9, 11], [9, 11, 12]],
            [[2, 1, 3], [1, 3, 12], [3, 12, 2], [12, 2, 1]],
            [[2, 7, 1], [7, 1, 12], [1, 12, 2], [12, 2, 7]],
            [[3, 8, 2], [8, 2, 12], [2, 12, 3], [12, 3, 8]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_cuts_br(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0, 0), (0, 0), (0.2, 0.1))

        geometry = SquareCutCuboid(shape=shape, cuts=cuts, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.4000000059604645, -0.5],
            [0.30000001192092896, -0.4000000059604645, -0.5], [0.30000001192092896, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.4000000059604645, 0.5],
            [0.30000001192092896, -0.4000000059604645, 0.5], [0.30000001192092896, -0.5, 0.5],
            [0.4333333373069763, -0.10000001639127731, 0.01864202693104744]],
            dtype=np.float32)

        polyhedra = [
            [[4, 9, 3], [9, 3, 12], [3, 12, 4], [12, 4, 9]],
            [[3, 8, 2], [8, 2, 12], [2, 12, 3], [12, 3, 8]],
            [[5, 10, 0], [10, 0, 11], [0, 11, 5], [11, 5, 10]],
            [[3, 8, 12], [8, 12, 9], [12, 9, 3], [9, 3, 8]],
            [[0, 12, 10], [12, 10, 4], [10, 4, 0], [4, 0, 12]],
            [[6, 12, 7], [12, 7, 10], [7, 10, 6], [10, 6, 12]],
            [[6, 12, 10], [12, 10, 0], [10, 0, 6], [0, 6, 12]],
            [[6, 0, 10], [0, 10, 11], [10, 11, 6], [11, 6, 0]],
            [[6, 12, 0], [12, 0, 1], [0, 1, 6], [1, 6, 12]],
            [[9, 10, 8], [10, 8, 12], [8, 12, 9], [12, 9, 10]],
            [[7, 12, 8], [12, 8, 10], [8, 10, 7], [10, 7, 12]],
            [[12, 6, 7], [6, 7, 1], [7, 1, 12], [1, 12, 6]],
            [[10, 9, 4], [9, 4, 12], [4, 12, 10], [12, 10, 9]],
            [[3, 2, 4], [2, 4, 12], [4, 12, 3], [12, 3, 2]],
            [[1, 12, 4], [12, 4, 2], [4, 2, 1], [2, 1, 12]],
            [[12, 7, 8], [7, 8, 2], [8, 2, 12], [2, 12, 7]],
            [[0, 12, 4], [12, 4, 1], [4, 1, 0], [1, 0, 12]],
            [[7, 12, 1], [12, 1, 2], [1, 2, 7], [2, 7, 12]],
            [[5, 10, 4], [10, 4, 0], [4, 0, 5], [0, 5, 10]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_cuts_type_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(BeartypeException, '.*', SquareCutCuboid,
                               shape=shape, cuts=None)

    def test_constructor_bad_cuts_negative_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=-0.1)

    def test_constructor_bad_cuts_shape_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = np.zeros((4, 3), dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)

    def test_constructor_bad_cuts_imbalanced_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = (0.1, 0)

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)

    def test_constructor_bad_cuts_left_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0.6, 0.6), (0.6, 0.6), (0, 0), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)

    def test_constructor_bad_cuts_top_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0.6, 0.6), (0.6, 0.6), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)

    def test_constructor_bad_cuts_right_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0, 0), (0, 0), (0.6, 0.6), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)

    def test_constructor_bad_cuts_bottom_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        cuts = ((0.6, 0.6), (0, 0), (0, 0), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', SquareCutCuboid,
                               shape=shape, cuts=cuts)
