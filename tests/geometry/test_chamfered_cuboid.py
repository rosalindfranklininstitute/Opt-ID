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
from optid.geometry import ChamferedCuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class ChamferedCuboidTest(unittest.TestCase):
    """
    Test ChamferedCuboid class.
    """

    ####################################################################################################################

    def test_constructor_shape_array(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        geometry = ChamferedCuboid(shape=shape)

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

        geometry = ChamferedCuboid(shape=shape)

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

        geometry = ChamferedCuboid(shape=shape)

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

        self.assertRaisesRegex(BeartypeException, '.*', ChamferedCuboid,
                               shape=None)

    def test_constructor_bad_shape_shape_raises_exception(self):

        shape = np.ones((2,), dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_bad_shape_array_type_raises_exception(self):

        shape = np.ones((3,), dtype=np.int32)

        self.assertRaisesRegex(TypeError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_bad_shape_raises_exception(self):

        shape = np.array([-1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_chamfer_array_scalar(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = np.zeros((), dtype=np.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_array_xz(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = np.zeros((2,), dtype=np.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_array_scalar_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = np.zeros((4,), dtype=np.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_array_xz_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = np.zeros((4, 2), dtype=np.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_scalar(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = 0

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_tuple_xz(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = (0, 0)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_tuple_scalar_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = (0, 0, 0, 0)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_tuple_xz_corners(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0, 0), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

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

    def test_constructor_chamfer_bl(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0.2, 0.1), (0, 0), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer, nobisect=True)

        vertices = np.array([
            [-0.30000001192092896, -0.5, -0.5], [-0.5, -0.4000000059604645, -0.5], [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.30000001192092896, -0.5, 0.5],
            [-0.5, -0.4000000059604645, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5], [-0.4333333373069763, -0.13333332538604736, 0.1218147948384285]],
            dtype=np.float32)

        polyhedra = [
            [[2, 6, 10], [6, 10, 7], [10, 7, 2], [7, 2, 6]],
            [[10, 4, 3], [4, 3, 8], [3, 8, 10], [8, 10, 4]],
            [[4, 10, 0], [10, 0, 9], [0, 9, 4], [9, 4, 10]],
            [[4, 10, 3], [10, 3, 0], [3, 0, 4], [0, 4, 10]],
            [[3, 10, 2], [10, 2, 0], [2, 0, 3], [0, 3, 10]],
            [[9, 10, 0], [10, 0, 5], [0, 5, 9], [5, 9, 10]],
            [[3, 10, 8], [10, 8, 7], [8, 7, 3], [7, 3, 10]],
            [[4, 10, 9], [10, 9, 8], [9, 8, 4], [8, 4, 10]],
            [[10, 8, 7], [8, 7, 5], [7, 5, 10], [5, 10, 8]],
            [[7, 5, 6], [5, 6, 10], [6, 10, 7], [10, 7, 5]],
            [[5, 1, 6], [1, 6, 10], [6, 10, 5], [10, 5, 1]],
            [[9, 10, 5], [10, 5, 8], [5, 8, 9], [8, 9, 10]],
            [[1, 5, 0], [5, 0, 10], [0, 10, 1], [10, 1, 5]],
            [[10, 3, 2], [3, 2, 7], [2, 7, 10], [7, 10, 3]],
            [[2, 6, 1], [6, 1, 10], [1, 10, 2], [10, 2, 6]],
            [[1, 0, 2], [0, 2, 10], [2, 10, 1], [10, 1, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_chamfer_tl(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0.2, 0.1), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.4000000059604645, -0.5], [-0.30000001192092896, 0.5, -0.5],
            [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.4000000059604645, 0.5], [-0.30000001192092896, 0.5, 0.5], [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5], [-0.4333333373069763, 0.13333334028720856, -0.02029827982187271]],
            dtype=np.float32)

        polyhedra = [
            [[10, 8, 9], [8, 9, 4], [9, 4, 10], [4, 10, 8]],
            [[9, 10, 5], [10, 5, 7], [5, 7, 9], [7, 9, 10]],
            [[2, 6, 10], [6, 10, 7], [10, 7, 2], [7, 2, 6]],
            [[3, 10, 7], [10, 7, 2], [7, 2, 3], [2, 3, 10]],
            [[9, 10, 4], [10, 4, 0], [4, 0, 9], [0, 9, 10]],
            [[8, 10, 3], [10, 3, 4], [3, 4, 8], [4, 8, 10]],
            [[5, 1, 6], [1, 6, 10], [6, 10, 5], [10, 5, 1]],
            [[6, 7, 5], [7, 5, 10], [5, 10, 6], [10, 6, 7]],
            [[8, 10, 7], [10, 7, 3], [7, 3, 8], [3, 8, 10]],
            [[10, 4, 0], [4, 0, 2], [0, 2, 10], [2, 10, 4]],
            [[8, 10, 9], [10, 9, 7], [9, 7, 8], [7, 8, 10]],
            [[10, 9, 5], [9, 5, 0], [5, 0, 10], [0, 10, 9]],
            [[1, 5, 0], [5, 0, 10], [0, 10, 1], [10, 1, 5]],
            [[3, 10, 2], [10, 2, 4], [2, 4, 3], [4, 3, 10]],
            [[2, 6, 1], [6, 1, 10], [1, 10, 2], [10, 2, 6]],
            [[1, 0, 2], [0, 2, 10], [2, 10, 1], [10, 1, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_chamfer_tr(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0, 0), (0.2, 0.1), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.30000001192092896, 0.5, -0.5],
            [0.5, 0.4000000059604645, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5], [0.30000001192092896, 0.5, 0.5], [0.5, 0.4000000059604645, 0.5],
            [0.5, -0.5, 0.5], [0.4333333373069763, 0.13333332538604736, 0.12245573103427887]],
            dtype=np.float32)

        polyhedra = [
            [[6, 10, 7], [10, 7, 5], [7, 5, 6], [5, 6, 10]],
            [[10, 5, 9], [5, 9, 7], [9, 7, 10], [7, 10, 5]],
            [[7, 3, 8], [3, 8, 10], [8, 10, 7], [10, 7, 3]],
            [[8, 9, 7], [9, 7, 10], [7, 10, 8], [10, 8, 9]],
            [[1, 10, 6], [10, 6, 5], [6, 5, 1], [5, 1, 10]],
            [[6, 10, 2], [10, 2, 7], [2, 7, 6], [7, 6, 10]],
            [[4, 8, 10], [8, 10, 9], [10, 9, 4], [9, 4, 8]],
            [[1, 10, 0], [10, 0, 2], [0, 2, 1], [2, 1, 10]],
            [[0, 10, 4], [10, 4, 2], [4, 2, 0], [2, 0, 10]],
            [[1, 10, 2], [10, 2, 6], [2, 6, 1], [6, 1, 10]],
            [[0, 10, 5], [10, 5, 9], [5, 9, 0], [9, 0, 10]],
            [[10, 1, 0], [1, 0, 5], [0, 5, 10], [5, 10, 1]],
            [[3, 7, 2], [7, 2, 10], [2, 10, 3], [10, 3, 7]],
            [[10, 0, 4], [0, 4, 9], [4, 9, 10], [9, 10, 0]],
            [[3, 2, 4], [2, 4, 10], [4, 10, 3], [10, 3, 2]],
            [[4, 8, 3], [8, 3, 10], [3, 10, 4], [10, 4, 8]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_chamfer_br(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0, 0), (0, 0), (0.2, 0.1))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer, nobisect=True)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
            [0.5, -0.4000000059604645, -0.5], [0.30000001192092896, -0.5, -0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.4000000059604645, 0.5],
            [0.30000001192092896, -0.5, 0.5], [0.4333333373069763, -0.13333332538604736, -0.11244573444128036]],
            dtype=np.float32)

        polyhedra = [
            [[5, 10, 6], [10, 6, 9], [6, 9, 5], [9, 5, 10]],
            [[10, 6, 7], [6, 7, 2], [7, 2, 10], [2, 10, 6]],
            [[6, 10, 1], [10, 1, 2], [1, 2, 6], [2, 6, 10]],
            [[5, 10, 0], [10, 0, 1], [0, 1, 5], [1, 5, 10]],
            [[0, 10, 9], [10, 9, 4], [9, 4, 0], [4, 0, 10]],
            [[6, 10, 7], [10, 7, 9], [7, 9, 6], [9, 6, 10]],
            [[5, 10, 9], [10, 9, 0], [9, 0, 5], [0, 5, 10]],
            [[0, 10, 4], [10, 4, 1], [4, 1, 0], [1, 0, 10]],
            [[8, 4, 9], [4, 9, 10], [9, 10, 8], [10, 8, 4]],
            [[8, 9, 7], [9, 7, 10], [7, 10, 8], [10, 8, 9]],
            [[10, 5, 6], [5, 6, 1], [6, 1, 10], [1, 10, 5]],
            [[1, 10, 4], [10, 4, 2], [4, 2, 1], [2, 1, 10]],
            [[3, 7, 10], [7, 10, 8], [10, 8, 3], [8, 3, 7]],
            [[3, 7, 2], [7, 2, 10], [2, 10, 3], [10, 3, 7]],
            [[3, 2, 4], [2, 4, 10], [4, 10, 3], [10, 3, 2]],
            [[4, 8, 3], [8, 3, 10], [3, 10, 4], [10, 4, 8]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_chamfer_type_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(BeartypeException, '.*', ChamferedCuboid,
                               shape=shape, chamfer=None)

    def test_constructor_bad_chamfer_negative_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=-0.1)

    def test_constructor_bad_chamfer_shape_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = np.zeros((4, 3), dtype=np.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_imbalanced_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = (0.1, 0)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_left_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0.6, 0.6), (0.6, 0.6), (0, 0), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_top_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0.6, 0.6), (0.6, 0.6), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_right_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0, 0), (0, 0), (0.6, 0.6), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_bottom_collision_raises_exception(self):

        shape = np.array([1, 1, 1], dtype=np.float32)

        chamfer = ((0.6, 0.6), (0, 0), (0, 0), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)
