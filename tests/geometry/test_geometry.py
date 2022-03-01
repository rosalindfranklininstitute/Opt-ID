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
from optid.geometry import Geometry
from optid.core.affine import translate, scale
from optid.material import NamedMaterial

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class GeometryTest(unittest.TestCase):
    """
    Test Geometry class.
    """

    ####################################################################################################################

    def test_constructor_vertices_array(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        # self.assertEqual(geometry.polyhedra, polyhedra)
        self.assertTrue(np.allclose(geometry.bounds, ([-0.5, -0.5, -0.5], [0.5,  0.5,  0.5]), atol=1e-5))

    def test_constructor_vertices_list(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertTrue(np.allclose(geometry.vertices, np.array(vertices, dtype=np.float32), atol=1e-5))
        # self.assertEqual(geometry.polyhedra, polyhedra)
        self.assertTrue(np.allclose(geometry.bounds, ([-0.5, -0.5, -0.5], [0.5,  0.5,  0.5]), atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_vertices_raises_exception(self):

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(BeartypeException, '.*', Geometry,
                               vertices=None, polyhedra=polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_polyhedra_raises_exception(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        self.assertRaisesRegex(BeartypeException, '.*', Geometry,
                               vertices=vertices, polyhedra=None)

    def test_constructor_bad_vertices_list_vertex_shape_raises_exception(self):

        vertices = [
            [-0.5, -0.5      ], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_vertices_array_vertices_shape_raises_exception(self):

        vertices = np.ones((8, 2), dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_vertices_array_type_raises_exception(self):

        vertices = np.ones((8, 3), dtype=np.int32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(TypeError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_polyhedra_list_of_list_of_sequences(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]],
            dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], (4, 5, 6, 7),
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, [[[vertex for vertex in face] for face in faces] for faces in polyhedra])

    def test_constructor_polyhedra_list_of_sequences_of_lists(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]],
            dtype=np.float32)

        polyhedra = [(
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7])]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, [[[vertex for vertex in face] for face in faces] for faces in polyhedra])

    def test_constructor_polyhedra_sequence_of_lists_of_lists(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]],
            dtype=np.float32)

        polyhedra = ([
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]],)

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, [[[vertex for vertex in face] for face in faces] for faces in polyhedra])

    def test_constructor_bad_polyhedra_empty_list_exception(self):

        vertices = np.ones((8, 3), dtype=np.float32)

        polyhedra = []

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_polyhedra_vertex_out_of_bounds_raises_exception(self):

        vertices = np.ones((8, 3), dtype=np.float32)

        polyhedra = [[
            [8, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_polyhedra_vertex_duplicated_raises_exception(self):

        vertices = np.ones((8, 3), dtype=np.float32)

        polyhedra = [[
            [0, 0, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_polyhedra_face_not_polygon_raises_exception(self):

        vertices = np.ones((8, 3), dtype=np.float32)

        polyhedra = [[
            [0, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    def test_constructor_bad_polyhedra_not_polyhedra_raises_exception(self):

        vertices = np.ones((8, 3), dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4]]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, polyhedra=polyhedra)

    ####################################################################################################################

    def test_transform(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        transformed_geometry = geometry.transform(translate(2, 2, 2))

        self.assertTrue(np.allclose(transformed_geometry.vertices, (vertices + 2.0), atol=1e-5))
        self.assertEqual(transformed_geometry.polyhedra, geometry.polyhedra)
        self.assertIs(transformed_geometry.polyhedra, geometry.polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_transform_bad_matrix_type_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(BeartypeException, '.*', geometry.transform,
                               matrix=None)

    def test_transform_bad_matrix_shape_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(ValueError, '.*', geometry.transform,
                               matrix=np.eye(3, dtype=np.float32))

    def test_transform_bad_matrix_array_type_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(TypeError, '.*', geometry.transform,
                               matrix=np.eye(4, dtype=np.int32))

    def test_transform_scaled_matrix_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(ValueError, '.*', geometry.transform,
                               matrix=scale(2, 1, 1))

    ####################################################################################################################

    def test_to_radia_array(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        obj = geometry.to_radia(np.ones((3,), dtype=np.float32))

    def test_to_radia_material(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra, material=NamedMaterial(name='Sm2Co17'))

        obj = geometry.to_radia(np.ones((3,), dtype=np.float32))

    def test_to_radia_list(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        obj = geometry.to_radia([0, 1, 0])

    def test_to_radia_tuple(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        obj = geometry.to_radia((0, 1, 0))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_to_radia_bad_vector_type_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(BeartypeException, '.*', geometry.to_radia,
                               vector=None)

    def test_to_radia_bad_vector_shape_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(ValueError, '.*', geometry.to_radia,
                               vector=np.ones((4,), dtype=np.float32))

    def test_to_radia_bad_vector_array_type_raises_exception(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [[
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]]

        geometry = Geometry(vertices=vertices, polyhedra=polyhedra)

        self.assertRaisesRegex(TypeError, '.*', geometry.to_radia,
                                   vector=np.ones((3,), dtype=np.int32))
