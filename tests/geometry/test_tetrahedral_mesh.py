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
from optid.geometry import TetrahedralMesh

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class TetrahedralMeshTest(unittest.TestCase):
    """
    Test TetrahedralMesh class.
    """

    ####################################################################################################################

    def test_constructor_vertices_array(self):

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        faces = [
            [1, 3, 2], [0, 3, 1], [5, 6, 7], [4, 5, 7], [1, 0, 4], [1, 4, 5],
            [2, 1, 5], [2, 5, 6], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4]]

        geometry = TetrahedralMesh(vertices=vertices, faces=faces)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

        polyhedra = [
            [[0, 1, 6], [1, 6, 2], [6, 2, 0], [2, 0, 1]],
            [[7, 0, 6], [0, 6, 2], [6, 2, 7], [2, 7, 0]],
            [[0, 1, 5], [1, 5, 6], [5, 6, 0], [6, 0, 1]],
            [[0, 5, 4], [5, 4, 6], [4, 6, 0], [6, 0, 5]],
            [[7, 0, 4], [0, 4, 6], [4, 6, 7], [6, 7, 0]],
            [[7, 0, 2], [0, 2, 3], [2, 3, 7], [3, 7, 0]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_vertices_list(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]

        faces = [
            [1, 3, 2], [0, 3, 1], [5, 6, 7], [4, 5, 7], [1, 0, 4], [1, 4, 5],
            [2, 1, 5], [2, 5, 6], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4]]

        geometry = TetrahedralMesh(vertices=vertices, faces=faces)

        vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]], dtype=np.float32)

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
    def test_constructor_bad_vertices_type_raises_exception(self):

        faces = [
            [1, 3, 2], [0, 3, 1], [5, 6, 7], [4, 5, 7], [1, 0, 4], [1, 4, 5],
            [2, 1, 5], [2, 5, 6], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4]]

        self.assertRaisesRegex(BeartypeException, '.*', TetrahedralMesh,
                               vertices=None, faces=faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_faces_type_raises_exception(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]

        self.assertRaisesRegex(BeartypeException, '.*', TetrahedralMesh,
                               vertices=vertices, faces=None)

    def test_constructor_bad_vertices_list_vertex_shape_raises_exception(self):

        vertices = [
            [-0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]

        faces = [
            [1, 3, 2], [0, 3, 1], [5, 6, 7], [4, 5, 7], [1, 0, 4], [1, 4, 5],
            [2, 1, 5], [2, 5, 6], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4]]

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_vertices_shape_raises_exception(self):

        vertices = np.ones((3, 3), dtype=np.float32)
        faces = np.ones((4, 3), dtype=np.int32)

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_vertices_array_vertices_shape_raises_exception(self):

        vertices = np.ones((4, 4), dtype=np.float32)
        faces = np.ones((4, 3), dtype=np.int32)

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_vertices_array_type_raises_exception(self):

        vertices = np.ones((4, 3), dtype=np.int32)
        faces = np.ones((4, 3), dtype=np.int32)

        self.assertRaisesRegex(TypeError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_list_face_shape_raises_exception(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]

        faces = [
            [1, 3], [0, 3, 1], [5, 6, 7], [4, 5, 7], [1, 0, 4], [1, 4, 5],
            [2, 1, 5], [2, 5, 6], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4]]

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_shape_raises_exception(self):

        vertices = np.ones((4, 3), dtype=np.float32)
        faces = np.ones((3, 3), dtype=np.int32)

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_array_face_shape_raises_exception(self):

        vertices = np.ones((4, 3), dtype=np.float32)
        faces = np.ones((4, 4), dtype=np.int32)

        self.assertRaisesRegex(ValueError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_array_type_raises_exception(self):

        vertices = np.ones((4, 3), dtype=np.float32)
        faces = np.ones((4, 3), dtype=np.float32)

        self.assertRaisesRegex(TypeError, '.*', TetrahedralMesh,
                               vertices=vertices, faces=faces)
