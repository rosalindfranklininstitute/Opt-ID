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
import jax.numpy as jnp

# Test imports
import optid
from optid.core import affine, lattice

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class LatticeTest(unittest.TestCase):
    """
    Test lattice arithmetic functions.
    """

    def test_unit_limits(self):
        """
        Test unit axis limits are correct.
        """

        with self.subTest(n=1):
            self.assertEqual(lattice.unit_limits(1), (0, 0))
            self.assertEqual(lattice.jnp_unit_limits(1), (0, 0))

        for n in range(2, 5):
            with self.subTest(n=n):
                self.assertEqual(lattice.unit_limits(n), (-0.5, 0.5))
                self.assertEqual(lattice.jnp_unit_limits(n), (-0.5, 0.5))

    def test_unit_lattice(self):
        """
        Test unit 3-lattice has correct coordinates.
        """

        with self.subTest(x=2, z=2, s=2):
            self.assertTrue(np.allclose(lattice.unit_lattice(2, 2, 2), jnp.array([
                [[[-0.5, -0.5, -0.5], [-0.5, -0.5, +0.5]],
                 [[-0.5, +0.5, -0.5], [-0.5, +0.5, +0.5]]],
                [[[+0.5, -0.5, -0.5], [+0.5, -0.5, +0.5]],
                 [[+0.5, +0.5, -0.5], [+0.5, +0.5, +0.5]]]
            ]), atol=1e-5))

            self.assertTrue(np.allclose(lattice.jnp_unit_lattice(2, 2, 2), jnp.array([
                [[[-0.5, -0.5, -0.5], [-0.5, -0.5, +0.5]],
                 [[-0.5, +0.5, -0.5], [-0.5, +0.5, +0.5]]],
                [[[+0.5, -0.5, -0.5], [+0.5, -0.5, +0.5]],
                 [[+0.5, +0.5, -0.5], [+0.5, +0.5, +0.5]]]
            ]), atol=1e-5))

        with self.subTest(x=1, z=2, s=2):
            self.assertTrue(np.allclose(lattice.unit_lattice(1, 2, 2), jnp.array([
                [[[0, -0.5, -0.5], [0, -0.5, +0.5]],
                 [[0, +0.5, -0.5], [0, +0.5, +0.5]]]
            ]), atol=1e-5))

            self.assertTrue(np.allclose(lattice.jnp_unit_lattice(1, 2, 2), jnp.array([
                [[[0, -0.5, -0.5], [0, -0.5, +0.5]],
                 [[0, +0.5, -0.5], [0, +0.5, +0.5]]]
            ]), atol=1e-5))

        with self.subTest(x=2, z=1, s=2):
            self.assertTrue(np.allclose(lattice.unit_lattice(2, 1, 2), jnp.array([
                [[[-0.5, 0, -0.5], [-0.5, 0, +0.5]]],
                [[[+0.5, 0, -0.5], [+0.5, 0, +0.5]]]
            ]), atol=1e-5))

            self.assertTrue(np.allclose(lattice.jnp_unit_lattice(2, 1, 2), jnp.array([
                [[[-0.5, 0, -0.5], [-0.5, 0, +0.5]]],
                [[[+0.5, 0, -0.5], [+0.5, 0, +0.5]]]
            ]), atol=1e-5))

        with self.subTest(x=2, z=2, s=1):
            self.assertTrue(np.allclose(lattice.unit_lattice(2, 2, 1), jnp.array([
                [[[-0.5, -0.5, 0]],
                 [[-0.5, +0.5, 0]]],
                [[[+0.5, -0.5, 0]],
                 [[+0.5, +0.5, 0]]]
            ]), atol=1e-5))

            self.assertTrue(np.allclose(lattice.jnp_unit_lattice(2, 2, 1), jnp.array([
                [[[-0.5, -0.5, 0]],
                 [[-0.5, +0.5, 0]]],
                [[[+0.5, -0.5, 0]],
                 [[+0.5, +0.5, 0]]]
            ]), atol=1e-5))

    def test_unit_to_orthonormal_matrix(self):
        """
        Test converting from unit to orthonormal coordinates.
        """

        with self.subTest(x=2, z=2, s=2):

            expected = jnp.array([
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                 [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]],
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                 [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]]
            ])

            self.assertTrue(np.allclose(affine.transform_points(lattice.unit_lattice(2, 2, 2),
                                                                lattice.unit_to_orthonormal_matrix(2, 2, 2)),
                                        expected, atol=1e-5))

            self.assertTrue(np.allclose(affine.jnp_transform_points(lattice.jnp_unit_lattice(2, 2, 2),
                                                                    lattice.jnp_unit_to_orthonormal_matrix(2, 2, 2)),
                                        expected, atol=1e-5))

    def test_any_unit_point_out_of_bounds(self):

        self.assertFalse(lattice.any_unit_point_out_of_bounds(np.array([0, 0, 0], dtype=np.float32), 1e-5))
        self.assertFalse(lattice.jnp_any_unit_point_out_of_bounds(np.array([0, 0, 0], dtype=np.float32), 1e-5))

        self.assertTrue(lattice.any_unit_point_out_of_bounds(np.array([-1, 1, 0], dtype=np.float32), 1e-5))
        self.assertTrue(lattice.jnp_any_unit_point_out_of_bounds(np.array([-1, 1, 0], dtype=np.float32), 1e-5))

    def test_any_orthonormal_point_out_of_bounds(self):

        self.assertFalse(lattice.any_orthonormal_point_out_of_bounds(np.array([0, 0, 0], dtype=np.float32), 2, 2, 2, 1e-5))
        self.assertFalse(lattice.jnp_any_orthonormal_point_out_of_bounds(np.array([0, 0, 0], dtype=np.float32), 2, 2, 2, 1e-5))

        self.assertTrue(lattice.any_orthonormal_point_out_of_bounds(np.array([-1, 2, 0], dtype=np.float32), 2, 2, 2, 1e-5))
        self.assertTrue(lattice.jnp_any_orthonormal_point_out_of_bounds(np.array([-1, 2, 0], dtype=np.float32), 2, 2, 2, 1e-5))

    def test_orthonormal_interpolate_1d(self):
        """
        Test interpolating lattices of coordinates into a 1d lattice of arbitrary channels.
        """

        with self.subTest('values (4,), points (3, 1), result (3,)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([0.0, 1.0, 2.0, 3.0]),
                point_lattice=jnp.array([[0.5], [1.5], [2.5]])),
                jnp.array([0.5, 1.5, 2.5]),
                atol=1e-5))

        with self.subTest('values (4,), points (3, 2, 1), result (3, 2)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([0.0, 1.0, 2.0, 3.0]),
                point_lattice=jnp.array([[[0.5], [0.5]],
                                         [[1.5], [0.5]],
                                         [[2.5], [0.5]]])),
                jnp.array([[0.5, 0.5],
                           [1.5, 0.5],
                           [2.5, 0.5]]),
                atol=1e-5))

        with self.subTest('values (4, 2), points (3, 1), result (3, 2)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([[0.0, 1.0],
                                         [1.0, 2.0],
                                         [2.0, 3.0],
                                         [3.0, 4.0]]),
                point_lattice=jnp.array([[0.5], [1.5], [2.5]])),
                jnp.array([[0.5, 1.5],
                           [1.5, 2.5],
                           [2.5, 3.5]]),
                atol=1e-5))

        with self.subTest('values (4, 2), points (3, 2, 1), result (3, 2, 2)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([[0.0, 1.0],
                                         [1.0, 2.0],
                                         [2.0, 3.0],
                                         [3.0, 4.0]]),
                point_lattice=jnp.array([[[0.5], [0.5]],
                                         [[1.5], [0.5]],
                                         [[2.5], [0.5]]])),
                jnp.array([[[0.5, 1.5], [0.5, 1.5]],
                           [[1.5, 2.5], [0.5, 1.5]],
                           [[2.5, 3.5], [0.5, 1.5]]]),
                atol=1e-5))

    def test_orthonormal_interpolate_2d(self):
        """
        Test interpolating lattices of coordinates into a 2d lattice of arbitrary channels.
        """

        with self.subTest('values (4, 2), points (3, 2), result (3,)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([[0.0, 1.0],
                                         [1.0, 2.0],
                                         [2.0, 3.0],
                                         [3.0, 4.0]]),
                point_lattice=jnp.array([[0.5, 0.5],
                                         [1.5, 0.5],
                                         [2.5, 0.5]])),
                jnp.array([1.0, 2.0, 3.0]),
                atol=1e-5))

        with self.subTest('values (4, 2), points (3, 2, 2), result (3, 2)'):
            self.assertTrue(np.allclose(lattice.jnp_orthonormal_interpolate(
                value_lattice=jnp.array([[0.0, 1.0],
                                         [1.0, 2.0],
                                         [2.0, 3.0],
                                         [3.0, 4.0]]),
                point_lattice=jnp.array([[[0.5, 0.5], [0.5, 0.5]],
                                         [[1.5, 0.5], [0.5, 0.5]],
                                         [[2.5, 0.5], [0.5, 0.5]]])),
                jnp.array([[1.0, 1.0],
                           [2.0, 1.0],
                           [3.0, 1.0]]),
                atol=1e-5))
