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
from optid.core import affine

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class AffineTest(unittest.TestCase):
    """
    Test affine arithmetic functions.
    """

    def test_transform_points(self):
        """
        Test transforming a lattice of points.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.eye(3), affine.scale(2, 2, 2)),
            (jnp.eye(3) * 2),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.eye(3), affine.jnp_scale(2, 2, 2)),
            (jnp.eye(3) * 2),
            atol=1e-5))

    def test_transform_vectors(self):
        """
        Test transforming a lattice of vectors.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.eye(3), affine.scale(2, 2, 2)),
            (jnp.eye(3) / 2),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.eye(3), affine.jnp_scale(2, 2, 2)),
            (jnp.eye(3) / 2),
            atol=1e-5))

    def test_transform_rescaled_vectors(self):
        """
        Test transforming a lattice of vectors.
        """

        self.assertTrue(np.allclose(
            affine.transform_rescaled_vectors(jnp.eye(3), affine.scale(2, 2, 2)),
            jnp.eye(3),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_rescaled_vectors(jnp.eye(3), affine.jnp_scale(2, 2, 2)),
            jnp.eye(3),
            atol=1e-5))

    def test_transform_points_translate(self):
        """
        Test translating a point moves it correctly.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.array([1, 2, 3]), affine.translate(2, 3, 4)),
            jnp.array([3, 5, 7]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.array([1, 2, 3]), affine.jnp_translate(2, 3, 4)),
            jnp.array([3, 5, 7]),
            atol=1e-5))

    def test_transform_vectors_translate(self):
        """
        Test translating a vector does nothing.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.array([1, 2, 3]), affine.translate(2, 3, 4)),
            jnp.array([1, 2, 3]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.array([1, 2, 3]), affine.jnp_translate(2, 3, 4)),
            jnp.array([1, 2, 3]),
            atol=1e-5))

    def test_transform_points_scale(self):
        """
        Test scaling a vector.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.array([1, 2, 3]), affine.scale(2, 3, 4)),
            jnp.array([2, 6, 12]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.array([1, 2, 3]), affine.jnp_scale(2, 3, 4)),
            jnp.array([2, 6, 12]),
            atol=1e-5))

    def test_transform_vectors_scale(self):
        """
        Test scaling a vector.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.array([1, 2, 3]), affine.scale(2, 3, 4)),
            jnp.array([1/2, 2/3, 3/4]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.array([1, 2, 3]), affine.jnp_scale(2, 3, 4)),
            jnp.array([1/2, 2/3, 3/4]),
            atol=1e-5))

    def test_transform_points_rotate_x(self):
        """
        Test rotating a point around x works.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.array([0, 0, 1]), affine.rotate_x(affine.radians(180))),
            jnp.array([0, 0, -1]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.array([0, 0, 1]), affine.jnp_rotate_x(affine.jnp_radians(180))),
            jnp.array([0, 0, -1]),
            atol=1e-5))

    def test_transform_points_rotate_z(self):
        """
        Test rotating a point around z works.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.array([1, 0, 0]), affine.rotate_z(affine.radians(180))),
            jnp.array([-1, 0, 0]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.array([1, 0, 0]), affine.jnp_rotate_z(affine.jnp_radians(180))),
            jnp.array([-1, 0, 0]),
            atol=1e-5))

    def test_transform_points_rotate_s(self):
        """
        Test rotating a point around s works.
        """

        self.assertTrue(np.allclose(
            affine.transform_points(jnp.array([0, 1, 0]), affine.rotate_s(affine.radians(180))),
            jnp.array([0, -1, 0]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_points(jnp.array([0, 1, 0]), affine.jnp_rotate_s(affine.jnp_radians(180))),
            jnp.array([0, -1, 0]),
            atol=1e-5))

    def test_transform_vectors_rotate_x(self):
        """
        Test rotating a vector around x works.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.array([0, 0, 1]), affine.rotate_x(affine.radians(180))),
            jnp.array([0, 0, -1]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.array([0, 0, 1]), affine.jnp_rotate_x(affine.jnp_radians(180))),
            jnp.array([0, 0, -1]),
            atol=1e-5))

    def test_transform_vectors_rotate_z(self):
        """
        Test rotating a vector around z works.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.array([1, 0, 0]), affine.rotate_z(affine.radians(180))),
            jnp.array([-1, 0, 0]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.array([1, 0, 0]), affine.jnp_rotate_z(affine.jnp_radians(180))),
            jnp.array([-1, 0, 0]),
            atol=1e-5))

    def test_transform_vectors_rotate_s(self):
        """
        Test rotating a vector around s works.
        """

        self.assertTrue(np.allclose(
            affine.transform_vectors(jnp.array([0, 1, 0]), affine.rotate_s(affine.radians(180))),
            jnp.array([0, -1, 0]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            affine.jnp_transform_vectors(jnp.array([0, 1, 0]), affine.jnp_rotate_s(affine.jnp_radians(180))),
            jnp.array([0, -1, 0]),
            atol=1e-5))

    def test_is_scale_preserving(self):
        """
        Test detecting if a matrix preserves scale.
        """

        self.assertTrue(affine.is_scale_preserving(affine.translate(1, 1, 1)))

        self.assertTrue(affine.jnp_is_scale_preserving(affine.translate(1, 1, 1)))

        self.assertFalse(affine.is_scale_preserving(affine.scale(2, 1, 1)))

        self.assertFalse(affine.jnp_is_scale_preserving(affine.scale(2, 1, 1)))
