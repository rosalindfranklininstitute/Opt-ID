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
from optid.lattice import Lattice
from optid.core.affine import scale
from optid.constants import MATRIX_IDENTITY

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class LatticeTest(unittest.TestCase):
    """
    Test Lattice class.
    """

    ####################################################################################################################

    def test_constructor(self):

        lattice = Lattice(unit_to_world_matrix=MATRIX_IDENTITY, shape=(1, 2, 3))

        self.assertTrue(np.allclose(lattice.step, (0.0, 1.0, 0.5), atol=1e-5))

        lattice = Lattice(unit_to_world_matrix=MATRIX_IDENTITY, shape=(1, 1, 1))

        self.assertTrue(np.allclose(lattice.unit_to_orthonormal_matrix,
                                    np.array([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [0.5, 0.5, 0.5, 1.0]],
                                             dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.orthonormal_to_unit_matrix,
                                    np.array([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [-0.5, -0.5, -0.5, 1.0]],
                                             dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.world_to_unit_matrix,
                                    np.array([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [0.0, 0.0, 0.0, 1.0]],
                                             dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.world_to_orthonormal_matrix,
                                    np.array([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [0.5, 0.5, 0.5, 1.0]],
                                             dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.orthonormal_to_world_matrix,
                                    np.array([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [-0.5, -0.5, -0.5, 1.0]],
                                             dtype=np.float32), atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shape_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(1, 1, 1.5))

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(1,))

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(1, 1))

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(1, 1, 1, 1))

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=None)

    def test_constructor_bad_shape_x_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(0, 2, 10))

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(-1, 2, 10))

    def test_constructor_bad_shape_z_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(2, 0, 10))

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(2, -1, 10))

    def test_constructor_bad_shape_s_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(2, 2, 0))

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=MATRIX_IDENTITY, shape=(2, 2, -1))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_matrix_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', Lattice,
                               unit_to_world_matrix=None, shape=(1, 1, 1))

    def test_constructor_bad_matrix_type_raises_exception(self):

        self.assertRaisesRegex(TypeError, '.*', Lattice,
                               unit_to_world_matrix=np.eye(4, dtype=np.int32), shape=(2, 2, 10))

    def test_constructor_bad_matrix_shape_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', Lattice,
                               unit_to_world_matrix=np.eye(2, dtype=np.float32), shape=(2, 2, 10))

    ####################################################################################################################

    def test_eq(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(lattice0 == lattice1)

    def test_eq_self(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(lattice == lattice)

    def test_eq_bad_type(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertFalse(lattice == 'None')

    def test_eq_bad_shape(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(1, 1, 1))

        self.assertFalse(lattice0 == lattice1)

    def test_eq_bad_matrix(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(1, 2, 3), shape=(2, 2, 2))

        self.assertFalse(lattice0 == lattice1)

    ####################################################################################################################

    def test_ne(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertFalse(lattice0 != lattice1)

    def test_ne_self(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertFalse(lattice != lattice)

    def test_ne_bad_type(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(lattice != 'None')

    def test_ne_bad_shape(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(1, 1, 1))

        self.assertTrue(lattice0 != lattice1)

    def test_ne_bad_matrix(self):

        lattice0 = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))
        lattice1 = Lattice(unit_to_world_matrix=scale(1, 2, 3), shape=(2, 2, 2))

        self.assertTrue(lattice0 != lattice1)

    ####################################################################################################################

    def test_unit_lattice(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(np.allclose(lattice.unit_lattice,
                                    np.array([[[[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5]],
                                                [[-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5]]],
                                               [[[ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5]],
                                                [[ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5]]]],
                                              dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.jnp_unit_lattice,
                                    np.array([[[[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5]],
                                               [[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]],
                                              [[[0.5, -0.5, -0.5], [0.5, -0.5, 0.5]],
                                               [[0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]]],
                                             dtype=np.float32), atol=1e-5))

    ####################################################################################################################

    def test_world_lattice(self):

        lattice = Lattice(unit_to_world_matrix=scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(np.allclose(lattice.world_lattice,
                                    np.array([[[[-1, -1, -1], [-1, -1,  1]],
                                                [[-1,  1, -1], [-1,  1,  1]]],
                                               [[[ 1, -1, -1], [ 1, -1,  1]],
                                                [[ 1,  1, -1], [ 1,  1,  1]]]],
                                              dtype=np.float32), atol=1e-5))

        self.assertTrue(np.allclose(lattice.jnp_world_lattice,
                                    np.array([[[[-1, -1, -1], [-1, -1, 1]],
                                               [[-1, 1, -1], [-1, 1, 1]]],
                                              [[[1, -1, -1], [1, -1, 1]],
                                               [[1, 1, -1], [1, 1, 1]]]],
                                             dtype=np.float32), atol=1e-5))
