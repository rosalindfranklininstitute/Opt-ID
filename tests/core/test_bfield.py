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
import radia as rad

# Test imports
import optid
from optid.core import bfield, affine
from optid.lattice import Lattice
from optid.geometry import Cuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class BfieldTest(unittest.TestCase):
    """
    Test bfield arithmetic functions.
    """

    def test_radia_evaluate_bfield_on_lattice(self):

        geometry = Cuboid(shape=(1, 1, 1))

        lattice = Lattice(affine.scale(2, 2, 2), shape=(2, 2, 2))

        with bfield.RadiaCleanEnv():
            field = bfield.radia_evaluate_bfield_on_lattice(geometry.to_radia((0, 1, 0)), lattice.world_lattice)

        self.assertEqual(field.shape, lattice.shape + (3,))

        self.assertTrue(np.allclose(field,
            [[[[ 0.01561037264764309,  -3.8137007440930404e-11,  0.01561037264764309],
               [ 0.01561037264764309,  -1.2789039792460155e-11, -0.01561037264764309]],
              [[-0.01561037264764309,   8.665831247034461e-11,  -0.01561037264764309],
               [-0.01561037264764309,  -1.290060593073239e-11,   0.01561037264764309]]],
             [[[-0.01561037264764309,   3.647449963589677e-12,   0.01561037264764309],
               [-0.015610371716320515, -1.3042447677413804e-10, -0.01561037264764309]],
              [[ 0.01561037264764309,  -1.6599992214150205e-11, -0.01561037264764309],
               [ 0.015610371716320515, -8.610688551069501e-11,   0.01561037264764309]]]], atol=1e-5))

    def test_jnp_radia_evaluate_bfield_on_lattice(self):

        geometry = Cuboid(shape=(1, 1, 1))

        lattice = Lattice(affine.scale(2, 2, 2), shape=(2, 2, 2))

        with bfield.RadiaCleanEnv():
            field = bfield.jnp_radia_evaluate_bfield_on_lattice(geometry.to_radia((0, 1, 0)), lattice.world_lattice)


        self.assertEqual(field.shape, lattice.shape + (3,))

        self.assertTrue(np.allclose(field,
            [[[[ 0.01561037264764309,  -3.8137007440930404e-11,  0.01561037264764309],
               [ 0.01561037264764309,  -1.2789039792460155e-11, -0.01561037264764309]],
              [[-0.01561037264764309,   8.665831247034461e-11,  -0.01561037264764309],
               [-0.01561037264764309,  -1.290060593073239e-11,   0.01561037264764309]]],
             [[[-0.01561037264764309,   3.647449963589677e-12,   0.01561037264764309],
               [-0.015610371716320515, -1.3042447677413804e-10, -0.01561037264764309]],
              [[ 0.01561037264764309,  -1.6599992214150205e-11, -0.01561037264764309],
               [ 0.015610371716320515, -8.610688551069501e-11,   0.01561037264764309]]]], atol=1e-5))

    def test_radia_evaluate_bfield_on_lattice_raises_bad_lattice_shape(self):

        geometry = Cuboid(shape=(1, 1, 1))

        world_lattice = np.zeros((2,), dtype=np.float32)

        rad.UtiDelAll()
        self.assertRaisesRegex(ValueError, '.*', bfield.radia_evaluate_bfield_on_lattice,
                               radia_object=geometry.to_radia((0, 1, 0)), lattice=world_lattice)
        rad.UtiDelAll()

    def test_radia_evaluate_bfield_on_lattice_raises_bad_lattice_type(self):

        geometry = Cuboid(shape=(1, 1, 1))

        world_lattice = np.zeros((3,), dtype=np.int32)

        rad.UtiDelAll()
        self.assertRaisesRegex(TypeError, '.*', bfield.radia_evaluate_bfield_on_lattice,
                               radia_object=geometry.to_radia((0, 1, 0)), lattice=world_lattice)
        rad.UtiDelAll()

    def test_bfield_from_lookup(self):
        """
        Test that a lattice of 3x3 matrices can be applied to a common field vector.
        """

        lookup = jnp.array([
            [[affine.rotate_x(affine.radians(90))[:3, :3], affine.rotate_x(affine.radians(-90))[:3, :3]],
             [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]],
            [[affine.rotate_s(affine.radians(90))[:3, :3], affine.rotate_s(affine.radians(-90))[:3, :3]],
             [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]]
        ])

        self.assertTrue(np.allclose(bfield.jnp_bfield_from_lookup(lookup, jnp.array([1, 0, 0])),
                                    jnp.array([[[[ 1,  0,  0], [ 1,  0,  0]],
                                                [[ 0,  0,  1], [ 0,  0, -1]]],
                                               [[[ 0, -1,  0], [ 0,  1,  0]],
                                                [[ 0,  0,  1], [ 0,  0, -1]]]]), atol=1e-5))

    def test_integrate_trajectories(self):

        geometry = Cuboid(shape=(1, 1, 1)).transform(affine.translate(0, -2, 0))

        lattice = Lattice(affine.scale(2, 2, 10), shape=(3, 3, 10))

        rad.UtiDelAll()
        field = bfield.radia_evaluate_bfield_on_lattice(geometry.to_radia((0, 1, 0)), lattice.world_lattice)
        rad.UtiDelAll()

        self.assertEqual(field.shape, lattice.shape + (3,))

        traj_1st, traj_2nd = bfield.jnp_integrate_trajectories(field, lattice.step[2], 3.0)

        self.assertEqual(traj_1st.shape, lattice.shape + (2,))
        self.assertEqual(traj_2nd.shape, lattice.shape + (2,))
