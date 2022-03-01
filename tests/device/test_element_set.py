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
import os
import unittest
import numpy as np
import tempfile
import pandas as pd
from more_itertools import SequenceView

# Test imports
import optid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class ElementSetTest(unittest.TestCase):
    """
    Test device ElementSet class.
    """

    def test_constructor(self):

        geometry = optid.geometry.Cuboid(shape=(1, 1, 1))
        element_set = optid.device.ElementSet(name='HH', geometry=geometry,
                                              vector=(0, 0, 1), candidates=None, flip_matrices=None, rescale_vector=False)

        self.assertEqual(element_set.name, 'HH')
        self.assertIs(element_set.geometry, geometry)

    def test_constructor_raises_exception(self):

        self.assertRaisesRegex(ValueError, '.*', optid.device.ElementSet,
                               name='', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                               vector=(0, 0, 1), candidates=None, flip_matrices=None, rescale_vector=False)

        self.assertRaisesRegex(TypeError, '.*', optid.device.ElementSet,
                               name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                               vector=np.array([0, 0, 1], dtype=np.int32),
                               candidates=None, flip_matrices=None, rescale_vector=False)

        self.assertRaisesRegex(ValueError, '.*', optid.device.ElementSet,
                               name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                               vector=(0, 0, 1, 0), candidates=None, flip_matrices=None, rescale_vector=False)

        self.assertRaisesRegex(ValueError, '.*', optid.device.ElementSet,
                               name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                               vector=(0, 0, 1), candidates=None, flip_matrices=[optid.core.affine.scale(2, 1, 1)],
                               rescale_vector=False)

        self.assertRaisesRegex(ValueError, '.*', optid.device.ElementSet,
                               name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                               vector=(0, 0, 1), candidates=None, flip_matrices=[np.eye(3, dtype=np.float32),
                                                                                 np.eye(3, dtype=np.float32)],
                               rescale_vector=False)

    def test_constructor_candidates(self):

        candidates = [optid.device.ElementCandidate(name='1', vector=(0, 0, 0)),
                      optid.device.ElementCandidate(name='2', vector=(0, 0, 1))]

        element_set = optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                                  vector=(0, 0, 1), candidates=SequenceView(candidates), flip_matrices=None,
                                                  rescale_vector=False)

        self.assertEqual(element_set.ncandidate, len(candidates))

    def test_constructor_rescale_vector(self):

        candidates = [optid.device.ElementCandidate(name='1', vector=(0, 0, 0)),
                      optid.device.ElementCandidate(name='2', vector=(0, 0, 1))]

        element_set = optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                                  vector=(0, 0, 1), candidates=candidates, flip_matrices=None,
                                                  rescale_vector=True)

        self.assertEqual(element_set.ncandidate, len(candidates))
        self.assertTrue(np.allclose(element_set.vector, [0, 0, 0.5], atol=1e-5))

    def test_constructor_candidates_csv(self):

        with tempfile.TemporaryDirectory() as path:

            csv_path = os.path.join(path, 'candidates.csv')

            df_candidates = pd.DataFrame([
                dict(name='1', x=0, z=0, s=0),
                dict(name='2', x=0, z=0, s=1)
            ])

            df_candidates.to_csv(csv_path, index=False)

            element_set = optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                                  vector=(0, 0, 1), candidates=csv_path, flip_matrices=None,
                                                  rescale_vector=False)

            self.assertEqual(element_set.ncandidate, len(df_candidates))

    def test_flip_matrix(self):

        element_set = optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                              vector=(0, 0, 1), candidates=None, flip_matrices=None, rescale_vector=False)

        self.assertTrue(np.allclose(element_set.flip_matrix(0),
                                    np.eye(4, dtype=np.float32), atol=1e-5))

    def test_flip_matrix_raises_exception(self):

        element_set = optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                              vector=(0, 0, 1), candidates=None, flip_matrices=None,
                                              rescale_vector=False)

        self.assertRaisesRegex(ValueError, '.*', element_set.flip_matrix, flip=-1)

        self.assertRaisesRegex(ValueError, '.*', element_set.flip_matrix, flip=1)

    def test_is_magnetized(self):

        self.assertTrue(optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                                vector=(0, 0, 1), candidates=None, flip_matrices=None,
                                                rescale_vector=False).is_magnetized)

        self.assertFalse(optid.device.ElementSet(name='HH', geometry=optid.geometry.Cuboid(shape=(1, 1, 1)),
                                                vector=(0, 0, 0), candidates=None, flip_matrices=None,
                                                rescale_vector=False).is_magnetized)
