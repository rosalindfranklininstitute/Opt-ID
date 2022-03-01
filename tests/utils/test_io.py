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
import sys
import unittest
from io import BytesIO
from pickle import PicklingError, UnpicklingError
from beartype.roar import BeartypeException
import numpy as np
import jax
import jax.numpy as jnp

# Test imports
import optid
from optid.utils.io import JAXPickler, JAXUnpickler

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class IOTest(unittest.TestCase):
    """
    Test JAX aware Pickler and Unpickler classes.
    """

    ####################################################################################################################

    def test_pickler_constructor(self):
        """
        Test constructing a JAX aware Pickler instance.
        """

        with BytesIO() as fp:
            pickler = JAXPickler(fp)

            self.assertTrue(all(str(device) in pickler.devices for device in jax.local_devices()))
            self.assertIsNone(pickler.device_map)
            self.assertTrue(pickler.raise_on_missing_device)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_pickler_constructor_bad_device_map_type_raises_exception(self):
        """
        Test that an incorrectly typed device map raises an exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(BeartypeException, '.*', JAXPickler,
                                   file=fp, device_map='BAD_DEVICE_MAP')

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_pickler_constructor_bad_raise_on_missing_device_type_raises_exception(self):
        """
        Test that an incorrectly typed boolean flag raises an exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(BeartypeException, '.*', JAXPickler,
                                   file=fp, raise_on_missing_device=None)

    def test_pickler_constructor_raises_on_bad_device_map(self):
        """
        Test that unknown local device name raises exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(PicklingError, '.*', JAXPickler,
                                   file=fp, device_map={ 'BAD_DEVICE_NAME': '???' })

    def test_pickler_persistent_id(self):
        """
        Test that attempting to persistent id an object which is not a JAX DeviceArray or ShardedDeviceArray
        returns None.
        """

        with BytesIO() as fp:
            pickler = JAXPickler(fp)

            self.assertIsNone(pickler.persistent_id('NOT_A_JAX_TENSOR'))

    ####################################################################################################################

    def test_unpickler_constructor(self):
        """
        Test constructing a JAX aware Pickler instance.
        """

        with BytesIO() as fp:
            unpickler = JAXUnpickler(fp)

            self.assertTrue(all(str(device) in unpickler.devices for device in jax.local_devices()))
            self.assertIsNone(unpickler.device_map)
            self.assertTrue(unpickler.raise_on_missing_device)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_unpickler_constructor_bad_device_map_type_raises_exception(self):
        """
        Test that an incorrectly typed device map raises an exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(BeartypeException, '.*', JAXUnpickler,
                                   file=fp, device_map='BAD_DEVICE_MAP')

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_unpickler_constructor_bad_raise_on_missing_device_type_raises_exception(self):
        """
        Test that an incorrectly typed boolean flag raises an exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(BeartypeException, '.*', JAXUnpickler,
                                   file=fp, raise_on_missing_device=None)

    def test_unpickler_constructor_raises_on_bad_device_map(self):
        """
        Test that unknown local device name raises exception on construction.
        """

        with BytesIO() as fp:
            self.assertRaisesRegex(UnpicklingError, '.*', JAXUnpickler,
                                   file=fp, device_map={ '???': 'BAD_DEVICE_NAME' })

    def test_unpickler_persistent_load(self):
        """
        Test that attempting to persistent load an object which is not a JAX DeviceArray or ShardedDeviceArray
        raises an exception.
        """

        with BytesIO() as fp:
            unpickler = JAXUnpickler(fp)

            self.assertRaisesRegex(UnpicklingError, '.*', unpickler.persistent_load,
                                   'NOT_AN_ENCODED_JAX_TENSOR')

            self.assertRaisesRegex(UnpicklingError, '.*', unpickler.persistent_load,
                                   ('NOT_AN_ENCODED_JAX_TENSOR', None))

    ####################################################################################################################

    def test_pickle_unpickle_non_jax_class(self):
        """
        Test that regular classes can be pickled and unpickled correctly.
        """

        exp_data = ['Hello', 'World']

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp)

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp)

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(exp_data, obs_data)

    ####################################################################################################################

    def test_pickle_unpickle_jax_device_array(self):
        """
        Test that JAX DeviceArray's can be pickled and unpickled correctly.
        """

        exp_data = [
            jnp.ones((1, 2, 3), dtype=jnp.float32),
            jnp.ones((4, 5, 6), dtype=jnp.int8)
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp)

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp)

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual(str(exp.device_buffer.device()), str(obs.device_buffer.device()))
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_pickle_unpickle_jax_device_array_with_empty_device_map(self):
        """
        Test that JAX DeviceArray's can be pickled and unpickled correctly.
        """

        exp_data = [
            jnp.ones((1, 2, 3), dtype=jnp.float32),
            jnp.ones((4, 5, 6), dtype=jnp.int8)
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map=dict())

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp, device_map=dict())

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual(str(exp.device_buffer.device()), str(obs.device_buffer.device()))
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_pickle_unpickle_jax_device_array_with_device_map(self):
        """
        Test that JAX DeviceArray's can be pickled and unpickled correctly with a valid device map.
        """

        device = jax.local_devices()[0]

        exp_data = [
            jax.device_put(jnp.ones((1, 2, 3), dtype=jnp.float32), device),
            jax.device_put(jnp.ones((4, 5, 6), dtype=jnp.int8), device)
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map={ str(device): 'MY_DEVICE_NAME' })

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp, device_map={ 'MY_DEVICE_NAME': str(device) })

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual(str(exp.device_buffer.device()), str(obs.device_buffer.device()))
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_unpickle_jax_device_array_with_bad_device_map_raises_exception(self):
        """
        Test that JAX DeviceArray's raise exception when unpickled with a bad device map.
        """

        device = jax.local_devices()[0]

        exp_data = [
            jax.device_put(jnp.ones((1, 2, 3), dtype=jnp.float32), device),
            jax.device_put(jnp.ones((4, 5, 6), dtype=jnp.int8), device)
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map={ str(device): 'MY_DEVICE_NAME' })

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp)

            # Read the observed data from the byte buffer
            self.assertRaisesRegex(UnpicklingError, '.*', unpickler.load)

    ####################################################################################################################

    def test_pickle_unpickle_jax_sharded_device_array(self):
        """
        Test that JAX ShardedDeviceArray's can be pickled and unpickled correctly.
        """

        exp_data = [
            jax.device_put_replicated(jnp.ones((1, 2, 3), dtype=jnp.float32), jax.local_devices()),
            jax.device_put_replicated(jnp.ones((4, 5, 6), dtype=jnp.int8), jax.local_devices())
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp)

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp)

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual([str(buffer.device()) for buffer in exp.device_buffers],
                             [str(buffer.device()) for buffer in obs.device_buffers])
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_pickle_unpickle_jax_sharded_device_array_with_empty_device_map(self):
        """
        Test that JAX ShardedDeviceArray's can be pickled and unpickled correctly.
        """

        exp_data = [
            jax.device_put_replicated(jnp.ones((1, 2, 3), dtype=jnp.float32), jax.local_devices()),
            jax.device_put_replicated(jnp.ones((4, 5, 6), dtype=jnp.int8), jax.local_devices())
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map=dict())

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp, device_map=dict())

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual([str(buffer.device()) for buffer in exp.device_buffers],
                             [str(buffer.device()) for buffer in obs.device_buffers])
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_pickle_unpickle_jax_sharded_device_array_with_device_map(self):
        """
        Test that JAX ShardedDeviceArray's can be pickled and unpickled correctly with a valid device map.
        """

        exp_data = [
            jax.device_put_replicated(jnp.ones((1, 2, 3), dtype=jnp.float32), jax.local_devices()),
            jax.device_put_replicated(jnp.ones((4, 5, 6), dtype=jnp.int8), jax.local_devices())
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map={ str(jax.local_devices()[0]): 'MY_DEVICE_NAME' })

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp, device_map={ 'MY_DEVICE_NAME': str(jax.local_devices()[0]) })

            # Read the observed data from the byte buffer
            obs_data = unpickler.load()

        self.assertEqual(len(exp_data), len(obs_data))

        for exp, obs in zip(exp_data, obs_data):
            self.assertEqual(exp.shape, obs.shape)
            self.assertEqual([str(buffer.device()) for buffer in exp.device_buffers],
                             [str(buffer.device()) for buffer in obs.device_buffers])
            self.assertTrue(np.allclose(exp, obs, atol=1e-5))

    def test_unpickle_jax_sharded_device_array_with_bad_device_map_raises_exception(self):
        """
        Test that JAX ShardedDeviceArray's raise exception when unpickled with a bad device map.
        """

        exp_data = [
            jax.device_put_replicated(jnp.ones((1, 2, 3), dtype=jnp.float32), jax.local_devices()),
            jax.device_put_replicated(jnp.ones((4, 5, 6), dtype=jnp.int8), jax.local_devices())
        ]

        with BytesIO() as fp:

            # Construct a JAX aware pickler instance
            pickler = JAXPickler(fp, device_map={ str(jax.local_devices()[0]): 'MY_DEVICE_NAME' })

            # Dump the expected data to the byte buffer
            pickler.dump(exp_data)

            # Rewind to the beginning of the byte buffer
            fp.seek(0)

            # Construct a JAX aware unpickler instance
            unpickler = JAXUnpickler(fp)

            # Read the observed data from the byte buffer
            self.assertRaisesRegex(UnpicklingError, '.*', unpickler.load)