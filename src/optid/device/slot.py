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


# External Imports
import numbers
from beartype import beartype
import typing as typ
import numpy as np


# Opt-ID Imports
from ..constants import VECTOR_ZERO
from ..core.affine import transform_rescaled_vectors, is_scale_preserving, translate
from ..core.bfield import RadiaCleanEnv, jnp_radia_evaluate_bfield_on_lattice
from ..core.utils import np_readonly
from ..lattice import Lattice
from ..bfield import Bfield
from .element_state import ElementState
from .pose import Pose
from .slot_type import SlotType


class Slot:

    @beartype
    def __init__(self,
            girder,
            index: int,
            name: str,
            period: str,
            slot_type: SlotType,
            slot_matrix: np.ndarray):
        """
        Construct a Slot instance.

        :param girder:
            Parent Girder instance this slot is a member of.

        :param index:
            Integer index for the slot in the girder.

        :param name:
            String name for the slot.

        :param period:
            String period name used for calculating device period length.

        :param slot_type:
            MagnetSlotType instance that this slot is one of.

        :param slot_matrix:
            Affine matrix for the placing this slot along its parent girder starting from 0 on the Z axis.
        """

        if index < 0:
            raise ValueError(f'index must be >= 0 but is : '
                             f'{index}')

        self._index = index

        self._girder = girder

        self._slot_type = slot_type

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if len(period) == 0:
            raise ValueError(f'period must be a non-empty string')

        self._period = period

        if slot_matrix.shape != (4, 4):
            raise ValueError(f'slot_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{slot_matrix.shape}')

        if slot_matrix.dtype != np.float32:
            raise TypeError(f'slot_matrix must have dtype (float32) but is : '
                            f'{slot_matrix.dtype}')

        if not is_scale_preserving(slot_matrix):
            raise ValueError(f'slot_matrix must be an affine world_matrix that preserves scale')

        self._slot_matrix = slot_matrix

    @beartype
    def world_matrix(self,
            pose: Pose,
            shim: typ.Union[np.ndarray, typ.Sequence[numbers.Real]] = VECTOR_ZERO,
            flip: int = 0) -> np.ndarray:
        """
        Calculate the affine matrix that places this magnet slot into the world in the correct major
        orientation except flip state.

        :param pose:
            Device Pose instance specifying gap and phase.

        :param shim:
            Shimming amount in XZS in the magnet aligned reference frame.

        :param flip:
            Flip matrix to use.

        :return:
            Affine matrix representing the major position of the slot in world space.
        """

        if not isinstance(shim, np.ndarray):
            shim = np.array(shim, dtype=np.float32)

        if shim.shape != (3,):
            raise ValueError(f'shim must be shape (3,) but is : '
                             f'{shim.shape}')

        if shim.dtype != np.float32:
            raise TypeError(f'shim must have dtype (float32) but is : '
                            f'{shim.dtype}')

        return self.slot_type.element_set.flip_matrix(flip) @ \
               self.slot_type.direction_matrix @ \
               self.slot_type.anchor_matrix @ \
               translate(*shim) @ \
               self.slot_matrix @ \
               self.girder.world_matrix(pose=pose)

    @beartype
    def to_radia(self,
            vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            pose: Pose,
            shim: typ.Union[np.ndarray, typ.Sequence[numbers.Real]] = VECTOR_ZERO,
            flip: int = 0,
            world_vector: bool = True) -> int:

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        matrix   = self.world_matrix(pose=pose, shim=shim, flip=flip)
        geometry = self.slot_type.element_set.geometry.transform(matrix)

        if not world_vector:
            vector = transform_rescaled_vectors(vector, matrix)

        return geometry.to_radia(vector)

    @beartype
    def bfield_from_vector(self,
            lattice: Lattice,
            vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            pose: Pose,
            shim: typ.Union[np.ndarray, typ.Sequence[numbers.Real]] = VECTOR_ZERO,
            flip: int = 0,
            world_vector: bool = True) -> Bfield:

        with RadiaCleanEnv():

            radia_object = self.to_radia(vector=vector, pose=pose, shim=shim,
                                         flip=flip, world_vector=world_vector)

            bfield = jnp_radia_evaluate_bfield_on_lattice(radia_object, lattice.world_lattice)

        return Bfield(lattice=lattice, field=bfield)

    @beartype
    def bfield(self,
            lattice: Lattice,
            pose: Pose,
            shim: typ.Union[np.ndarray, typ.Sequence[numbers.Real]] = VECTOR_ZERO,
            flip: int = 0) -> Bfield:

        return self.bfield_from_vector(lattice=lattice, pose=pose, shim=shim, flip=flip,
                                       vector=self.slot_type.element_set.vector, world_vector=False)

    @beartype
    def bfield_from_state(self,
                          lattice: Lattice,
                          pose: Pose,
                          state: ElementState) -> Bfield:

        if state.slot != self.qualified_name:
            raise ValueError(f'state must be assigned to this slot but is : '
                             f'slot={self.qualified_name} state={state.slot}')

        if state.candidate not in self.slot_type.element_set.candidates:
            raise ValueError(f'state must be assigned to a valid candidate for this slot but is : '
                             f'{state.candidate}')

        candidate = self.slot_type.element_set.candidates[state.candidate]

        return self.bfield_from_vector(lattice=lattice, pose=pose, shim=state.shim, flip=state.flip,
                                       vector=candidate.vector, world_vector=False)

    @property
    def girder(self):
        return self._girder

    @property
    @beartype
    def slot_type(self) -> SlotType:
        return self._slot_type

    @property
    @beartype
    def index(self) -> int:
        return self._index

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def qualified_name(self) -> str:
        return f'{self.girder.qualified_name}::{self.slot_type.qualified_name}::{self.name}'

    @property
    @beartype
    def period(self) -> str:
        return str(self._period)

    @property
    @beartype
    def slot_matrix(self) -> np.ndarray:
        return np_readonly(self._slot_matrix)
