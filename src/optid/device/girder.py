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
from collections import defaultdict, Counter
from types import MappingProxyType
from beartype import beartype
from more_itertools import SequenceView
import numbers
import beartype.typing as typ
import numpy as np
import re
import radia as rad

# Opt-ID Imports
from ..utils.cached import Memoized, cached_property, invalidates_cached_properties
from ..constants import MATRIX_IDENTITY
from ..core.utils import np_readonly
from ..core.affine import translate, is_scale_preserving
from ..core.bfield import RadiaCleanEnv, jnp_radia_evaluate_bfield_on_lattice
from ..bfield import Bfield
from ..lattice import Lattice
from .pose import Pose
from .slot import Slot
from .slot_type import SlotType
from .element_set import ElementSet
from .device_state import DeviceState


class Girder(Memoized):

    @beartype
    def __init__(self,
            device,
            name: str,
            gap_vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            phase_vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            girder_matrix: np.ndarray = MATRIX_IDENTITY):
        """
        Construct a Girder instance.

        :param device:
            Parent Device instance this Girder is a member of.

        :param name:
            String name for the girder.

        :param girder_matrix:
            Affine matrix for the placing this girder into the world.
            
        :param gap_vector:
            Vector representing the Girder's response to the gap value.

        :param phase_vector:
            Vector representing the Girder's response to the phase value.
        """

        self._device = device

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if girder_matrix.shape != (4, 4):
            raise ValueError(f'girder_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{girder_matrix.shape}')

        if girder_matrix.dtype != np.float32:
            raise TypeError(f'girder_matrix must have dtype (float32) but is : '
                            f'{girder_matrix.dtype}')

        if not is_scale_preserving(girder_matrix):
            raise ValueError(f'girder_matrix must be an affine world_matrix that preserves scale')

        self._girder_matrix = girder_matrix
        
        if not isinstance(gap_vector, np.ndarray):
            gap_vector = np.array(gap_vector, dtype=np.float32)

        if gap_vector.shape != (3,):
            raise ValueError(f'gap_vector must be shape (3,) but is : '
                             f'{gap_vector.shape}')

        if gap_vector.dtype != np.float32:
            raise TypeError(f'gap_vector must have dtype (float32) but is : '
                            f'{gap_vector.dtype}')
        
        self._gap_vector = gap_vector

        if not isinstance(phase_vector, np.ndarray):
            phase_vector = np.array(phase_vector, dtype=np.float32)

        if phase_vector.shape != (3,):
            raise ValueError(f'phase_vector must be shape (3,) but is : '
                             f'{phase_vector.shape}')

        if phase_vector.dtype != np.float32:
            raise TypeError(f'phase_vector must have dtype (float32) but is : '
                            f'{phase_vector.dtype}')

        self._phase_vector = phase_vector

        self._centre_matrix = MATRIX_IDENTITY
        self._smin, self._smax, self._spacing = 0, 0, 0
        self._slots = list()
        self._slots_by_name = dict()
        self._period_bounds = dict()

    def invalidate_cached_properties(self):
        # Handle memoized cached parameters
        super().invalidate_cached_properties()
        self.device.invalidate_cached_properties()

    @beartype
    def world_matrix(self, pose: Pose) -> np.ndarray:
        """
        Calculate the affine matrix that places this girder into the world.

        :param pose:
            Device Pose instance specifying gap and phase.

        :return:
            Affine matrix representing the major position of the girder in world space.
        """

        return self.centre_matrix @ \
               translate(*((self.gap_vector * pose.gap) + (self.phase_vector * pose.phase))) @ \
               self.girder_matrix @ \
               self.device.world_matrix

    @invalidates_cached_properties
    @beartype
    def add_slot(self,
            period: str,
            slot_type: SlotType,
            after_spacing: numbers.Real = 0,
            name: typ.Optional[str] = None):

        bmin, bmax = slot_type.bounds
        thickness  = (bmax[2] - bmin[2])

        if self.nslot == 0:
            cur_smax = self._smin = bmin[2]
            self._smax = bmax[2]
        else:
            cur_smax = self._smax
            self._smax += self._spacing + thickness

        self._spacing = 0

        if period not in self._period_bounds:
            self._period_bounds[period] = (float(cur_smax), float(self._smax))
        else:
            if self._slots[-1].period != period:
                raise ValueError(f'period already defined but previous slot is different : '
                                 f'previous={self._slots[-1].period} current={period}')

            p0, p1 = self._period_bounds[period]
            self._period_bounds[period] = (p0, float(self._smax))

        slot_matrix = translate(0, 0, (self._smax - bmax[2]))

        self._centre_matrix = translate(0, 0, (-self._smin) - ((self._smax - self._smin) / 2.0))

        name = name if (name is not None) else f'{self.nslot:06d}'

        slot = Slot(index=self.nslot, girder=self, name=name, period=period, slot_type=slot_type, slot_matrix=slot_matrix)

        self._slots.append(slot)

        if slot.name in self._slots_by_name:
            raise ValueError(f'slot must have unique name but conflicts with another in this girder : '
                             f'{slot.name}')

        self._slots_by_name[slot.name] = slot

        self.add_spacing(spacing=after_spacing)

        # Validate the whole device is still valid
        self.device.validate()

        return slot

    @beartype
    def add_spacing(self, spacing: numbers.Real):

        if spacing < 0:
            raise ValueError(f'spacing must be > 0 but is : '
                             f'{spacing}')

        self._spacing += spacing

    @beartype
    def full_bfield(self,
            lattice: Lattice,
            pose: Pose) -> Bfield:

        radia_objects = list()
        with RadiaCleanEnv():

            for slot in self.slots:
                if slot.slot_type.element_set.is_magnetized:
                    radia_objects.append(slot.to_radia(vector=slot.slot_type.element_set.vector, pose=pose, world_vector=False))

            radia_object = rad.ObjCnt(radia_objects)
            girder_field = jnp_radia_evaluate_bfield_on_lattice(radia_object, lattice.world_lattice)

        return Bfield(lattice=lattice, field=girder_field)

    @beartype
    def bfield(self,
            lattice: Lattice,
            pose: Pose,
            add_noise=False) -> Bfield:

        girder_field = None
        for slot in self.slots:
            if slot.slot_type.element_set.is_magnetized:
                slot_field = slot.bfield(lattice=lattice, pose=pose, add_noise=add_noise).field
                girder_field = slot_field if (girder_field is None) else (girder_field + slot_field)

        return Bfield(lattice=lattice, field=girder_field)

    @beartype
    def bfield_from_state(self,
                           lattice: Lattice,
                           pose: Pose,
                           state: DeviceState) -> Bfield:

        girder_field = None
        for slot in self.slots:
            if slot.slot_type.element_set.is_magnetized:
                if slot.qualified_name not in state.slots:
                    raise ValueError(f'slot.qualified_name is not found in genome.slots : '
                                     f'{slot.qualified_name}')

                slot_field = slot.bfield_from_state(lattice=lattice, pose=pose,
                                                    state=state.element_state(girder=self.name,
                                                                              slot=slot.name)).field

                girder_field = slot_field if (girder_field is None) else (girder_field + slot_field)

        return Bfield(lattice=lattice, field=girder_field)

    @cached_property
    @beartype
    def slots(self) -> typ.Sequence[Slot]:
        return SequenceView(self._slots)

    @cached_property
    @beartype
    def nslot(self) -> int:
        return len(self.slots)

    @cached_property
    @beartype
    def slots_by_type(self) -> typ.Mapping[str, typ.Sequence[Slot]]:

        slots = defaultdict(list)
        for slot in self.slots:
            slots[slot.slot_type.element_set.name].append(slot)

        return MappingProxyType({ magnet_type: SequenceView(magnet_slots)
                                  for magnet_type, magnet_slots in slots.items() })

    @cached_property
    @beartype
    def nslot_by_type(self) -> typ.Mapping[str, int]:
        return MappingProxyType(Counter([slot.slot_type.element_set.name
                                         for slot in self.slots]))

    @cached_property
    @beartype
    def element_sets(self) -> typ.Mapping[str, ElementSet]:
        return MappingProxyType({ slot.slot_type.element_set.name: slot.slot_type.element_set
                                  for slot in self.slots })

    @property
    def device(self):
        return self._device

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def qualified_name(self) -> str:
        return f'{self.device.name}::{self.name}'

    @cached_property
    @beartype
    def girder_matrix(self) -> np.ndarray:
        return np_readonly(self._girder_matrix)

    @cached_property
    @beartype
    def centre_matrix(self) -> np.ndarray:
        return np_readonly(self._centre_matrix)

    @cached_property
    @beartype
    def gap_vector(self) -> np.ndarray:
        return np_readonly(self._gap_vector)

    @cached_property
    @beartype
    def phase_vector(self) -> np.ndarray:
        return np_readonly(self._phase_vector)

    @cached_property
    @beartype
    def length(self) -> numbers.Real:
        return float(self._smax - self._smin)

    @cached_property
    @beartype
    def period_lengths(self) -> typ.Mapping[str, float]:
        return MappingProxyType({ period: float(p1 - p0)
                                  for period, (p0, p1) in self._period_bounds.items() })

    @cached_property
    @beartype
    def period_length(self) -> numbers.Real:
        re_period = re.compile(r'\d+')
        return np.mean(list(map((lambda kv: kv[1]),
                                filter((lambda kv: (re_period.match(kv[0]) is not None)),
                                       self.period_lengths.items()))))

    @cached_property
    @beartype
    def nperiod(self) -> int:
        re_period = re.compile(r'\d+')
        return len(list(filter((lambda kv: (re_period.match(kv[0]) is not None)),
                               self.period_lengths.items())))
