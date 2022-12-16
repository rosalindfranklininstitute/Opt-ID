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
from types import MappingProxyType
from more_itertools import SequenceView
from collections import Counter, defaultdict
from beartype import beartype
import numbers
import beartype.typing as typ
import numpy as np
import re
import radia as rad

# Opt-ID Imports
from ..utils.cached import Memoized, cached_property, invalidates_cached_properties
from ..core.affine import is_scale_preserving
from ..core.bfield import RadiaCleanEnv, jnp_radia_evaluate_bfield_on_lattice
from ..core.utils import np_readonly
from ..bfield import Bfield
from ..lattice import Lattice
from .girder import Girder
from .element_set import ElementSet
from .slot_type import SlotType
from .slot import Slot
from .pose import Pose
from .device_state import DeviceState


class Device(Memoized):

    @beartype
    def __init__(self,
            name: str,
            world_matrix: np.ndarray):
        """
        Construct a Device instance.

        :param name:
            String name for the device.

        :param world_matrix:
            Affine matrix for the placing this device into the world.
        """

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if world_matrix.shape != (4, 4):
            raise ValueError(f'world_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{world_matrix.shape}')

        if world_matrix.dtype != np.float32:
            raise TypeError(f'world_matrix must have dtype (float32) but is : '
                            f'{world_matrix.dtype}')

        if not is_scale_preserving(world_matrix):
            raise ValueError(f'world_matrix must be an affine world_matrix that preserves scale')

        self._world_matrix = world_matrix

        self._girders = dict()

    @invalidates_cached_properties
    @beartype
    def add_girder(self,
                   name: str,
                   girder_matrix: np.ndarray,
                   gap_vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
                   phase_vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]]):

        if name in self._girders:
            raise ValueError(f'girders already contains a girder with name : '
                             f'{name}')

        girder = Girder(device=self, name=name, girder_matrix=girder_matrix,
                        gap_vector=gap_vector, phase_vector=phase_vector)

        self._girders[name] = girder

        return girder

    @invalidates_cached_properties
    @beartype
    def add_slots(
            self,
            period: str,
            slot_types: typ.Mapping[str, SlotType],
            after_spacing: numbers.Real = 0,
            name: typ.Optional[str] = None):

        if len(slot_types) != len(self.girders):
            raise ValueError(f'slot_types must contain keys for all girders in the device : '
                             f'{len(slot_types)} != {len(self.girders)}')

        for girder in self.girders.values():

            if girder.name not in slot_types:
                raise ValueError(f'slot_types must contain keys for all girders in the device : '
                                 f'{girder.name} not in {list(slot_types.keys())}')

            slot_type = slot_types[girder.name]

            girder.add_slot(period=period, slot_type=slot_type,
                          after_spacing=after_spacing, name=name)

    def validate(self):

        elements = dict()

        for girder in self.girders.values():
            for slot in girder.slots:

                key = slot.slot_type.element_set.name
                if key not in elements:
                    elements[key] = slot.slot_type.element_set
                elif elements[key] is not slot.slot_type.element_set:
                    raise ValueError(f'multiple magnets with same name refer to different objects : '
                                     f'{key}, {slot.qualified_name}')

        for element_type in self.nslot_by_type.keys():



            if self.element_sets[element_type].is_magnetized and \
                    (self.nslot_by_type[element_type] > self.element_sets[element_type].ncandidate):
                raise ValueError(f'device has more slots of type "{element_type} than candidates : '
                                 f'slots={self.nslot_by_type[element_type]} > '
                                 f'candidates={self.element_sets[element_type].ncandidate}')

    @beartype
    def full_bfield(self,
            lattice: Lattice,
            pose: Pose) -> Bfield:

        radia_objects = list()
        with RadiaCleanEnv():

            for girder in self.girders.values():
                for slot in girder.slots:
                    if slot.slot_type.element_set.is_magnetized:
                        radia_objects.append(slot.to_radia(vector=slot.slot_type.element_set.vector, pose=pose, world_vector=False))

            radia_object = rad.ObjCnt(radia_objects)

            IM = rad.RlxPre(radia_object)
            res = rad.RlxAuto(IM, 0.001, 5000)

            print('Relaxation Results:', res)

            girder_field = jnp_radia_evaluate_bfield_on_lattice(radia_object, lattice.world_lattice)

        return Bfield(lattice=lattice, field=girder_field)

    @beartype
    def bfield(self,
            lattice: Lattice,
            pose: Pose,
            add_noise=False) -> Bfield:

        device_field = None
        for girder in self.girders.values():
            girder_field = girder.bfield(lattice=lattice, pose=pose, add_noise=add_noise).field
            device_field = girder_field if (device_field is None) else (device_field + girder_field)

        return Bfield(lattice=lattice, field=device_field)

    @beartype
    def bfield_from_state(self,
            lattice: Lattice,
            pose: Pose,
            state: DeviceState) -> Bfield:

        device_field = None
        for girder in self.girders.values():
            girder_field = girder.bfield_from_state(lattice=lattice, pose=pose, state=state).field
            device_field = girder_field if (device_field is None) else (device_field + girder_field)

        return Bfield(lattice=lattice, field=device_field)

    @cached_property
    @beartype
    def girders(self) -> typ.Mapping[str, Girder]:
        return MappingProxyType(self._girders)

    @cached_property
    @beartype
    def slots_by_type(self) -> typ.Mapping[str, typ.Mapping[str, typ.Sequence[Slot]]]:

        slots = defaultdict(dict)
        for girder in self.girders.values():
            for element_type, element_slots in girder.slots_by_type.items():
                slots[element_type][girder.name] = element_slots

        return MappingProxyType({ magnet_type: MappingProxyType({ girder_name: SequenceView(magnet_slots)
                                  for girder_name, magnet_slots in girders.items() })
                                  for magnet_type, girders in slots.items() })

    @cached_property
    @beartype
    def element_sets(self) -> typ.Mapping[str, ElementSet]:
        return MappingProxyType({ element_set.name: element_set
                                  for girder in self.girders.values()
                                  for element_set in girder.element_sets.values() })

    @cached_property
    @beartype
    def period_lengths(self) -> typ.Mapping[str, numbers.Real]:

        period_lengths = defaultdict(list)
        for girder in self.girders.values():
            for period, length in girder.period_lengths.items():
                period_lengths[period].append(length)

        return MappingProxyType({ period: np.mean(period_lengths)
                                  for period, period_lengths in period_lengths.items() })

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

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @cached_property
    @beartype
    def world_matrix(self) -> np.ndarray:
        return np_readonly(self._world_matrix)

    @cached_property
    @beartype
    def nslot(self) -> int:
        return sum(girder.nslot for girder in self.girders.values())

    @cached_property
    @beartype
    def nslot_by_type(self) -> typ.Mapping[str, int]:
        return MappingProxyType(Counter([ slot.slot_type.element_set.name
                                          for girder in self.girders.values()
                                          for slot in girder.slots ]))

    @cached_property
    @beartype
    def length(self) -> numbers.Real:
        return max(girder.length for girder in self.girders.values())

