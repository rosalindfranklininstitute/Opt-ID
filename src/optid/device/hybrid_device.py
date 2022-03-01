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
from beartype import beartype
import numbers
import numpy as np


# Opt-ID Imports
from ..constants import MATRIX_IDENTITY, MATRIX_ROTX_180
from .element_set import ElementSet
from .slot_type import SlotType
from .device import Device


class HybridDevice(Device):

    @beartype
    def __init__(self,
            name: str,
            nperiod: int,
            hh: ElementSet,
            he: ElementSet,
            ht: ElementSet,
            pp: ElementSet,
            pt: ElementSet,
            interstice: numbers.Real = 0.0625,
            symmetric: bool = True,
            world_matrix: np.ndarray = MATRIX_IDENTITY):

        super().__init__(name=name, world_matrix=world_matrix)

        if hh.name != 'HH':
            raise ValueError(f'hh.name must be HH but is : '
                             f'{hh.name}')

        if he.name != 'HE':
            raise ValueError(f'he.name must be HE but is : '
                             f'{he.name}')

        if ht.name != 'HT':
            raise ValueError(f'ht.name must be HT but is : '
                             f'{ht.name}')

        if pp.name != 'PP':
            raise ValueError(f'pp.name must be PP but is : '
                             f'{pp.name}')

        if pt.name != 'PT':
            raise ValueError(f'pt.name must be PT but is : '
                             f'{pt.name}')

        if nperiod < 0:
            raise ValueError(f'nperiod must be >= 0 but is : '
                             f'{nperiod}')

        if interstice < 0:
            raise ValueError(f'interstice must be >= 0 but is : '
                             f'{interstice}')

        hh_top_f = SlotType(name='+S', element=hh, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        hh_top_b = SlotType(name='-S', element=hh, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        he_top_f = SlotType(name='+S', element=he, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        he_top_b = SlotType(name='-S', element=he, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        ht_top_f = SlotType(name='+S', element=ht, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        ht_top_b = SlotType(name='-S', element=ht, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        pp_top   = SlotType(name='PP', element=pp, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        pt_top   = SlotType(name='PT', element=pt, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)

        hh_btm_f = SlotType(name='+S', element=hh, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        hh_btm_b = SlotType(name='-S', element=hh, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        he_btm_f = SlotType(name='+S', element=he, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        he_btm_b = SlotType(name='-S', element=he, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        ht_btm_f = SlotType(name='+S', element=ht, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        ht_btm_b = SlotType(name='-S', element=ht, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        pp_btm   = SlotType(name='PP', element=pp, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        pt_btm   = SlotType(name='PT', element=pt, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)

        self.add_girder(name='TOP', girder_matrix=MATRIX_IDENTITY, gap_vector=(0, 0.5, 0), phase_vector=(0, 0, 0))
        self.add_girder(name='BTM', girder_matrix=MATRIX_IDENTITY, gap_vector=(0, -0.5, 0), phase_vector=(0, 0, 0))

        period = 'START'
        self.add_slots(slot_types={ 'TOP': ht_top_f, 'BTM': ht_btm_b }, period=period, after_spacing=interstice)
        self.add_slots(slot_types={ 'TOP': pt_top,   'BTM': pt_btm   }, period=period, after_spacing=interstice)
        self.add_slots(slot_types={ 'TOP': he_top_b, 'BTM': he_btm_f }, period=period, after_spacing=interstice)

        for index in range(nperiod):
            period = f'{index:04d}'
            self.add_slots(slot_types={ 'TOP': pp_top,   'BTM': pp_btm   }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_f, 'BTM': hh_btm_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pp_top,   'BTM': pp_btm   }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_b, 'BTM': hh_btm_f }, period=period, after_spacing=interstice)

        if symmetric:
            period = 'SYM'
            self.add_slots(slot_types={ 'TOP': pp_top,   'BTM': pp_btm   }, period=period, after_spacing=interstice)

            period = 'END'
            self.add_slots(slot_types={ 'TOP': he_top_f, 'BTM': he_btm_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pt_top,   'BTM': pt_btm   }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': ht_top_b, 'BTM': ht_btm_f }, period=period)

        else:
            period = 'ANTISYM'
            self.add_slots(slot_types={ 'TOP': pp_top,   'BTM': pp_btm   }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_f, 'BTM': hh_btm_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pp_top,   'BTM': pp_btm   }, period=period, after_spacing=interstice)

            period = 'END'
            self.add_slots(slot_types={ 'TOP': he_top_b, 'BTM': he_btm_f }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pt_top,   'BTM': pt_btm   }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': ht_top_f, 'BTM': ht_btm_b }, period=period)
