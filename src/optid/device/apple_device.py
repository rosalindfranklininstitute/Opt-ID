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
import numpy as np


# Opt-ID Imports
from ..constants import MATRIX_IDENTITY, MATRIX_ROTS_270_ROTZ_180, MATRIX_ROTS_180, MATRIX_ROTS_270, \
                        MATRIX_ROTZ_180, MATRIX_ROTX_180, MATRIX_ROTS_270_ROTX_180, MATRIX_ROTS_90

from ..core.affine import translate
from .element_set import ElementSet
from .slot_type import SlotType
from .device import Device


class APPLEDevice(Device):

    @beartype
    def __init__(self,
            name: str,
            nperiod: int,
            hh: ElementSet,
            he: ElementSet,
            vv: ElementSet,
            ve: ElementSet,
            beam_gap: numbers.Real = 0.0625,
            interstice: numbers.Real = 0.0625,
            terminal: numbers.Real = 0.0625,
            symmetric: bool = True,
            world_matrix: np.ndarray = MATRIX_IDENTITY):

        super().__init__(name=name, world_matrix=world_matrix)

        if hh.name != 'HH':
            raise ValueError(f'hh_magnet.name must be HH but is : '
                             f'{hh.name}')

        if he.name != 'HE':
            raise ValueError(f'he_magnet.name must be HE but is : '
                             f'{he.name}')

        if vv.name != 'VV':
            raise ValueError(f'vv_magnet.name must be VV but is : '
                             f'{vv.name}')

        if ve.name != 'VE':
            raise ValueError(f've_magnet.name must be VE but is : '
                             f'{ve.name}')
    
        q1_hh_f = SlotType(name='+S', element_set=hh, anchor=(1, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        q1_hh_b = SlotType(name='-S', element_set=hh, anchor=(1, 0, 0.5), direction_matrix=MATRIX_ROTS_270_ROTZ_180)
        q1_he_f = SlotType(name='+S', element_set=he, anchor=(1, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        q1_he_b = SlotType(name='-S', element_set=he, anchor=(1, 0, 0.5), direction_matrix=MATRIX_ROTS_270_ROTZ_180)
        q1_vv_u = SlotType(name='+Z', element_set=vv, anchor=(1, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        q1_vv_d = SlotType(name='-Z', element_set=vv, anchor=(1, 0, 0.5), direction_matrix=MATRIX_ROTS_180)
        q1_ve_u = SlotType(name='+Z', element_set=ve, anchor=(1, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        q1_ve_d = SlotType(name='-Z', element_set=ve, anchor=(1, 0, 0.5), direction_matrix=MATRIX_ROTS_180)
    
        q2_hh_f = SlotType(name='+S', element_set=hh, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTS_270)
        q2_hh_b = SlotType(name='-S', element_set=hh, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q2_he_f = SlotType(name='+S', element_set=he, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTS_270)
        q2_he_b = SlotType(name='-S', element_set=he, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q2_vv_u = SlotType(name='+Z', element_set=vv, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q2_vv_d = SlotType(name='-Z', element_set=vv, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        q2_ve_u = SlotType(name='+Z', element_set=ve, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q2_ve_d = SlotType(name='-Z', element_set=ve, anchor=(0, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
    
        q3_hh_f = SlotType(name='+S', element_set=hh, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_180)
        q3_hh_b = SlotType(name='-S', element_set=hh, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_270_ROTX_180)
        q3_he_f = SlotType(name='+S', element_set=he, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_180)
        q3_he_b = SlotType(name='-S', element_set=he, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_270_ROTX_180)
        q3_vv_u = SlotType(name='+Z', element_set=vv, anchor=(0, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        q3_vv_d = SlotType(name='-Z', element_set=vv, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_180)
        q3_ve_u = SlotType(name='+Z', element_set=ve, anchor=(0, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        q3_ve_d = SlotType(name='-Z', element_set=ve, anchor=(0, 1, 0.5), direction_matrix=MATRIX_ROTS_180)
    
        q4_hh_f = SlotType(name='+S', element_set=hh, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTS_90)
        q4_hh_b = SlotType(name='-S', element_set=hh, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        q4_he_f = SlotType(name='+S', element_set=he, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTS_90)
        q4_he_b = SlotType(name='-S', element_set=he, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        q4_vv_u = SlotType(name='+Z', element_set=vv, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q4_vv_d = SlotType(name='-Z', element_set=vv, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        q4_ve_u = SlotType(name='+Z', element_set=ve, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTZ_180)
        q4_ve_d = SlotType(name='-Z', element_set=ve, anchor=(1, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
    
        l_matrix = translate(-(beam_gap / 2), 0, 0)
        r_matrix = translate( (beam_gap / 2), 0, 0)
    
        self.add_girder(name='Q1', girder_matrix=l_matrix, gap_vector=(0, 0.5, 0), phase_vector=(0, 0, -0.5))
        self.add_girder(name='Q2', girder_matrix=r_matrix, gap_vector=(0, 0.5, 0), phase_vector=(0, 0, 0.5))
        self.add_girder(name='Q3', girder_matrix=r_matrix, gap_vector=(0, -0.5, 0), phase_vector=(0, 0, -0.5))
        self.add_girder(name='Q4', girder_matrix=l_matrix, gap_vector=(0, -0.5, 0), phase_vector=(0, 0, 0.5))
        
        period = 'START'
        self.add_slots(slot_types={ 'Q1': q1_he_f, 'Q2': q2_he_f, 'Q3': q3_he_b, 'Q4': q4_he_b }, period=period, after_spacing=terminal)
        self.add_slots(slot_types={ 'Q1': q1_ve_d, 'Q2': q2_ve_d, 'Q3': q3_ve_d, 'Q4': q4_ve_d }, period=period, after_spacing=interstice)
        self.add_slots(slot_types={ 'Q1': q1_he_b, 'Q2': q2_he_b, 'Q3': q3_he_f, 'Q4': q4_he_f }, period=period, after_spacing=interstice)
    
        for index in range(nperiod):
            period = f'{index:04d}'
            self.add_slots(slot_types={ 'Q1': q1_vv_u, 'Q2': q2_vv_u, 'Q3': q3_vv_u, 'Q4': q4_vv_u }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_hh_f, 'Q2': q2_hh_f, 'Q3': q3_hh_b, 'Q4': q4_hh_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_vv_d, 'Q2': q2_vv_d, 'Q3': q3_vv_d, 'Q4': q4_vv_d }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_hh_b, 'Q2': q2_hh_b, 'Q3': q3_hh_f, 'Q4': q4_hh_f }, period=period, after_spacing=interstice)
    
        if symmetric:
            period = 'SYM'
            self.add_slots(slot_types={ 'Q1': q1_vv_u, 'Q2': q2_vv_u, 'Q3': q3_vv_u, 'Q4': q4_vv_u }, period=period, after_spacing=interstice)
            
            period = 'END'
            self.add_slots(slot_types={ 'Q1': q1_he_f, 'Q2': q2_he_f, 'Q3': q3_he_b, 'Q4': q4_he_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_ve_d, 'Q2': q2_ve_d, 'Q3': q3_ve_d, 'Q4': q4_ve_d }, period=period, after_spacing=terminal)
            self.add_slots(slot_types={ 'Q1': q1_he_b, 'Q2': q2_he_b, 'Q3': q3_he_f, 'Q4': q4_he_f }, period=period)
        else:
            period = 'ANTISYM'
            self.add_slots(slot_types={ 'Q1': q1_vv_u, 'Q2': q2_vv_u, 'Q3': q3_vv_u, 'Q4': q4_vv_u }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_hh_f, 'Q2': q2_hh_f, 'Q3': q3_hh_b, 'Q4': q4_hh_b }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_vv_d, 'Q2': q2_vv_d, 'Q3': q3_vv_d, 'Q4': q4_vv_d }, period=period, after_spacing=interstice)

            period = 'END'
            self.add_slots(slot_types={ 'Q1': q1_he_b, 'Q2': q2_he_b, 'Q3': q3_he_f, 'Q4': q4_he_f }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'Q1': q1_ve_u, 'Q2': q2_ve_u, 'Q3': q3_ve_u, 'Q4': q4_ve_u }, period=period, after_spacing=terminal)
            self.add_slots(slot_types={ 'Q1': q1_he_f, 'Q2': q2_he_f, 'Q3': q3_he_b, 'Q4': q4_he_b }, period=period)
