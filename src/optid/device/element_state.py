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
import beartype.typing as typ
import numpy as np


# Opt-ID Imports
from ..constants import VECTOR_ZERO
from ..core.utils import np_readonly
# from .magnet_slot import MagnetSlot
from .element_candidate import ElementCandidate

TVector = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]


class ElementState:

    @beartype
    def __init__(self,
                 slot: typ.Optional[typ.Any],
                 candidate: ElementCandidate,
                 shim: TVector = VECTOR_ZERO,
                 flip: int = 0):
        """
        Construct a ElementState instance.

        :param slot:
            MagnetSlot instance if this state is assigned to one.

        :param candidate:
            ElementCandidate instance.

        :param shim:
            Shimming amount in XZS in the magnet aligned reference frame.

        :param flip:
            Integer flip state for the selection.
        """

        if slot.candidates[candidate.name] is not candidate:
            raise ValueError(f'candidate instance does not match a valid candidate of this slot')

        self._slot = slot
        self._candidate = candidate

        if not isinstance(shim, np.ndarray):
            shim = np.array(shim, dtype=np.float32)

        if shim.shape != (3,):
            raise ValueError(f'shim must be shape (3,) but is : '
                             f'{shim.shape}')

        if shim.dtype != np.float32:
            raise TypeError(f'shim must have dtype (float32) but is : '
                            f'{shim.dtype}')

        self._shim = shim

        if (flip < 0) or (flip >= slot.magnet.nflip):
            raise ValueError(f'flip must be [0, {slot.magnet.nflip}) but is : '
                             f'{flip}')

        self._flip = flip

    @property
    @beartype
    def slot(self) -> typ.Optional[typ.Any]:
        return self._slot

    @property
    @beartype
    def candidate(self) -> ElementCandidate:
        return self._candidate

    @property
    @beartype
    def shim(self) -> np.ndarray:
        return np_readonly(self._shim)

    @property
    @beartype
    def flip(self) -> int:
        return self._flip
