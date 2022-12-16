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
import radia as rad


# Opt-ID Imports
from .material import Material


TVector = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]


class NamedMaterial(Material):

    def __init__(self,
            name: str,
            remanent_magnetization: typ.Optional[numbers.Real] = None):
        super().__init__()

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if (remanent_magnetization is not None) and (remanent_magnetization < 0):
            raise ValueError(f'remanent_magnetization must be >= 0 but is : '
                             f'{remanent_magnetization}')

        self._remanent_magnetization = remanent_magnetization

    @beartype
    def build(self,
            vector: TVector) -> typ.Optional[int]:

        if self.remanent_magnetization is None:
            return rad.MatStd(self.name)

        return rad.MatStd(self.name, self.remanent_magnetization)

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def remanent_magnetization(self) -> typ.Optional[numbers.Real]:
        return self._remanent_magnetization
