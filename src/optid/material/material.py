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


TVector = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]


class Material:

    @beartype
    def to_radia(self,
            vector: TVector) -> typ.Optional[int]:
        """
        Instance a new Radia object containing all the polyhedra in the Geometry instance.

        :param vector:
            World space directional vector for the magnetic field of the polyhedra.

        :return:
            Integer handle to a Radia object.
        """

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        return self.build(vector)

    @beartype
    def build(self,
            vector: TVector) -> typ.Optional[int]:

        return None


DefaultMaterial = Material
