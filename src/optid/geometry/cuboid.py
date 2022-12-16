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
from ..geometry import ExtrudedPolygon
from ..material import Material, DefaultMaterial


class Cuboid(ExtrudedPolygon):

    @beartype
    def __init__(self,
            shape: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            subdiv: numbers.Real = 0,
            material: Material = DefaultMaterial(),
            **tetgen_kargs):
        """
        Construct a Cuboid instance.

        :param shape:
            Aligned size vector in 3-space.

        :param subdiv:
            Scale for introducing new vertices to aid in subdivision. Ignored if less than or equal to zero.

        :param **tetgen_kargs:
            All additional parameters are forwarded to TetGen's tetrahedralize function.
        """

        if not isinstance(shape, np.ndarray):
            shape = np.array(shape, dtype=np.float32)

        if shape.shape != (3,):
            raise ValueError(f'shape must be a vector of shape (3,) but is : '
                             f'{shape.shape}')

        if shape.dtype != np.float32:
            raise TypeError(f'shape must have dtype (float32) but is : '
                            f'{shape.dtype}')

        if np.any(shape <= 0):
            raise ValueError(f'shape must be greater than zero in every dimension but is : '
                             f'{shape}')

        x, z, s = shape.tolist()
        x *= 0.5
        z *= 0.5

        polygon = np.array(
            [[-x, -z], [-x, z], [x, z], [x, -z]], dtype=np.float32)

        super().__init__(polygon=polygon, thickness=s, subdiv=subdiv, material=material, **tetgen_kargs)
