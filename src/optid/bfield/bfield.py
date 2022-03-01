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
import jax.numpy as jnp

# Opt-ID Imports
from optid.lattice import Lattice


class Bfield:

    @beartype
    def __init__(self,
                 lattice: Lattice,
                 field: jnp.ndarray):

        self._lattice = lattice

        if field.shape[:3] != lattice.shape:
            raise ValueError(f'field must be have leading dimensions matching the lattice shape but is : '
                             f'field={field.shape[:3]} != lattice={lattice.shape}')

        if field.shape[3:] != (3,):
            raise ValueError(f'field must be have trailing dimensions (3,) but is : '
                             f'field={field.shape[3:]}')

        if field.dtype != jnp.float32:
            raise TypeError(f'field must have dtype (float32) but is : '
                            f'{field.dtype}')

        self._field = field

    @property
    @beartype
    def lattice(self) -> Lattice:
        return self._lattice

    @property
    @beartype
    def field(self) -> jnp.ndarray:
        return self._field
