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
import jax.numpy as jnp

# Opt-ID Imports
from ..core.utils import np_readonly
from ..core.affine import transform_points, jnp_transform_points
from ..core.lattice import unit_lattice, jnp_unit_lattice, unit_to_orthonormal_matrix


class Lattice:

    @beartype
    def __init__(self,
            unit_to_world_matrix: np.ndarray,
            shape: typ.Tuple[int, int, int]):
        """
        Construct a Lattice instance from a subdivision of a unit cube at origin and a matrix that transforms it
        into world space coordinates.

        :param unit_to_world_matrix:
            Affine matrix that transforms the unit cube at origin spanning -0.5 to +0.5 into world space.

        :param shape:
            Subdivision in each axis for the lattice.
        """

        if unit_to_world_matrix.shape != (4, 4):
            raise ValueError(f'unit_to_world_matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{unit_to_world_matrix.shape}')

        if unit_to_world_matrix.dtype != np.float32:
            raise TypeError(f'unit_to_world_matrix must have dtype (float32) but is : '
                            f'{unit_to_world_matrix.dtype}')

        self._unit_to_world_matrix = unit_to_world_matrix

        if np.any(np.array(shape) <= 0):
            raise ValueError(f'shape must be a 3-tuple of positive integers but is : '
                             f'{shape}')
        self._shape = shape

        lattice = self.world_lattice
        x_step = 0 if shape[0] == 1 else np.mean(np.linalg.norm(lattice[0, 0, 0] - lattice[1, 0, 0], axis=-1))
        z_step = 0 if shape[1] == 1 else np.mean(np.linalg.norm(lattice[0, 0, 0] - lattice[0, 1, 0], axis=-1))
        s_step = 0 if shape[2] == 1 else np.mean(np.linalg.norm(lattice[0, 0, 0] - lattice[0, 0, 1], axis=-1))
        self._step = (x_step, z_step, s_step)

        # Derive complementary matrices
        self._unit_to_orthonormal_matrix  = unit_to_orthonormal_matrix(*self.shape)
        self._orthonormal_to_unit_matrix  = np.linalg.inv(self.unit_to_orthonormal_matrix)
        self._world_to_unit_matrix        = np.linalg.inv(self.unit_to_world_matrix)
        self._world_to_orthonormal_matrix = self.world_to_unit_matrix @ self.unit_to_orthonormal_matrix
        self._orthonormal_to_world_matrix = self.orthonormal_to_unit_matrix @ self.unit_to_world_matrix

    @beartype
    def __eq__(self, other) -> bool:
        """
        Compare this Lattice instance to another to see if they represent the same lattice.

        :param other:
            Lattice to compare to.

        :return:
            True if the two lattices have the same unit to world matrix and the same subdivision, otherwise False.
        """

        if self is other:
            return True

        if not isinstance(other, Lattice):
            return False

        if self.shape != other.shape:
            return False

        return np.allclose(self.unit_to_world_matrix, other.unit_to_world_matrix, atol=1e-5)

    @beartype
    def __ne__(self, other) -> bool:
        """
        Compare this Lattice instance to another to see if they represent the different lattices.

        :param other:
            Lattice to compare to.

        :return:
            False if the two lattices have the same unit to world matrix and the same subdivision, otherwise True.
        """
        return not (self == other)

    @property
    @beartype
    def jnp_unit_lattice(self) -> typ.Any:
        """
        Lattice tensor with the desired shape centred at origin spanning -0.5 to +0.5
        """
        return jnp_unit_lattice(*self.shape)

    @property
    @beartype
    def unit_lattice(self) -> np.ndarray:
        """
        Lattice tensor with the desired shape centred at origin spanning -0.5 to +0.5
        """
        return unit_lattice(*self.shape)

    @property
    @beartype
    def jnp_world_lattice(self) -> typ.Any:
        """
        Lattice tensor with the desired shape in world coordinates.
        """
        return jnp_transform_points(self.jnp_unit_lattice, self.unit_to_world_matrix)

    @property
    @beartype
    def world_lattice(self) -> np.ndarray:
        """
        Lattice tensor with the desired shape in world coordinates.
        """
        return transform_points(self.unit_lattice, self.unit_to_world_matrix)

    @property
    @beartype
    def world_to_unit_matrix(self) -> np.ndarray:
        """
        Matrix that maps world space coordinates to unit coordinates centred at origin -0.5 to +0.5.
        """
        return np_readonly(self._world_to_unit_matrix)

    @property
    @beartype
    def world_to_orthonormal_matrix(self) -> np.ndarray:
        """
        Matrix that maps world space coordinates to orthonormal space.
        """
        return np_readonly(self._world_to_orthonormal_matrix)

    @property
    @beartype
    def orthonormal_to_unit_matrix(self) -> np.ndarray:
        """
        Matrix that maps orthonormal space coordinates to unit coordinates centred at origin -0.5 to +0.5.
        """
        return np_readonly(self._orthonormal_to_unit_matrix)

    @property
    @beartype
    def orthonormal_to_world_matrix(self) -> np.ndarray:
        """
        Matrix that maps orthonormal space coordinates to world space.
        """
        return np_readonly(self._orthonormal_to_world_matrix)

    @property
    @beartype
    def unit_to_orthonormal_matrix(self) -> np.ndarray:
        """
        Matrix that maps from unit coordinates centred at origin -0.5 to +0.5 to orthonormal space.
        """
        return np_readonly(self._unit_to_orthonormal_matrix)

    @property
    @beartype
    def unit_to_world_matrix(self) -> np.ndarray:
        """
        Matrix that maps unit coordinates centred at origin -0.5 to +0.5 to world space.
        """
        return np_readonly(self._unit_to_world_matrix)

    @property
    @beartype
    def shape(self) -> typ.Tuple[int, int, int]:
        """
        Subdivision shape of the lattice.
        """
        return self._shape

    @property
    @beartype
    def step(self) -> typ.Tuple[numbers.Real, numbers.Real, numbers.Real]:
        return self._step
