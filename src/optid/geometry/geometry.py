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
import typing as typ
import numpy as np
import jax.numpy as jnp
import radia as rad

# Opt-ID Imports
from ..core.utils import np_readonly
from ..core.affine import transform_points, is_scale_preserving
from ..material import Material, DefaultMaterial


class Geometry:

    @beartype
    def __init__(self,
            vertices: typ.Union[np.ndarray, typ.Sequence[typ.Sequence[numbers.Real]]],
            polyhedra: typ.Sequence[typ.Sequence[typ.Sequence[int]]],
            material: Material = DefaultMaterial()):
        """
        Construct a Geometry instance from a set of unique vertices in 3-space and a list of polygons.

        :param vertices:
            Tensor of vertices in 3-space of shape (N, 3).

        :param polyhedra:
            List of lists of lists of integer vertex IDs in the range [0, N).
            Each polyhedra must have at least 4 faces. Each face must have at least 3 vertices.
        """

        if not isinstance(vertices, np.ndarray):

            def is_vertex_not_3d(vertex) -> bool:
                return len(vertex) != 3

            if any(map(is_vertex_not_3d, vertices)):
                raise ValueError(f'vertices must be a list of 3D XZS coordinates but is : '
                                 f'{vertices}')

            vertices = np.array(vertices, dtype=np.float32)

        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f'vertices must be shape (N, 3) but is : '
                             f'{vertices.shape}')

        if vertices.dtype != np.float32:
            raise TypeError(f'vertices must have dtype (float32) but is : '
                            f'{vertices.dtype}')

        self._vertices = vertices

        def is_not_list(seq):
            return not isinstance(seq, list)

        if is_not_list(polyhedra) or \
           any(map(is_not_list, polyhedra)) or \
           any(any(map(is_not_list, faces)) for faces in polyhedra):
            # Coerce sequence of sequences of sequences into list of lists of lists
            polyhedra = [[[vertex for vertex in face] for face in faces] for faces in polyhedra]

        if len(polyhedra) < 1:
            raise ValueError(f'polyhedra must have at least one element but is : '
                             f'{len(polyhedra)}')

        for idx, faces in enumerate(polyhedra):

            if len(faces) < 4:
                raise ValueError(f'polyhedra {idx} must have at least 4 faces but has : '
                                 f'{len(faces)}')

            def any_vertex_out_of_bounds(face) -> bool:
                face = np.array(face)
                return np.any((face < 0) | (face >= vertices.shape[0]))

            if any(map(any_vertex_out_of_bounds, faces)):
                raise ValueError(f'faces of polyhedra {idx} must be list of lists of unique integers in range '
                                 f'[0, {vertices.shape[0]}) but is : '
                                 f'{faces}')

            def any_vertex_duplicated(face) -> bool:
                return len(set(face)) < len(face)

            if any(map(any_vertex_duplicated, faces)):
                raise ValueError(f'faces of polyhedra {idx} must be list of lists of unique integers but is : '
                                 f'{faces}')

            def is_face_not_polygon(face) -> bool:
                return len(face) < 3

            if any(map(is_face_not_polygon, faces)):
                raise ValueError(f'faces of polyhedra {idx} must contain faces of at least 3 vertices but is : '
                                 f'{faces}')

        self._polyhedra = polyhedra

        self._material = material

        self._bounds = np_readonly(np.min(vertices, axis=0)), \
                       np_readonly(np.max(vertices, axis=0))

    @beartype
    def transform(self, matrix: np.ndarray) -> 'Geometry':
        """
        Apply an affine matrix transformation to the vertices of this Geometry instance.

        :param matrix:
            Affine transformation matrix to apply.

        :return:
            Tensor the same shape as the vertices with the affine transformation applied.
        """

        if matrix.shape != (4, 4):
            raise ValueError(f'matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{matrix.shape}')

        if matrix.dtype != np.float32:
            raise TypeError(f'matrix must have dtype (float32) but is : '
                            f'{matrix.dtype}')

        if not is_scale_preserving(matrix):
            raise ValueError(f'direction_matrix must be an affine rotation matrix that preserves scale')

        return Geometry(vertices=transform_points(self.vertices, matrix),
                        polyhedra=self.polyhedra)

    @beartype
    def to_radia(self,
            vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]]) -> int:
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

        obj = []
        for polyhedra in self.polyhedra:

            # Find the vertices used by this polyhedra
            used_ids = sorted(list(set(vertex for face in polyhedra for vertex in face)))

            # Extract a minimal vertex list of just the points used by this polyhedra
            local_vertices = [self.vertices[global_id].tolist() for global_id in used_ids]

            # Create a lookup table mapping vertex ids in the full list to those in them minimal list
            map_ids  = { global_id: local_id for local_id, global_id in enumerate(used_ids) }

            # Create a face list using the minimal vertex list
            faces = [[(map_ids[global_id] + 1) for global_id in face] for face in polyhedra]

            # Create a radia object for this polyhedra
            obj += [rad.ObjPolyhdr(local_vertices, faces, vector.tolist())]

        obj = rad.ObjCnt(obj) if len(obj) > 1 else obj[0]

        mat = self.material.to_radia(vector)
        if mat is not None:
            obj = rad.MatApl(obj, mat)

        return obj

    @property
    @beartype
    def bounds(self) -> typ.Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    @property
    @beartype
    def vertices(self) -> jnp.ndarray:
        """
        Tensor of vertices in 3-space.
        """
        return np_readonly(self._vertices)

    @property
    @beartype
    def polyhedra(self) -> typ.Sequence[typ.Sequence[typ.Sequence[int]]]:
        """
        List of lists of lists of integer vertex IDs.
        """
        return self._polyhedra

    @property
    @beartype
    def material(self) -> Material:
        return self._material
