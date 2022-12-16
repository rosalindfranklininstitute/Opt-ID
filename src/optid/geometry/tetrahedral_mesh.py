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
import tetgen

# Opt-ID Imports
from ..geometry import Geometry
from ..material import Material, DefaultMaterial


class TetrahedralMesh(Geometry):

    @beartype
    def __init__(self,
            vertices: typ.Union[np.ndarray, typ.Sequence[typ.Sequence[numbers.Real]]],
            faces: typ.Union[np.ndarray, typ.Sequence[typ.Sequence[int]]],
            material: Material = DefaultMaterial(),
            **tetgen_kargs):
        """
        Construct a tetrahedralized mesh from an input triangular surface mesh.

        :param vertices:
            Tensor of vertices in 3-space of shape (N, 3).

        :param faces:
            Tensor of integer IDs of triangular faces with shape (N, 3).

        :param **tetgen_kargs:
            All additional parameters are forwarded to TetGen's tetrahedralize function.
        """

        if not isinstance(vertices, np.ndarray):

            def is_vertex_not_3d(vertex) -> bool:
                return len(vertex) != 3

            if any(map(is_vertex_not_3d, vertices)):
                raise ValueError(f'vertices must be a list of 3D XZS coordinates but is : '
                                 f'{vertices}')

            vertices = np.array(vertices, dtype=np.float32)

        if vertices.shape[-1] != 3:
            raise ValueError(f'vertices must be a list of 3D XZS coordinates (N, 3) but is : '
                             f'{vertices.shape}')

        if vertices.shape[0] < 4:
            raise ValueError(f'vertices must be a list of at least 4 3D XZS coordinates (N >= 4, 3) but is : '
                             f'{vertices.shape}')

        if vertices.dtype != np.float32:
            raise TypeError(f'vertices must have dtype (float32) but is : '
                            f'{vertices.dtype}')

        if not isinstance(faces, np.ndarray):

            def is_face_not_triangle(face) -> bool:
                return len(face) != 3

            if any(map(is_face_not_triangle, faces)):
                raise ValueError(f'faces must be a list of 3 integer face IDs but is : '
                                 f'{faces}')

            faces = np.array(faces, dtype=np.int32)

        if faces.shape[-1] != 3:
            raise ValueError(f'faces must be a list of 3 ID triangles (M, 3) but is : '
                             f'{faces.shape}')

        if faces.shape[0] < 4:
            raise ValueError(f'faces must be a list of at least 4 faces (M >= 4, 3) but is : '
                             f'{faces.shape}')

        if faces.dtype != np.int32:
            raise TypeError(f'faces must have dtype (int32) but is : '
                            f'{faces.dtype}')

        kargs = dict(mindihedral=20, minratio=1.5, nobisect=False)
        kargs.update(tetgen_kargs)
        tet = tetgen.TetGen(np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32))
        tet.tetrahedralize(order=1, **kargs)

        vertices = np.array(tet.grid.points.astype(np.float32))

        tetrahedron_faces = [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
        polyhedra = [[[int(cell[vertex]) for vertex in face] for face in tetrahedron_faces]
                     for cell in tet.grid.cells.reshape(-1, 5)[:, 1:]]

        super().__init__(vertices=vertices, polyhedra=polyhedra, material=material)

