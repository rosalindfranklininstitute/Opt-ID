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
from sect.triangulation import constrained_delaunay_triangles

# Opt-ID Imports
from ..geometry import TetrahedralMesh
from ..material import Material, DefaultMaterial


class ExtrudedPolygon(TetrahedralMesh):

    @beartype
    def __init__(self,
            polygon: typ.Union[np.ndarray, typ.Sequence[typ.Sequence[numbers.Real]]],
            thickness: numbers.Real,
            subdiv: numbers.Real = 0,
            material: Material = DefaultMaterial(),
            **tetgen_kargs):
        """
        Construct an ExtrudedPolygon instance from a set of unique polygon vertices in 2-space and a thickness.

        :param polygon:
            Tensor of vertices in 2-space of shape (N, 2).

        :param thickness:
            Thickness of the geometry along the S-axis.

        :param subdiv:
            Scale for introducing new vertices to aid in subdivision. Ignored if less than or equal to zero.

        :param **tetgen_kargs:
            All additional parameters are forwarded to TetGen's tetrahedralize function.
        """

        if not isinstance(polygon, np.ndarray):

            def is_vertex_not_2d(vertex) -> bool:
                return len(vertex) != 2

            if any(map(is_vertex_not_2d, polygon)):
                raise ValueError(f'polygon must be a list of 2D XZ coordinates but is : '
                                 f'{polygon}')

            polygon = np.array(polygon, dtype=np.float32)

        if polygon.shape[-1] != 2:
            raise ValueError(f'polygon must be a list of 2D XZ coordinates (N, 2) but is : '
                             f'{polygon.shape}')

        if polygon.shape[0] < 3:
            raise ValueError(f'polygon must be a list of at least 3 2D XZ coordinates (N >= 3, 2) but is : '
                             f'{polygon.shape}')

        if polygon.dtype != np.float32:
            raise TypeError(f'polygon must have dtype (float32) but is : '
                            f'{polygon.dtype}')

        thickness = float(thickness)

        if thickness <= 0:
            raise TypeError(f'thickness must a positive float but is : '
                            f'{thickness}')

        # Introduce new vertices into end polygon loop
        def subdiv_edge(p0, p1):
            dist = np.sqrt(np.sum(np.square(p1 - p0)))
            segments = 2 if subdiv <= 0 else max(2, int(np.ceil(dist / subdiv)))
            return np.stack([np.linspace(p0[0], p1[0], segments)[:-1],
                             np.linspace(p0[1], p1[1], segments)[:-1]], axis=1)

        cap_polygon = np.concatenate([polygon, polygon[:1]], axis=0)
        cap_polygon = np.concatenate([subdiv_edge(p0, p1) for p0, p1 in zip(cap_polygon[:-1], cap_polygon[1:])], axis=0)
        n = cap_polygon.shape[0]

        # Naive mesh
        vertex_to_id = { (*vertex,): idx for idx, vertex in enumerate(cap_polygon) }
        cap_mesh_polygons = [[vertex_to_id[vertex] for vertex in face]
                             for face in constrained_delaunay_triangles(list(vertex_to_id.keys()))]

        # Project end polygon 2D vertices into 3D vertices for each slice
        s_limit = thickness * 0.5
        s_slices = 2 if subdiv <= 0 else max(2, int(np.ceil(thickness / subdiv)))
        vertices = np.concatenate([
            np.pad(cap_polygon, ((0, 0), (0, 1)), constant_values=s)
            for s in np.linspace(-s_limit, +s_limit, s_slices)])

        faces = [*cap_mesh_polygons]
        s = (n * (s_slices - 1))

        for a, b, c in cap_mesh_polygons:
            faces += [[(a + s), (c + s), (b + s)]]

        # Edge polygons
        for v0 in range(n):
            v1 = (v0 + 1) % n
            for s in range(s_slices - 1):
                # D A
                # C B
                a, b, c, d = (v1 + (n * s)), (v0 + (n * s)), (v0 + (n * (s + 1))), (v1 + (n * (s + 1)))
                faces += [[a, b, c], [a, c, d]]

        faces = np.array(faces, dtype=np.int32)
        vertices = np.array(vertices, dtype=np.float32)

        super().__init__(vertices=vertices, faces=faces, material=material, **tetgen_kargs)

