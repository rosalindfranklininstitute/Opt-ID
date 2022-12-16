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
from types import MappingProxyType
import beartype.typing as typ
import numpy as np
import pandas as pd


# Opt-ID Imports
from ..constants import MATRIX_IDENTITY
from ..core.affine import is_scale_preserving
from ..core.utils import np_readonly
from ..geometry import Geometry
from .element_candidate import ElementCandidate


class ElementSet:

    @beartype
    def __init__(self,
            name: str,
            geometry: Geometry,
            vector: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            candidates: typ.Optional[typ.Union[str, pd.DataFrame, typ.Sequence[ElementCandidate]]] = None,
            flip_matrices: typ.Optional[typ.Union[np.ndarray, typ.Sequence[np.ndarray]]] = None,
            rescale_vector: bool = True):

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        self._geometry = geometry

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        self._vector = vector

        if candidates is None:
            candidates = []

        if isinstance(candidates, str):
            candidates = pd.read_csv(candidates)

        if isinstance(candidates, pd.DataFrame):
            candidates = ElementCandidate.from_dataframe(candidates)

        if not isinstance(candidates, list):
            candidates = list(candidates)

        self._candidates = { candidate.name: candidate for candidate in candidates }

        if (len(candidates) > 0) and rescale_vector:
            mean_candidate = np.mean([np.linalg.norm(candidate.vector)
                                      for candidate in self.candidates.values()])

            self._vector = (self._vector / np.linalg.norm(self._vector)) * mean_candidate

        flip_matrices = [] if (flip_matrices is None) else [matrix for matrix in flip_matrices]

        if not all(map(is_scale_preserving, flip_matrices)):
            raise ValueError(f'flip_matrices must all be scale preserving affine matrices')

        self._flip_matrices = np.array([MATRIX_IDENTITY] + flip_matrices, dtype=np.float32)

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    @beartype
    def vector(self) -> np.ndarray:
        return np_readonly(self._vector)

    @beartype
    def flip_matrix(self, flip: int) -> np.ndarray:

        if (flip < 0) or (flip >= self.nflip):
            raise ValueError(f'flip must be in range [0, {self.nflip}) but is : '
                             f'{flip}')

        return np_readonly(self._flip_matrices[flip])

    @property
    @beartype
    def nflip(self) -> int:
        return len(self._flip_matrices)

    @property
    @beartype
    def candidates(self) -> typ.Mapping[str, ElementCandidate]:
        return MappingProxyType(self._candidates)

    @property
    @beartype
    def ncandidate(self) -> int:
        return len(self._candidates)

    @property
    @beartype
    def is_magnetized(self) -> bool:
        return bool(np.linalg.norm(self.vector) > 1e-5)
