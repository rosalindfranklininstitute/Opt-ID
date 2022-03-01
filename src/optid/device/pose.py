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


class Pose:

    @beartype
    def __init__(self,
            gap: numbers.Real,
            phase: numbers.Real):
        """
        Construct a Pose instance representing the gap and phase of a Device.

        :param gap:
            Real valued positive gap distance between upper and lower girder sets.

        :param phase:
            Real valued positive phasing distance for helical undulators.
        """

        if gap < 0:
            raise ValueError(f'gap must be >= 0 but is : '
                             f'{gap}')

        self._gap = gap

        if phase < 0:
            raise ValueError(f'phase must be >= 0 but is : '
                             f'{phase}')

        self._phase = phase

    def __repr__(self):
        return f'Pose(gap={self.gap}, phase={self.phase})'

    def __eq__(self, other):
        if isinstance(other, Pose):
            return (self.gap == other.gap) and (self.phase == other.phase)
        return False

    def __hash__(self):
        return hash((self.gap, self.phase))

    @property
    @beartype
    def gap(self) -> numbers.Real:
        return self._gap

    @property
    @beartype
    def phase(self) -> numbers.Real:
        return self._phase
