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

# Opt-ID Imports
from ..geometry import ExtrudedPolygon
from ..material import Material, DefaultMaterial


TCutTuple = typ.Tuple[numbers.Real, numbers.Real]
TCutCornerScalars = typ.Tuple[numbers.Real, numbers.Real, numbers.Real, numbers.Real]
TCutCornerTuples = typ.Tuple[TCutTuple, TCutTuple, TCutTuple, TCutTuple]
TCuts = typ.Union[np.ndarray, numbers.Real, TCutTuple, TCutCornerScalars, TCutCornerTuples]


class SquareCutCuboid(ExtrudedPolygon):

    @beartype
    def __init__(self,
            shape: typ.Union[np.ndarray, typ.Sequence[numbers.Real]],
            cuts: TCuts = 0,
            subdiv: numbers.Real = 0,
            material: Material = DefaultMaterial(),
            **tetgen_kargs):
        """
        Construct a SquareCutCuboid instance.

        :param shape:
            Aligned size vector in 3-space for the main cuboid.

        :param cuts:
            Size of the cut regions on each corner of the main cuboid in the XZ plane.

            Cuts can be specified as either:
                Scalar value to be applied in X and Z at every corner.
                Tuple (X, Z) to be applied at every corner.
                Tuple (BL, TL, TR, BR) to be applied in X and Z the four corners.
                Tuple ((BLx, BLz), (TLx, TLz), (TRx, TRz), (BRx, BRz)) to be applied at the four corners.

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

        if not isinstance(cuts, np.ndarray):
            cuts = np.array(cuts, dtype=np.float32)

        if np.any(cuts < 0):
            raise ValueError(f'cuts must be greater than or equal to zero but is : '
                             f'{cuts}')

        if cuts.shape == () or cuts.shape == (1,):
            # Common value for X and Z shared on all corners
            cuts = np.tile(np.reshape(cuts, (1, 1)), (4, 2))
        elif cuts.shape == (2,):
            # Separate value for X and Z shared on all corners
            cuts = np.tile(np.reshape(cuts, (1, 2)), (4, 1))
        elif cuts.shape == (4,):
            # Common value for X and Z separate on all corners
            cuts = np.tile(np.reshape(cuts, (4, 1)), (1, 2))

        if cuts.shape != (4, 2):
            # Separate value for X and Z separate on all corners
            raise ValueError(f'cuts must be coercible into shape (4, 2) but is : '
                             f'{cuts.shape}')

        # Which cut values are zeroed?
        cuts_zeros = (cuts == 0)

        if np.any(np.logical_xor(cuts_zeros[:, 0], cuts_zeros[:, 1])):
            raise ValueError(f'cuts cannot be zero unless it is zero for both X and Z components but is : '
                             f'{cuts}')

        # Cuts defined in BL TL TR BR order
        (blx, blz), (tlx, tlz), (trx, trz), (brx, brz) = cuts.tolist()

        if -(z - blz) >= (z - tlz):
            raise ValueError(f'cuts left edge top and bottom cuts collide : '
                             f'{cuts}')

        if -(x - tlx) >= (x - trx):
            raise ValueError(f'cuts top edge left and right cuts collide : '
                             f'{cuts}')

        if -(z - brz) >= (z - trz):
            raise ValueError(f'cuts right edge top and bottom cuts collide : '
                             f'{cuts}')

        if -(x - blx) >= (x - brx):
            raise ValueError(f'cuts bottom edge left and right cuts collide : '
                             f'{cuts}')

        # Which corners have a non zero cuts?
        bl, tl, tr, br = (~np.logical_and(cuts_zeros[:, 0], cuts_zeros[:, 1])).tolist()

        polygon = np.array([
            *([[-(x - blx), -z], [-(x - blx), -(z - blz)], [-x, -(z - blz)]] if bl else [[-x, -z]]),
            *([[-x,  (z - tlz)], [-(x - tlx),  (z - tlz)], [-(x - tlx),  z]] if tl else [[-x,  z]]),
            *([[ (x - trx),  z], [ (x - trx),  (z - trz)], [ x,  (z - trz)]] if tr else [[ x,  z]]),
            *([[ x, -(z - brz)], [ (x - brx), -(z - brz)], [ (x - brx), -z]] if br else [[ x, -z]])],
            dtype=np.float32)

        super().__init__(polygon=polygon, thickness=s, subdiv=subdiv, material=material, **tetgen_kargs)
