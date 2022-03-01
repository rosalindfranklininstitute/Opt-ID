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
from contextlib import contextmanager
import typing as typ
from beartype import beartype
import jax
import jax.numpy as jnp
import numpy as np
import radia as rad

# Opt-ID Imports
from .lattice import jnp_orthonormal_interpolate
from .affine import jnp_transform_points, jnp_transform_rescaled_vectors


@beartype
def radia_evaluate_bfield_on_lattice(
        radia_object: int,
        lattice: np.ndarray) -> np.ndarray:
    """
    Wraps rad.Fld to take numpy tensors as inputs and return them as outputs.

    :param radia_object:
        Handle to the radia object to simulate the field of.

    :param lattice:
        Tensor representing 3-space world coordinates to evaluate the field at.

    :return:
        Tensor of 3-space field vectors at each location in the lattice.
    """

    if lattice.shape[-1] != 3:
        raise ValueError(f'lattice must be a mesh of vectors in 3-space with shape (..., 3) but is : '
                         f'{lattice.shape}')

    if lattice.dtype != np.float32:
        raise TypeError(f'lattice must have dtype (float32) but is : '
                        f'{lattice.dtype}')

    return np.array(rad.Fld(radia_object, 'b', lattice.reshape((-1, 3)).tolist()),
                    dtype=np.float32).reshape(lattice.shape)


def jnp_radia_evaluate_bfield_on_lattice(
        radia_object: int,
        lattice: np.ndarray,
        device: typ.Optional[typ.Any] = None) -> jnp.ndarray:
    """
    Wraps rad.Fld to take numpy tensors as inputs and return JAX tensors as outputs.

    :param radia_object:
        Handle to the radia object to simulate the field of.

    :param lattice:
        Tensor representing 3-space world coordinates to evaluate the field at.

    :param device:
        JAX Device to place result on.

    :return:
        Tensor of 3-space field vectors at each location in the lattice.
    """

    return jax.device_put(radia_evaluate_bfield_on_lattice(radia_object=radia_object, lattice=lattice), device=device)


@contextmanager
def RadiaCleanEnv():
    # TODO consider attempting to virtualize scopes and deleting individual objects as needed
    rad.UtiDelAll()
    try:
        yield
    finally:
        rad.UtiDelAll()


@jax.jit
def jnp_interpolate_bfield(value_lattice, point_lattice, world_to_orthonormal_matrix):

    ortho_value_lattice = jnp_transform_rescaled_vectors(value_lattice, world_to_orthonormal_matrix)
    ortho_point_lattice = jnp_transform_points(point_lattice, world_to_orthonormal_matrix)
    ortho_interp_value_lattice = jnp_orthonormal_interpolate(ortho_value_lattice, ortho_point_lattice)

    return jnp_transform_rescaled_vectors(ortho_interp_value_lattice, jnp.linalg.inv(world_to_orthonormal_matrix))


@jax.jit
def jnp_bfield_from_lookup(lookup, vector):
    """
    Compute the bfield from a magnet with the given field vector using a lookup table of field rotation matrices.

    :param lookup:
        Lattice of 3x3 rotation matrices representing field curvature and scale over a spatial lattice.

        Magnet shape and geometry is baked into the lookup table, but the actual magnetization direction can be applied
        by a simple matmul of the desired field vector against the matrix at each location on the lattice, yielding
        a lattice of field 3-vectors.

    :param vector:
        Field vector for the magnet whose field we want to solve for.

    :return:
        Lattice of 3-vectors representing the field direction and magnitude at each spatial location represented on
        the lattice of the lookup table.
    """
    return lookup @ vector


@jax.jit
def jnp_integrate_trajectories(bfield, s_step, energy):

    # Unknown constant... evaluates to 1e-4 for 3 GeV storage ring
    const = (0.03 / energy) * 1e-2

    # Trapezium rule applied to bfield measurements in X and Z helps compute the second integral of motion
    trap_bfield = jnp.roll(bfield[..., :2], shift=1, axis=2)
    trap_bfield = trap_bfield.at[..., 0, :].set(0)  # Set first samples on S axis to 0
    trap_bfield = (trap_bfield + bfield[..., :2]) * (s_step * 0.5)

    # Accumulate the second integral of motion w.r.t the X and Z axes, along the orbital S axis
    traj_2nd_integral = jnp.cumsum((trap_bfield * const), axis=2)[..., ::-1] * jnp.array([-1, 1])

    # Trapezium rule applied to second integral of motion helps compute the first integral of motion
    trap_traj_2nd_integral = jnp.roll(traj_2nd_integral, shift=1, axis=2)
    trap_traj_2nd_integral = trap_traj_2nd_integral.at[:, :, 0, :].set(0)  # Set first samples on S axis to 0
    trap_traj_2nd_integral = (trap_traj_2nd_integral + traj_2nd_integral) * (s_step * 0.5)

    # Accumulate the first integral of motion w.r.t the X and Z axes, along the orbital S axis
    traj_1st_integral = jnp.cumsum(trap_traj_2nd_integral, axis=2)

    return traj_1st_integral, traj_2nd_integral
