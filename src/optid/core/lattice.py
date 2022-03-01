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
import jax
import jax.numpy as jnp
import numpy as np

# Opt-ID Imports
from .affine import translate, scale, jnp_translate, jnp_scale


@jax.jit
def jnp_unit_limits(n):
    """
    Find the limits along an axis for a unit-cube centred at origin.

    If n = 1 then the only step is centred at 0.
    If n > 1 then steps are uniformly distributed from -0.5 to +0.5.

    :param n:
        Number of steps along the axis.

    :return:
        Min-max limits along the axis.
    """
    limit = (jnp.clip((n - 1), 0.0, 1.0) * 0.5).astype(jnp.float32)
    return -limit, limit


def unit_limits(n):
    """
    Find the limits along an axis for a unit-cube centred at origin.

    If n = 1 then the only step is centred at 0.
    If n > 1 then steps are uniformly distributed from -0.5 to +0.5.

    :param n:
        Number of steps along the axis.

    :return:
        Min-max limits along the axis.
    """
    limit = (np.clip((n - 1), 0.0, 1.0) * 0.5).astype(np.float32)
    return -limit, limit


@jax.partial(jax.jit, static_argnums=(0, 1, 2))
def jnp_unit_lattice(x, z, s):
    """
    Generate a 3-lattice of XZS coordinates across the unit-cube at origin.

    Singleton dimensions are centred at 0.
    Non-singleton dimensions are uniformly distributed from -0.5 to +0.5.

    Note: jax.jit needs static_argnums because x,z,s change the output size.

    :param x:
        Number of steps along the X-axis.

    :param z:
        Number of steps along the Z-axis.

    :param s:
        Number of steps along the S-axis.

    :return:
        Lattice of XZS coordinates uniformly distributed over the unit-cube at origin.
    """
    return jnp.stack(jnp.meshgrid(jnp.linspace(*jnp_unit_limits(x), x, dtype=jnp.float32),
                                  jnp.linspace(*jnp_unit_limits(z), z, dtype=jnp.float32),
                                  jnp.linspace(*jnp_unit_limits(s), s, dtype=jnp.float32),
                                  indexing='ij'), axis=-1)


def unit_lattice(x, z, s):
    """
    Generate a 3-lattice of XZS coordinates across the unit-cube at origin.

    Singleton dimensions are centred at 0.
    Non-singleton dimensions are uniformly distributed from -0.5 to +0.5.

    Note: jax.jit needs static_argnums because x,z,s change the output size.

    :param x:
        Number of steps along the X-axis.

    :param z:
        Number of steps along the Z-axis.

    :param s:
        Number of steps along the S-axis.

    :return:
        Lattice of XZS coordinates uniformly distributed over the unit-cube at origin.
    """
    return np.stack(np.meshgrid(np.linspace(*unit_limits(x), x, dtype=np.float32),
                                np.linspace(*unit_limits(z), z, dtype=np.float32),
                                np.linspace(*unit_limits(s), s, dtype=np.float32),
                                indexing='ij'), axis=-1)


@jax.jit
def jnp_unit_to_orthonormal_matrix(x, z, s):
    """
    Starting at a coordinate system centred at origin spanning -0.5 to +0.5 transform the coordinates
    to a coordinate system spanning from [0, x), [0, z), and [0, s).

    :param x:
        Size of the resulting span on the X-axis.

    :param z:
        Size of the resulting span on the Z-axis.

    :param s:
        Size of the resulting span on the S-axis.

    :return:
        Affine transformation matrix that converts between coordinate spaces.
    """
    return jnp_translate(0.5, 0.5, 0.5) @ \
           jnp_scale(jnp.maximum(1, (x - 1)),
                     jnp.maximum(1, (z - 1)),
                     jnp.maximum(1, (s - 1)))


def unit_to_orthonormal_matrix(x, z, s):
    """
    Starting at a coordinate system centred at origin spanning -0.5 to +0.5 transform the coordinates
    to a coordinate system spanning from [0, x), [0, z), and [0, s).

    :param x:
        Size of the resulting span on the X-axis.

    :param z:
        Size of the resulting span on the Z-axis.

    :param s:
        Size of the resulting span on the S-axis.

    :return:
        Affine transformation matrix that converts between coordinate spaces.
    """
    return translate(0.5, 0.5, 0.5) @ \
           scale(np.maximum(1, (x - 1)),
                 np.maximum(1, (z - 1)),
                 np.maximum(1, (s - 1)))


def any_unit_point_out_of_bounds(point_lattice, eps):
    """
    Test that all points in the point lattice are within the bounds of the unit lattice centred at origin spanning
    -0.5 to +0.5.

    :param point_lattice:
        Lattice of points in unit space.

    :param eps:
        Tolerance value for comparison.

    :return:
        True if any point lays outside the unit cube.
    """
    return np.any((point_lattice > +(0.5 + eps)) |
                  (point_lattice < -(0.5 + eps)))


def jnp_any_unit_point_out_of_bounds(point_lattice, eps):
    """
    Test that all points in the point lattice are within the bounds of the unit lattice centred at origin spanning
    -0.5 to +0.5.

    :param point_lattice:
        Lattice of points in unit space.

    :param eps:
        Tolerance value for comparison.

    :return:
        True if any point lays outside the unit cube.
    """
    return jnp.any((point_lattice > +(0.5 + eps)) |
                   (point_lattice < -(0.5 + eps)))


def any_orthonormal_point_out_of_bounds(point_lattice, x, z, s, eps):
    """
    Test that all points in the point lattice are within the bounds of the orthonormal lattice spanning from
    [0, x), [0, z), and [0, s).

    :param point_lattice:
        Lattice of points in orthonormal space.

    :param x:
        Size of the resulting span on the X-axis.

    :param z:
        Size of the resulting span on the Z-axis.

    :param s:
        Size of the resulting span on the S-axis.

    :param eps:
        Tolerance value for comparison.

    :return:
        True if any point lays outside the orthonormal bounds.
    """
    return np.any((point_lattice > (np.array([(x-1), (z-1), (s-1)], dtype=np.float32) + eps)) |
                  (point_lattice < -eps))


def jnp_any_orthonormal_point_out_of_bounds(point_lattice, x, z, s, eps):
    """
    Test that all points in the point lattice are within the bounds of the orthonormal lattice spanning from
    [0, x), [0, z), and [0, s).

    :param point_lattice:
        Lattice of points in orthonormal space.

    :param x:
        Size of the resulting span on the X-axis.

    :param z:
        Size of the resulting span on the Z-axis.

    :param s:
        Size of the resulting span on the S-axis.

    :param eps:
        Tolerance value for comparison.

    :return:
        True if any point lays outside the orthonormal bounds.
    """
    return jnp.any((point_lattice > (jnp.array([(x-1), (z-1), (s-1)], dtype=jnp.float32) + eps)) |
                   (point_lattice < -eps))


@jax.jit
def jnp_orthonormal_interpolate(value_lattice, point_lattice, order=1, mode='nearest', **kargs):
    """
    Perform an interpolated lookup of the values in the value lattice using the coordinates in the coordinate lattice.

    i.e. Assume:
        (point lattice : (5, 5, 3), value lattice : (2, 2, 2, 10, 10)) -> result : (5, 5, 10, 10)

        The trailing 3 in the shape of the point lattice means that the leading 3 dims of the value lattice will be
        used to perform Tri-(linear, quadratic, cubic) interpolation for each of the remaining dimensions.

        Varying the size of the coordinates in the point lattice we can see the effect on the result:

        Uni-(linear, quadratic, cubic) interpolation:
            (point lattice : (5, 5, 1), value lattice : (2, 2, 2, 10, 10)) -> result : (5, 5, 2, 2, 10, 10)

        Bi-(linear, quadratic, cubic) interpolation:
            (point lattice : (5, 5, 2), value lattice : (2, 2, 2, 10, 10)) -> result : (5, 5, 2, 10, 10)

        Tri-(linear, quadratic, cubic) interpolation:
            (point lattice : (5, 5, 3), value lattice : (2, 2, 2, 10, 10)) -> result : (5, 5, 10, 10)

        Quad-(linear, quadratic, cubic) interpolation:
            (point lattice : (5, 5, 4), value lattice : (2, 2, 2, 10, 10)) -> result : (5, 5, 10)


    :param value_lattice:
        Lattice of values to lookup from during interpolation.

        The coordinates of the values in the value lattice are assumed to be placed at their integer index locations
        in each axis. i.e.

            If the value lattice has shape (2, 3, 4, 4) where the first two dims are the spatial dims,
            then there are matrix values of shape (4, 4) defined at the following six coordinates in 2-space:
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), and (1, 2)

        If multiple value dims are present (i.e. a lattice of matrices) then each element of the multi-dim values are
        interpolated independently as scalars and stacked back together after interpolation.

        The number of value dims can be any size and will be the shape of the trailing dims in the result.

    :param point_lattice:
        Lattice of coordinates from which to sample interpolated values at from the values lattice.

        The last dim of the point lattice must be the size of the number of leading dims from the value lattice
        to interpolate over.

        The number of leading dims can be any size and will be the shape of the leading dims in the result.

    :param order:
        Integer order of the spline-interpolation.
        1 = linear (default), 2 = quadratic, 3 = cubic

    :param mode:
        Padding mode for interpolated values at coordinates outside of the value lattice.
        'reflect', 'constant', 'nearest' (default), 'mirror', 'wrap'

    :param kargs:
        All additional named parameters are forwarded as keyword arguments to jax.scipy.ndimage.map_coordinates.

    :return:
        Lattice containing the interpolated values at the coordinates from the point lattice.

        Shape of the result has the leading dims from the point lattice and the trailing dims from the value lattice.
    """

    # The point lattice last dimension holds vector coordinates into the leading dimensions of the value lattice
    p_shape    = point_lattice.shape[:-1]  # The result will have this shape for its leading dimensions
    p_channels = point_lattice.shape[-1]

    # Collapse multiple shape dims down to one dim, and transpose result into list of vectors
    point_lattice = jnp.reshape(point_lattice, (-1, p_channels)).T

    # The coordinates in the point lattice will index into the leading dimensions of the value lattice
    v_shape    = value_lattice.shape[:p_channels]
    v_channels = value_lattice.shape[p_channels:]  # The result will have this shape for its trailing dimensions

    # Collapse multiple channel dims to one dim
    value_lattice = jnp.reshape(value_lattice, (*v_shape, -1))

    def interp_channel(channel):
        return jax.scipy.ndimage.map_coordinates(channel, point_lattice, order=order, mode=mode, **kargs)

    # Interpolate each channel independently (in parallel) and stack the results together
    interp_values = jax.vmap(interp_channel, in_axes=-1, out_axes=-1)(value_lattice)

    # Reshape the results to have the spatial dims from the point lattice and the value dims from the value lattice
    return jnp.reshape(interp_values, (*p_shape, *v_channels))
