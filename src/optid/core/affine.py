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


@jax.jit
def jnp_transform_points(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of points in XZS.

    :param lattice:
        A tensor of points in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of points in XZS.
    """
    return (lattice @ matrix[:3, :3]) + matrix[-1, :3]


def transform_points(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of points in XZS.

    :param lattice:
        A tensor of points in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of points in XZS.
    """
    return (lattice @ matrix[:3, :3]) + matrix[-1, :3]


@jax.jit
def jnp_transform_vectors(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of vectors in XZS.

    :param lattice:
        A tensor of vectors in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of vectors in XZS.
    """
    return lattice @ jnp.linalg.inv(matrix[:3, :3]).T


def transform_vectors(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of vectors in XZS.

    :param lattice:
        A tensor of vectors in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of vectors in XZS.
    """
    return lattice @ np.linalg.inv(matrix[:3, :3]).T


@jax.jit
def jnp_transform_rescaled_vectors(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of vectors in XZS.

    :param lattice:
        A tensor of vectors in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of vectors in XZS.
    """
    # Transform the candidates field vector into world space at the magnet slot orientation
    norm = jnp.linalg.norm(lattice, axis=-1, keepdims=True)
    lattice = jnp_transform_vectors((lattice / norm), matrix)
    return (lattice / jnp.linalg.norm(lattice, axis=-1, keepdims=True)) * norm


def transform_rescaled_vectors(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of vectors in XZS.

    :param lattice:
        A tensor of vectors in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of vectors in XZS.
    """
    # Transform the candidates field vector into world space at the magnet slot orientation
    norm = np.linalg.norm(lattice, axis=-1, keepdims=True)
    lattice = transform_vectors((lattice / norm), matrix)
    return (lattice / np.linalg.norm(lattice, axis=-1, keepdims=True)) * norm


@jax.jit
def jnp_radians(degrees):
    """
    Convert degrees to radians.

    :param degrees:
        Angle in degrees.

    :return:
        Angle in radians.
    """
    return degrees * (jnp.pi / 180.0)


def radians(degrees):
    """
    Convert degrees to radians.

    :param degrees:
        Angle in degrees.

    :return:
        Angle in radians.
    """
    return degrees * (np.pi / 180.0)


@jax.jit
def jnp_rotate_x(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the X-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ 1,  0,  0,  0],
                      [ 0,  c,  s,  0],
                      [ 0, -s,  c,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32)


def rotate_x(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the X-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ 1,  0,  0,  0],
                     [ 0,  c,  s,  0],
                     [ 0, -s,  c,  0],
                     [ 0,  0,  0,  1]],
                    dtype=np.float32)


@jax.jit
def jnp_rotate_z(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the Z-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ c,  0, -s,  0],
                      [ 0,  1,  0,  0],
                      [ s,  0,  c,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32)


def rotate_z(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the Z-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  0, -s,  0],
                     [ 0,  1,  0,  0],
                     [ s,  0,  c,  0],
                     [ 0,  0,  0,  1]],
                    dtype=np.float32)


@jax.jit
def jnp_rotate_s(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the S-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ c,  s,  0,  0],
                      [-s,  c,  0,  0],
                      [ 0,  0,  1,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32)


def rotate_s(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the S-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s,  0,  0],
                     [-s,  c,  0,  0],
                     [ 0,  0,  1,  0],
                     [ 0,  0,  0,  1]],
                    dtype=np.float32)


@jax.jit
def jnp_scale(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal scale transformations.

    :param x:
        Scaling coefficient on the X-axis.

    :param z:
        Scaling coefficient on the Z-axis.

    :param s:
        Scaling coefficient on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return jnp.array([[x, 0, 0, 0],
                      [0, z, 0, 0],
                      [0, 0, s, 0],
                      [0, 0, 0, 1]],
                     dtype=jnp.float32)


def scale(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal scale transformations.

    :param x:
        Scaling coefficient on the X-axis.

    :param z:
        Scaling coefficient on the Z-axis.

    :param s:
        Scaling coefficient on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return np.array([[x, 0, 0, 0],
                     [0, z, 0, 0],
                     [0, 0, s, 0],
                     [0, 0, 0, 1]],
                    dtype=np.float32)


@jax.jit
def jnp_translate(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal translation transformations.

    :param x:
        Translation offset on the X-axis.

    :param z:
        Translation offset on the Z-axis.
        
    :param s:
        Translation offset on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [x, z, s, 1]],
                     dtype=jnp.float32)


def translate(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal translation transformations.

    :param x:
        Translation offset on the X-axis.

    :param z:
        Translation offset on the Z-axis.

    :param s:
        Translation offset on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [x, z, s, 1]],
                    dtype=np.float32)


@jax.jit
def jnp_is_scale_preserving(matrix):
    """
    Test if a matrix transformation preserves scale.

    :param matrix:
        Affine 4x4 matrix.

    :return:
        True if the distance between any two points remains the same after transformation.
    """
    return jnp.allclose(jnp.linalg.norm(matrix[:3, :3], axis=0), 1.0, atol=1e-5)


def is_scale_preserving(matrix):
    """
    Test if a matrix transformation preserves scale.

    :param matrix:
        Affine 4x4 matrix.

    :return:
        True if the distance between any two points remains the same after transformation.
    """
    return np.allclose(np.linalg.norm(matrix[:3, :3], axis=0), 1.0, atol=1e-5)
