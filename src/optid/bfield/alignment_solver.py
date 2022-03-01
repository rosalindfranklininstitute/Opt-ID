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
import jax
import jax.numpy as jnp

# Opt-ID Imports
from ..core.utils import yield_gridsearch
from ..core.affine import jnp_transform_points, scale, rotate_s, radians, rotate_x, rotate_z, translate, \
                          jnp_scale, jnp_rotate_s, jnp_rotate_x, jnp_rotate_z, jnp_radians, jnp_translate
from ..core.lattice import jnp_unit_lattice
from ..core.bfield import jnp_interpolate_bfield
from .bfield import Bfield
from ..lattice import Lattice


@beartype
def argmin_alignment_solver(
        bfield: Bfield,
        observation: jnp.ndarray,
        loss_fn: typ.Callable,
        matrices: typ.Iterable[np.ndarray]) -> typ.Tuple[numbers.Real, Lattice]:

    if (observation.ndim != 4) or (observation.shape[3:] != (3,)):
        raise ValueError(f'observation must be have shape (X,Z,S,3) but is : '
                         f'{observation.shape[3:]}')

    if observation.dtype != jnp.float32:
        raise TypeError(f'observation must have dtype (float32) but is : '
                        f'{observation.dtype}')

    min_loss, min_arg = None, None
    for matrix in matrices:

        if matrix.shape != (4, 4):
            raise ValueError(f'matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{matrix.shape}')

        if matrix.dtype != np.float32:
            raise TypeError(f'matrix must have dtype (float32) but is : '
                            f'{matrix.dtype}')

        point_lattice = jnp_transform_points(jnp_unit_lattice(*observation.shape[:3]), matrix)
        interp_field  = jnp_interpolate_bfield(bfield.field, point_lattice, bfield.lattice.world_to_orthonormal_matrix)

        loss = float(loss_fn(observation, interp_field))
        if (min_loss is None) or (loss < min_loss):
            min_loss, min_arg = loss, matrix

    return min_loss, Lattice(min_arg, shape=observation.shape[:3])


@beartype
def parameterized_matrix(
            sx: numbers.Real = 1, sz: numbers.Real = 1, ss: numbers.Real = 1,
            rx: numbers.Real = 0, rz: numbers.Real = 0, rs: numbers.Real = 0,
            tx: numbers.Real = 0, tz: numbers.Real = 0, ts: numbers.Real = 0) -> np.ndarray:

    return scale(sx, sz, ss) @ \
           rotate_s(radians(rs)) @ rotate_x(radians(rx)) @ rotate_z(radians(rz)) @ \
           translate(tx, tz, ts)


@jax.jit
def jnp_parameterized_matrix(sx, sz, ss, rx, rz, rs, tx, tz, ts):

    return jnp_scale(sx, sz, ss) @ \
           jnp_rotate_s(jnp_radians(rs)) @ jnp_rotate_x(jnp_radians(rx)) @ jnp_rotate_z(jnp_radians(rz)) @ \
           jnp_translate(tx, tz, ts)


@beartype
def gridsearch_alignment_solver(
        bfield: Bfield,
        observation: jnp.ndarray,
        loss_fn: typ.Callable,
        params: typ.Dict[str, typ.Sequence[numbers.Real]]) -> typ.Tuple[numbers.Real, Lattice]:

    matrices = map((lambda p: parameterized_matrix(**p)), yield_gridsearch(params))

    return argmin_alignment_solver(bfield=bfield, observation=observation,
                                   loss_fn=loss_fn, matrices=matrices)


@beartype
def sgd_alignment_solver(
        bfield: Bfield,
        observation: jnp.ndarray,
        loss_fn: typ.Callable,
        params: typ.Dict[str, numbers.Real],
        mask: typ.Dict[str, numbers.Real],
        epochs: int = 8,
        steps: int = 8,
        learing_rate: numbers.Real = 0.1) -> typ.Tuple[numbers.Real, Lattice]:

    if (observation.ndim != 4) or (observation.shape[3:] != (3,)):
        raise ValueError(f'observation must be have shape (X,Z,S,3) but is : '
                         f'{observation.shape[3:]}')

    if observation.dtype != jnp.float32:
        raise TypeError(f'observation must have dtype (float32) but is : '
                        f'{observation.dtype}')

    default_params = dict(sx=1, sz=1, ss=1, rx=0, rz=0, rs=0, tx=0, tz=0, ts=0)
    default_params.update(params)
    default_mask   = dict(sx=0, sz=0, ss=0, rx=0, rz=0, rs=0, tx=0, tz=0, ts=0)
    default_mask.update(mask)
    param_names = ['sx', 'sz', 'ss', 'rx', 'rz', 'rs', 'tx', 'tz', 'ts']
    params = [float(default_params[name]) for name in param_names]
    mask   = [float(default_mask[name])   for name in param_names]

    @jax.jit
    def params_loss_fn(params):
        matrix        = jnp_parameterized_matrix(*params)
        point_lattice = jnp_transform_points(jnp_unit_lattice(*observation.shape[:3]), matrix)
        interp_field  = jnp_interpolate_bfield(bfield.field, point_lattice, bfield.lattice.world_to_orthonormal_matrix)

        return loss_fn(observation, interp_field)

    params_loss_grad_fn = jax.value_and_grad(params_loss_fn)

    @jax.jit
    def sgd_fn(param, update, mask):
        return param - ((learing_rate * mask) * update)

    @jax.jit
    def update_step(params):
        loss, grads = params_loss_grad_fn(params)
        params = jax.tree_multimap(sgd_fn, params, grads, mask)
        return loss, params

    for epoch in range(epochs):
        for step in range(steps):

            loss, params = update_step(params)

        print(f'epoch {epoch:-6d} loss {float(loss)}')

    params = { name: float(param) for name, param in zip(param_names, params) }

    return float(loss), Lattice(parameterized_matrix(**params), shape=observation.shape[:3])
