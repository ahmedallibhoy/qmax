from typing import ClassVar, Callable
from abc import abstractmethod
from functools import partial

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array, ArrayLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState


def _to_tuple(x, dtype=float):
    if jnp.isscalar(x):
        return (x,)
    return tuple(dtype(s) for s in x)


class SpatiallyDiscretizedState(AbstractState):

    @property
    @abstractmethod
    def values(self) -> Array:
        pass


class SpatialDiscretization(AbstractHilbertSpace):
    state_type: eqx.AbstractClassVar[type[SpatiallyDiscretizedState]]

    # To avoid array valued fields which trigger a warning on static hilbert_spaces
    x0: tuple[float, ...] = eqx.field(converter=_to_tuple)
    xf: tuple[float, ...] = eqx.field(converter=_to_tuple)
    mesh_size: tuple[int, ...] = eqx.field(converter=partial(_to_tuple, dtype=int))

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def from_values(self, values: ArrayLike) -> SpatiallyDiscretizedState:
        pass

    @property
    def spatial_dim(self) -> int:
        return len(self.mesh_size)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.spatial_dim, 0))

    def flatten(self, arr_grid):
        return arr_grid.reshape(*arr_grid.shape[:-self.spatial_dim], -1)

    def to_grid(self, arr, sizes=None):
        if sizes is None:
            sizes = self.mesh_size
        return arr.reshape(*arr.shape[:-1], *sizes)

    @staticmethod
    def grid_vectors(per_axis: list[Array]) -> Array:
        """
        Given a list of scalars for each axis, returns all vectors in the 
        cartesian product of the lists. 
        """
        grid = jnp.stack(jnp.meshgrid(*per_axis, indexing="ij"), axis=-1) 
        return grid.reshape(-1, len(per_axis))

    @property
    def x_ranges(self) -> Array:
        return [
            jnp.linspace(self.x0[i], self.xf[i], self.mesh_size[i], endpoint=False) 
            for i in range(self.spatial_dim)
        ]

    @property
    def x_range(self) -> Array:
        if self.spatial_dim == 1:
            return jnp.linspace(self.x0[0], self.xf[0], self.mesh_size[0], endpoint=False) 
        raise Exception(f"x_range only supported on 1d spatial discretizations but dim={self.dim}, did you mean x_ranges?")

    @property
    def x_meshgrid(self) -> Array:
        return jnp.meshgrid(*self.x_ranges, indexing="ij")

    @property
    def points(self) -> Array:   
        if self.spatial_dim == 1:
            return self.x_range

        return self.grid_vectors(self.x_ranges)
 
    def eval(self, fn: Callable[[Array], ScalarLike]) -> Array:
        return self.to_grid(jax.vmap(fn)(self.points))

    @property
    def dx_range(self) -> Array:
        return (jnp.array(self.xf) - jnp.array(self.x0)) / jnp.array(self.mesh_size)


