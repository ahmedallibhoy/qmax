from typing import ClassVar, Callable
from abc import abstractmethod
from functools import partial

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Scalar, ScalarLike, Array, ArrayLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState


class SpatialState(AbstractState):

    @property
    @abstractmethod
    def values(self) -> Array:
        pass


class SpatialHilbertSpace(AbstractHilbertSpace):
    state_type: eqx.AbstractClassVar[type[SpatialState]]

    @abstractmethod
    def from_values(self, values: ArrayLike) -> SpatiallyDiscretizedState:
        pass

    @property
    @abstractmethod
    def spatial_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def mesh_size(self) -> tuple[int]:
        pass

    @abstractmethod
    def flatten(self, arr_grid):
        pass 

    @abstractmethod
    def to_grid(self, arr, sizes=None):
        pass

    @property
    @abstractmethod
    def x_ranges(self) -> Array:
        pass

    @staticmethod
    def grid_vectors(per_axis: list[Array]) -> Array:
        """
        Given a list of scalars for each axis, returns all vectors in the 
        cartesian product of the lists. 
        """
        grid = jnp.stack(jnp.meshgrid(*per_axis, indexing="ij"), axis=-1) 
        return grid.reshape(-1, len(per_axis))

    @property
    def x_meshgrid(self) -> Array:
        return jnp.meshgrid(*self.x_ranges, indexing="ij")

    @property
    def points(self) -> Array:   
        return self.grid_vectors(self.x_ranges)
 
    def eval(self, fn: Callable[[Array], ScalarLike]) -> Array:
        return self.to_grid(jax.vmap(fn)(self.points))

    @property
    def dx_range(self) -> Array:
        return jnp.array([x_range[1] - x_range[0] for x_range in self.x_ranges])


class SpatiallyDiscretizedState1D(AbstractState):
    pass


class SpatialDiscretization1D(SpatialHilbertSpace):
    state_type: eqx.AbstractClassVar[type[SpatiallyDiscretizedState1D]]
    x0: float
    xf: float
    _mesh_size: int

    @property
    def spatial_dim(self) -> int:
        return 1

    @property
    def mesh_size(self) -> tuple[int]:
        return (self._mesh_size,)

    def flatten(self, arr_grid: ArrayLike) -> Array:
        return arr_grid

    def to_grid(self, arr: ArrayLike, sizes=None) -> Array:
        return arr

    @property
    def x_ranges(self) -> Array:
        return [self.x_range]

    @property
    def x_range(self) -> Array:
        return jnp.linspace(self.x0, self.xf, self._mesh_size, endpoint=self.endpoint)

    @property
    def points(self) -> Array:   
        return self.x_range

    @property
    def dx(self) -> Scalar:
        return (self.xf - self.x0) / self._mesh_size


