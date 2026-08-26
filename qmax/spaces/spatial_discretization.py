from typing import ClassVar, Callable, Optional
from abc import abstractmethod
from functools import partial

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import ScalarLike, Array, ArrayLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator, AbstractHermitianOperator


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
    endpoint: eqx.AbstractClassVar[bool]

    # To avoid array valued fields which trigger a warning on static hilbert_spaces
    # TODO: I feel like there should be a better workaround for this...
    x0: tuple[float, ...] = eqx.field(converter=_to_tuple)
    xf: tuple[float, ...] = eqx.field(converter=_to_tuple)
    mesh_size: tuple[int, ...] = eqx.field(converter=partial(_to_tuple, dtype=int))

    @abstractmethod
    def from_values(self, values: ArrayLike) -> SpatiallyDiscretizedState:
        pass

    def from_function(self, fn: Callable[[ArrayLike], ScalarLike]) -> SpatiallyDiscretizedState:
        return self.from_values(self.eval(fn))

    def innerp(self, 
        y1: SpatiallyDiscretizedState, 
        y2: SpatiallyDiscretizedState) -> ScalarLike:

        return jnp.prod(self.dx_range) * jnp.sum(
            jnp.conj(y1.values) * y2.values, axis=self.spatial_axes)

    @property
    def spatial_dim(self) -> int:
        return len(self.mesh_size)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.spatial_dim, 0))

    def flatten(self, arr_grid: ArrayLike):
        return arr_grid.reshape(*arr_grid.shape[:-self.spatial_dim], -1)

    def to_grid(self, arr: ArrayLike, sizes: Optional[tuple[int, ...]]=None):
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
            jnp.linspace(self.x0[i], self.xf[i], self.mesh_size[i], endpoint=self.endpoint) 
            for i in range(self.spatial_dim)
        ]

    @property
    def x_range(self) -> Array:
        if self.spatial_dim == 1:
            return jnp.linspace(self.x0[0], self.xf[0], self.mesh_size[0], endpoint=self.endpoint) 
        raise Exception(f"x_range only supported on 1d spatial discretizations but dim={self.dim}, did you mean x_ranges?")

    @property
    def x_meshgrid(self) -> Array:
        return jnp.meshgrid(*self.x_ranges, indexing="ij")

    @property
    def points(self) -> Array:   
        if self.spatial_dim == 1:
            return self.x_range

        return self.grid_vectors(self.x_ranges)
 
    def eval(self, fn: Callable[[ArrayLike], ScalarLike]) -> Array:
        return self.to_grid(jax.vmap(fn)(self.points))

    @property
    def dx_range(self) -> Array:
        if self.endpoint:
            sizes = jnp.array(self.mesh_size) - 1
        else:
            sizes = jnp.array(self.mesh_size)

        return (jnp.array(self.xf) - jnp.array(self.x0)) / sizes

    # These factories are not abstract properties since intermediate classes 
    # like _FiniteDifference1D need to be instantiable without overrides

    def laplacian(self) -> Operator:
        raise NotImplementedError 

    def potential_energy(self, potential: Callable[[ArrayLike], ScalarLike]) -> Operator:
        raise NotImplementedError 

    def position(self, axis: int = 0) -> Operator:
        if self.spatial_dim == 1:
            return self.potential_energy(lambda x: x)
        return self.potential_energy(lambda x, i=axis: x[i])

    def momentum(self, axis: int) -> Operator:
        raise NotImplementedError 


class AbstractPotentialEnergy(AbstractHermitianOperator):
    potential: Callable[[ArrayLike], ScalarLike]

    @property
    def values(self) -> Array:
        return self.domain.eval(self.potential)

    def action(self, y: SpatiallyDiscretizedState) -> SpatiallyDiscretizedState:
        return self.domain.from_values(self.values * y.values)

    def exp_action(self, h: ScalarLike, y: SpatiallyDiscretizedState) -> SpatiallyDiscretizedState:
        return self.domain.from_values(jnp.exp(h * self.values) * y.values)

    def _solve(
        self,
        b: SpatiallyDiscretizedState,
        scale: ScalarLike=-1.0,
        shift: ScalarLike=0.0) -> SpatiallyDiscretizedState:

        return self.domain.from_values(b.values / (scale * self.values + shift))

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([jnp.min(self.values), jnp.max(self.values)])
