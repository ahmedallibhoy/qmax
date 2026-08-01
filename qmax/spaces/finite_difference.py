from typing import Callable, ClassVar

from functools import reduce

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator
from ..exponentiators import AbstractExponentiator, ExactExponentiator, CrankNicolson
from .spatial_discretization import SpatialDiscretization, SpatiallyDiscretizedState


class FiniteDifferenceState(SpatiallyDiscretizedState):

    @property
    def values(self) -> Array:
        return self.hilbert_space.to_grid(self.coeffs)


class FiniteDifference(SpatialDiscretization):
    state_type: ClassVar = FiniteDifferenceState

    @property
    def dim(self) -> Array:
        return reduce(lambda a, b: a * b, self.mesh_size)

    def from_values(self, values: ArrayLike) -> FiniteDifferenceState:
        return FiniteDifferenceState(self.flatten(values), self)


class FiniteDifferenceLaplacian(Operator):
    domain: ClassVar[type[AbstractHilbertSpace]] = FiniteDifference
    exponentiator: AbstractExponentiator = eqx.field(default=CrankNicolson(), kw_only=True)

    def action(self, y: FiniteDifferenceState) -> FiniteDifferenceState:
        hilbert_space = y.hilbert_space
        vals = y.values
        lapl_vals = jnp.zeros_like(vals)

        for idx, dx in zip(hilbert_space.spatial_axes, hilbert_space.dx_range):
            vals_next = jax.lax.slice_in_dim(vals, 1, vals.shape[idx], axis=idx)
            vals_prev = jax.lax.slice_in_dim(vals, 0, vals.shape[idx] - 1, axis=idx)
            
            z_shape = list(vals.shape)
            z_shape[idx] = 1

            vals_next = jnp.concatenate((vals_next, jnp.zeros(z_shape)), axis=idx)   
            vals_prev = jnp.concatenate((jnp.zeros(z_shape), vals_prev), axis=idx)  
            lapl_vals += (vals_next - 2 * vals + vals_prev) / (dx ** 2)

        return hilbert_space.from_values(lapl_vals)

    def solve(
        self, 
        b: FiniteDifferenceState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> FiniteDifferenceState:

        hilbert_space = b.hilbert_space
        
        if hilbert_space.spatial_dim == 1:
            dim = hilbert_space.dim
            dx = hilbert_space.dx_range[0]

            diag = jnp.full(dim, shift - 2 * scale / (dx ** 2), dtype=complex)
            lower_diag = scale * jnp.full(dim - 1, 1 / (dx ** 2), dtype=complex)
            upper_diag = scale * jnp.full(dim - 1, 1 / (dx ** 2), dtype=complex)

            lx_op = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
            lx_op = lx.TaggedLinearOperator(lx_op, lx.symmetric_tag)
            sol = lx.linear_solve(lx_op, b.values)

            return hilbert_space.from_values(sol.value)
        else:
            return super().solve(b, scale, shift)


    def spectral_bounds(self, hilbert_space: FiniteDifference) -> Array:
        return jnp.array([-jnp.sum(4 / hilbert_space.dx_range ** 2), 0.0])

    def to_matrix(self, hilbert_space: FiniteDifference) -> Array:
        """Kronecker sum of the per-axis stencils, matching action's Dirichlet
        zero padding: the ghost nodes outside the grid are held at zero."""
        total = jnp.zeros((hilbert_space.dim, hilbert_space.dim))

        for axis, (n, dx) in enumerate(
            zip(hilbert_space.mesh_size, hilbert_space.dx_range)
        ):
            stencil = (
                jnp.diag(jnp.full(n, -2.0))
                + jnp.diag(jnp.ones(n - 1), 1)
                + jnp.diag(jnp.ones(n - 1), -1)
            ) / dx**2

            factors = [jnp.eye(m) for m in hilbert_space.mesh_size]
            factors[axis] = stencil
            total += reduce(jnp.kron, factors)

        return total


class FiniteDifferencePotentialEnergy(Operator):
    domain: ClassVar[type[AbstractHilbertSpace]] = FiniteDifference
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)
    potential: Callable[[ScalarLike], ScalarLike]

    def values(self, hilbert_space: FiniteDifference) -> Array:
        return hilbert_space.eval(self.potential)

    def action(self, y: FiniteDifferenceState) -> FiniteDifferenceState:
        return y.hilbert_space.from_values(self.values(y.hilbert_space) * y.values)

    def exp_action(self, h: ScalarLike, y: FiniteDifferenceState) -> FiniteDifferenceState:
        return y.hilbert_space.from_values(jnp.exp(h * self.values(y.hilbert_space)) * y.values)

    def solve(
        self, 
        b: FiniteDifferenceState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> FiniteDifferenceState:

        hilbert_space = b.hilbert_space
        values = self.values(hilbert_space)
        # values() is grid-shaped, so divide grid by grid rather than mixing it
        # with the flat coefficient vector
        return hilbert_space.from_values(b.values / (scale * values + shift))

    def spectral_bounds(self, hilbert_space: FiniteDifference) -> Array:
        values = self.values(hilbert_space)
        return jnp.array([jnp.min(values), jnp.max(values)])

    def to_matrix(self, hilbert_space: FiniteDifference) -> Array:
        return jnp.diag(hilbert_space.flatten(self.values(hilbert_space)))

