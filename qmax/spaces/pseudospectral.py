from typing import Callable, ClassVar, Optional

from functools import partial, reduce

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator, AbstractDiagonalOperator
from ..exponentiators import Order, AbstractExponentiator
from .spatial_discretization import (
    SpatialDiscretization, 
    SpatiallyDiscretizedState, 
    AbstractPotentialEnergy, 
    _to_tuple
)


class PseudoSpectralState(SpatiallyDiscretizedState):

    @property
    def values(self) -> Array:
        hs = self.hilbert_space
        grid = hs.to_grid(self.coeffs, hs.num_modes)
        return jnp.fft.ifftn(hs.pad(grid), axes=hs.spatial_axes, norm="forward")


class PseudoSpectral(SpatialDiscretization):
    state_type: ClassVar = PseudoSpectralState
    endpoint: ClassVar[bool] = False
    num_modes: tuple[int, ...] = eqx.field(converter=partial(_to_tuple, dtype=int))

    @property
    def dim(self):
        return reduce(lambda a, b: a * b, self.num_modes)

    def pad(self, v_fft: Array) -> Array:
        """Spectral zero-pad the mode grid (num_modes) up to the mesh grid (mesh_size),
        inserting zeros at the high frequencies (no fftshift)."""
        out = v_fft
        for ax, M in zip(self.spatial_axes, self.mesh_size):
            N = out.shape[ax]
            a = (N + 1) // 2                                   # DC + positive freqs
            lo = jax.lax.slice_in_dim(out, 0, a, axis=ax)      # low positive half
            hi = jax.lax.slice_in_dim(out, a, N, axis=ax)      # negative half
            zshape = list(out.shape) 
            zshape[ax] = M - N
            out = jnp.concatenate([lo, jnp.zeros(zshape, out.dtype), hi], axis=ax)
        return out

    def truncate(self, v_fft: Array) -> Array:
        """Spectral truncate the mesh grid (mesh_size) down to the mode grid (num_modes),
        dropping the high frequencies (no fftshift)."""
        out = v_fft
        for ax, m in zip(self.spatial_axes, self.num_modes):
            N = out.shape[ax]
            lo = jax.lax.slice_in_dim(out, 0, (m + 1) // 2, axis=ax)   # keep ceil(m/2) low
            hi = jax.lax.slice_in_dim(out, N - m // 2, N, axis=ax)     # keep floor(m/2) high
            out = jnp.concatenate([lo, hi], axis=ax)
        return out

    def innerp(self, 
        y1: PseudoSpectralState, 
        y2: PseudoSpectralState) -> ScalarLike:

        return jnp.sum(jnp.conj(y1.coeffs) * y2.coeffs, axis=-1)

    def from_values(self, values: ArrayLike) -> PseudoSpectralState:
        full = jnp.fft.fftn(values, self.mesh_size, axes=self.spatial_axes, norm="forward")
        return PseudoSpectralState(self.flatten(self.truncate(full)), self)

    @property
    def lossless(self):
        return self.num_modes == self.mesh_size

    def laplacian(self) -> PseudoSpectralLaplacian:
        return PseudoSpectralLaplacian(self)

    def potential_energy(
        self, 
        potential: Callable[[ArrayLike], ScalarLike]) -> PseudoSpectralPotentialEnergy:

        return PseudoSpectralPotentialEnergy(self, potential)


class PseudoSpectralLaplacian(AbstractDiagonalOperator):

    @property
    def eigvals(self) -> Array:
        hilbert_space = self.domain
        num_modes = hilbert_space.num_modes
        ds = [(hilbert_space.xf[i] - hilbert_space.x0[i]) / n for i, n in enumerate(num_modes)]
        k_per_axis = [2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for (n, d) in zip(num_modes, ds)]
        ks = hilbert_space.grid_vectors(k_per_axis)
        return -jnp.linalg.norm(ks, axis=-1) ** 2


class PseudoSpectralExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return AbstractPotentialEnergy.exp_action(op, h, y)

    @property
    def operator_type(self) -> type:
        return PseudoSpectralPotentialEnergy

    def effective_order(self, op: PseudoSpectralPotentialEnergy) -> Order:
        return None if op.domain.lossless else 1

    @property
    def order(self) -> Order:
        return None


class PseudoSpectralPotentialEnergy(AbstractPotentialEnergy):
    exponentiator: AbstractExponentiator = eqx.field(default=PseudoSpectralExponentiator(), kw_only=True)

    @property
    def has_exact_exponential(self) -> bool:
        return self.domain.lossless

    def _solve(
        self, 
        b: PseudoSpectralState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> PseudoSpectralState:

        if self.domain.lossless:
            return AbstractPotentialEnergy._solve(self, b, scale, shift)
        else:
            return Operator._solve(self, b, scale, shift)

    def to_matrix(self):
        Vhat = jnp.fft.fftn(
            self.values, self.domain.mesh_size,
            axes=self.domain.spatial_axes, norm="forward").reshape(-1)
        
        mode_per_axis = [jnp.round(jnp.fft.fftfreq(n) * n).astype(int) for n in self.domain.num_modes]
        mode_vecs = self.domain.grid_vectors(mode_per_axis)     
        diff = (mode_vecs[:, None, :] - mode_vecs[None, :, :]) % jnp.array(self.domain.mesh_size) 
        
        flat = jnp.ravel_multi_index(
            [diff[..., a] for a in range(diff.shape[-1])], 
            self.domain.mesh_size, mode="wrap") 
            
        return Vhat[flat]

