from typing import Callable, ClassVar

from functools import partial, reduce

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator
from ..exponentiators import AbstractExponentiator, ExactExponentiator
from .spatial_discretization import SpatialDiscretization, SpatiallyDiscretizedState, _to_tuple


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

    def from_values(self, values: ArrayLike) -> PseudoSpectralState:
        full = jnp.fft.fftn(values, self.mesh_size, axes=self.spatial_axes, norm="forward")
        return PseudoSpectralState(self.flatten(self.truncate(full)), self)


class PseudoSpectralLaplacian(Operator):
    domain: ClassVar = PseudoSpectral
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def eigvals(self, hilbert_space: PseudoSpectral) -> Array:
        num_modes = hilbert_space.num_modes
        ds = [(hilbert_space.xf[i] - hilbert_space.x0[i]) / n for i, n in enumerate(num_modes)]
        k_per_axis = [2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for (n, d) in zip(num_modes, ds)]
        ks = hilbert_space.grid_vectors(k_per_axis)
        return -jnp.linalg.norm(ks, axis=-1) ** 2

    def action(self, y: PseudoSpectralState) -> PseudoSpectralState:
        return PseudoSpectralState(self.eigvals(y.hilbert_space) * y.coeffs, y.hilbert_space)

    def exp_action(self, h: ScalarLike, y: PseudoSpectralState) -> PseudoSpectralState:
        coeffs = jnp.exp(h * self.eigvals(y.hilbert_space)) * y.coeffs
        return PseudoSpectralState(coeffs, y.hilbert_space)

    def solve(
        self, 
        b: PseudoSpectralState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> PseudoSpectralState:

        hilbert_space = b.hilbert_space
        eigvals = self.eigvals(hilbert_space)
        return hilbert_space.from_coeffs(b.coeffs / (scale * eigvals + shift))

    def spectral_bounds(self, hilbert_space: PseudoSpectral) -> Array:
        eigvals = self.eigvals(hilbert_space)
        return jnp.array([jnp.min(eigvals), jnp.max(eigvals)])

    def to_matrix(self, hilbert_space: PseudoSpectral) -> Array:
        return jnp.diag(self.eigvals(hilbert_space))


class PseudoSpectralExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        exp_vals = jnp.exp(h * op.values(y.hilbert_space)) * y.values
        return y.hilbert_space.from_values(exp_vals)

    @property
    def order(self) -> int:
        return 1


class PseudoSpectralPotentialEnergy(Operator):
    domain: ClassVar = PseudoSpectral
    exponentiator: AbstractExponentiator = eqx.field(default=PseudoSpectralExponentiator(), kw_only=True)
    potential: Callable[[ScalarLike], ScalarLike]

    def values(self, hilbert_space: PseudoSpectral) -> Array:
        return hilbert_space.eval(self.potential)

    def action(self, y: PseudoSpectralState) -> PseudoSpectralState:
        vals = self.values(y.hilbert_space) * y.values 
        return y.hilbert_space.from_values(vals)

    def spectral_bounds(self, hilbert_space: PseudoSpectral) -> Array:
        values = self.values(hilbert_space)
        return jnp.array([jnp.min(values), jnp.max(values)])

    def solve(
        self, 
        b: PseudoSpectralState, 
        scale: ScalarLike=-1.0, 
        shift: ScalarLike=0.0) -> PseudoSpectralState:

        if b.hilbert_space.num_modes == b.hilbert_space.mesh_size:
            hilbert_space = b.hilbert_space
            values = self.values(hilbert_space)
            return hilbert_space.from_values(b.values / (scale * values + shift))
        else:
            return super().solve(b, scale, shift)

    def to_matrix(self, hilbert_space: PseudoSpectral):
        Vhat = jnp.fft.fftn(
            self.values(hilbert_space), hilbert_space.mesh_size,
            axes=hilbert_space.spatial_axes, norm="forward").reshape(-1)
        
        mode_per_axis = [jnp.round(jnp.fft.fftfreq(n) * n).astype(int) for n in hilbert_space.num_modes]
        mode_vecs = hilbert_space.grid_vectors(mode_per_axis)     
        diff = (mode_vecs[:, None, :] - mode_vecs[None, :, :]) % jnp.array(hilbert_space.mesh_size) 
        
        flat = jnp.ravel_multi_index(
            [diff[..., a] for a in range(diff.shape[-1])], 
            hilbert_space.mesh_size, mode="wrap") 
            
        return Vhat[flat]

