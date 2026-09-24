from functools import partial, reduce
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._types import ComplexScalarLike
from ..exponentiators.base import AbstractExponentiator, ExactExponentiator, Order
from ..operator import AbstractDiagonalOperator, Operator
from .spatial_discretization import (
    AbstractPotentialEnergy,
    PotentialFunction,
    SpatialDiscretization,
    SpatiallyDiscretizedState,
    _to_tuple,
)

__all__ = ["PseudoSpectral"]


class PseudoSpectralState(SpatiallyDiscretizedState["PseudoSpectral"]):
    @property
    def values(self) -> Array:
        hs = self.hilbert_space
        grid = hs.to_grid(self.coeffs, hs.num_modes)
        return jnp.fft.ifftn(hs.pad(grid), axes=hs.spatial_axes, norm="forward")


class PseudoSpectral(SpatialDiscretization[PseudoSpectralState]):
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
            a = (N + 1) // 2  # DC + positive freqs
            lo = jax.lax.slice_in_dim(out, 0, a, axis=ax)  # low positive half
            hi = jax.lax.slice_in_dim(out, a, N, axis=ax)  # negative half
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
            lo = jax.lax.slice_in_dim(out, 0, (m + 1) // 2, axis=ax)  # keep ceil(m/2) low
            hi = jax.lax.slice_in_dim(out, N - m // 2, N, axis=ax)  # keep floor(m/2) high
            out = jnp.concatenate([lo, hi], axis=ax)
        return out

    def from_values(self, values: ArrayLike) -> PseudoSpectralState:
        full = jnp.fft.fftn(values, self.mesh_size, axes=self.spatial_axes, norm="forward")
        return PseudoSpectralState(self.flatten(self.truncate(full)), hilbert_space=self)

    @property
    def lossless(self):
        return self.num_modes == self.mesh_size

    def laplacian(self) -> PseudoSpectralLaplacian:
        return PseudoSpectralLaplacian(self)

    def potential_energy(self, potential: PotentialFunction) -> PseudoSpectralPotentialEnergy:

        return PseudoSpectralPotentialEnergy(self, potential)

    def momentum(self, axis: int = 0) -> PseudoSpectralMomentum:
        return PseudoSpectralMomentum(self, axis)


class PseudoSpectralLaplacian(AbstractDiagonalOperator[PseudoSpectralState]):
    domain: PseudoSpectral = eqx.field(static=True)

    @property
    def eigvals(self) -> Array:
        hilbert_space = self.domain
        num_modes = hilbert_space.num_modes
        ds = [(hilbert_space.x1[i] - hilbert_space.x0[i]) / n for i, n in enumerate(num_modes)]
        k_per_axis = [2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for (n, d) in zip(num_modes, ds)]
        ks = hilbert_space.grid_vectors(k_per_axis)
        return -(jnp.linalg.norm(ks, axis=-1) ** 2)


class PseudoSpectralExponentiator(
    ExactExponentiator["PseudoSpectralPotentialEnergy", PseudoSpectralState]
):
    @property
    def operator_type(self) -> type[PseudoSpectralPotentialEnergy]:
        return PseudoSpectralPotentialEnergy

    def effective_order(self, op: PseudoSpectralPotentialEnergy) -> Order:
        return None if op.domain.lossless else 1


class PseudoSpectralPotentialEnergy(AbstractPotentialEnergy[PseudoSpectralState]):
    domain: PseudoSpectral = eqx.field(static=True)
    exponentiator: AbstractExponentiator = eqx.field(
        default=PseudoSpectralExponentiator(), kw_only=True
    )

    def _solve(
        self,
        b: PseudoSpectralState,
        scale: ComplexScalarLike = -1.0,
        shift: ComplexScalarLike = 0.0,
    ) -> PseudoSpectralState:

        if self.domain.lossless:
            return AbstractPotentialEnergy._solve(self, b, scale, shift)
        else:
            return Operator._solve(self, b, scale, shift)

    def to_matrix(self):
        Vhat = jnp.fft.fftn(
            self.values, self.domain.mesh_size, axes=self.domain.spatial_axes, norm="forward"
        ).reshape(-1)

        mode_per_axis = [
            jnp.round(jnp.fft.fftfreq(n) * n).astype(int) for n in self.domain.num_modes
        ]
        mode_vecs = self.domain.grid_vectors(mode_per_axis)
        diff = (mode_vecs[:, None, :] - mode_vecs[None, :, :]) % jnp.array(self.domain.mesh_size)

        flat = jnp.ravel_multi_index(
            [diff[..., a] for a in range(diff.shape[-1])], self.domain.mesh_size, mode="wrap"
        )

        return Vhat[flat]


class PseudoSpectralMomentum(AbstractDiagonalOperator[PseudoSpectralState]):
    domain: PseudoSpectral = eqx.field(static=True)
    axis: int = 0

    @property
    def eigvals(self) -> Array:
        hilbert_space = self.domain
        num_modes = hilbert_space.num_modes
        ds = [(hilbert_space.x1[i] - hilbert_space.x0[i]) / n for i, n in enumerate(num_modes)]
        k_per_axis = [2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for (n, d) in zip(num_modes, ds)]
        ks = hilbert_space.grid_vectors(k_per_axis)
        return ks[:, self.axis]
