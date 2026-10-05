from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._types import ComplexScalarLike
from .exponentiators.base import AbstractExponentiator, ExactExponentiator
from .operator import AbstractHermitianOperator

if TYPE_CHECKING:
    from .hilbert_space import AbstractState


class Identity[S: AbstractState[Any]](AbstractHermitianOperator[S]):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def action(self, y: S) -> S:
        return y

    def exp_action(self, h: ComplexScalarLike, y: S) -> S:
        return jnp.exp(h) * y

    def _solve(self, b: S, scale: ComplexScalarLike = -1.0, shift: ComplexScalarLike = 0.0) -> S:
        return b / (shift + scale)

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([1.0, 1.0])

    def to_matrix(self) -> Array:
        return jnp.eye(self.domain.dim)


class Zero[S: AbstractState[Any]](AbstractHermitianOperator[S]):
    exponentiator: AbstractExponentiator = eqx.field(default=ExactExponentiator(), kw_only=True)

    def action(self, y: S) -> S:
        return self.domain.zeros_like(y)

    def exp_action(self, h: ComplexScalarLike, y: S) -> S:
        return y

    def _solve(self, b: S, scale: ComplexScalarLike = -1.0, shift: ComplexScalarLike = 0.0) -> S:
        return b / shift

    @property
    def spectral_bounds(self) -> Array:
        return jnp.array([0.0, 0.0])

    def to_matrix(self) -> Array:
        return jnp.zeros((self.domain.dim, self.domain.dim))


class Gram[S: AbstractState[Any]](AbstractHermitianOperator[S]):
    def action(self, y: S) -> S:
        fn = lambda c: self.domain.innerp(y, self.domain.from_coeffs(c))
        f_transpose = jax.linear_transpose(fn, y.coeffs)
        (coeffs,) = f_transpose(jnp.ones(y.shape, dtype=complex))
        return self.domain.from_coeffs(jnp.conj(coeffs))

    def to_matrix(self) -> Array:
        Y = self.domain.from_coeffs(jnp.eye(self.domain.dim))
        return jnp.conj(self.action(Y).coeffs)
