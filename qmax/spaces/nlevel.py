from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._types import ComplexScalarLike
from ..hilbert_space import AbstractHilbertSpace, AbstractState
from ..operator import Operator

__all__ = ["NLevel"]


class NLevelState[H: NLevel[Any] = NLevel](AbstractState[H]):
    """
    """


class NLevel[S: NLevelState[Any] = NLevelState](AbstractHilbertSpace[S]):
    state_type: ClassVar = NLevelState
    _dim: int

    @property
    def dim(self) -> int:
        return self._dim

    def fock(self, idx: int) -> S:
        if idx >= self.dim:
            raise ValueError(f"idx={idx} must be less than dimension of space (dim={self.dim})")

        coeffs = jnp.zeros((self.dim,))
        coeffs = coeffs.at[idx].set(1.0)
        return self.from_coeffs(coeffs)

    def coherent(self, alpha: ComplexScalarLike) -> S:
        """
        Given alpha, generates a state such that a(y) ≈ alpha * y where a is the Annihilator operator. 
        """
        def next_coeff(c, k):
            c_next = alpha / jnp.sqrt(k) * c
            return c_next, c_next

        _, coeffs = jax.lax.scan(next_coeff, 1.0, jnp.arange(1, self.dim))
        y = self.from_coeffs(jnp.concatenate((jnp.asarray(1.0)[None], coeffs)))
        y = y / y.norm()
        return y

    def annihilator(self) -> Annihilator[S]:
        return Annihilator(self)

    def creator(self) -> Creator[S]:
        return Creator(self)

    
def annihilate(y: NLevelState[Any]):
    dim = y.hilbert_space.dim
    vals = jnp.sqrt(jnp.arange(1, dim))
    coeffs = jnp.concatenate([vals * y.coeffs[..., 1:], jnp.zeros_like(y.coeffs[..., :1])], axis=-1)
    return y.hilbert_space.from_coeffs(coeffs)


def create(y: NLevelState[Any]):
    dim = y.hilbert_space.dim
    vals = jnp.sqrt(jnp.arange(1, dim))
    coeffs = jnp.concatenate([jnp.zeros_like(y.coeffs[..., :1]), vals * y.coeffs[..., :-1]], axis=-1)
    return y.hilbert_space.from_coeffs(coeffs)


class Annihilator[S: NLevelState[Any]](Operator[S]):

    def action(self, y: S) -> S:
        return annihilate(y)

    def adj_action(self, y: S) -> S:
        return create(y)

    def to_matrix(self) -> Array:
        vals = jnp.sqrt(jnp.arange(1, self.domain.dim))
        return jnp.diag(vals, k=1)

    def adjoint(self) -> Creator[S]:
        return Creator(self.domain)


class Creator[S: NLevelState[Any]](Operator[S]):

    def action(self, y: S) -> S:
        return create(y)

    def adj_action(self, y: S) -> S:
        return annihilate(y)

    def to_matrix(self) -> Array:
        vals = jnp.sqrt(jnp.arange(1, self.domain.dim))
        return jnp.diag(vals, k=-1)

    def adjoint(self) -> Annihilator[S]:
        return Annihilator(self.domain)


