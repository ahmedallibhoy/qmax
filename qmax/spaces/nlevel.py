from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ScalarLike

from ..hilbert_space import AbstractState, AbstractHilbertSpace
from ..operator import Operator, Identity
from ..exponentiators import AbstractExponentiator, TruncatedTaylorExponentiator, NoExponentiator
from ..tensor import TensorState, TensorPower, KroneckerProduct


#   TODO:
#       1. coherent states
#       2. generators on NLevel spaces
#       3. merge with Qubit
#

class NLevelState(AbstractState):
    """
    """


class NLevel(AbstractHilbertSpace):
    state_type: ClassVar = NLevelState
    _dim: int 

    @property
    def dim(self) -> int:
        return self._dim

    def fock(self, idx: int) -> NLevelState:
        if idx >= self.dim:
            raise ValueError(f"idx={idx} must be less than dimension of space (dim={self.dim})")

        coeffs = jnp.zeros((self.dim,))
        coeffs = coeffs.at[idx].set(1.0)
        return self.from_coeffs(coeffs)

    def coherent(self, alpha: ScalarLike) -> NLevelState:
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

    def annihilator(self) -> Annihilator:
        return Annihilator(self)

    def creator(self) -> Creator:
        return Creator(self)

    
def annihilate(y: NLevelState) -> NLevelState:
    dim = y.hilbert_space.dim
    vals = jnp.sqrt(jnp.arange(1, dim))
    coeffs = jnp.concatenate([vals * y.coeffs[..., 1:], jnp.zeros_like(y.coeffs[..., :1])], axis=-1)
    return y.hilbert_space.from_coeffs(coeffs)


def create(y: NLevelState) -> NLevelState:
    dim = y.hilbert_space.dim
    vals = jnp.sqrt(jnp.arange(1, dim))
    coeffs = jnp.concatenate([jnp.zeros_like(y.coeffs[..., :1]), vals * y.coeffs[..., :-1]], axis=-1)
    return y.hilbert_space.from_coeffs(coeffs)


class Annihilator(Operator):

    def action(self, y: NLevelState) -> NLevelState:
        return annihilate(y)

    def adj_action(self, y: NLevelState) -> NLevelState:
        return create(y)

    def to_matrix(self) -> Array:
        vals = jnp.sqrt(jnp.arange(1, self.domain.dim))
        return jnp.diag(vals, k=1)

    def adjoint(self) -> Creator:
        return Creator(self.domain)


class Creator(Operator):

    def action(self, y: NLevelState) -> NLevelState:
        return create(y)

    def adj_action(self, y: NLevelState) -> NLevelState:
        return annihilate(y)

    def to_matrix(self) -> Array:
        vals = jnp.sqrt(jnp.arange(1, self.domain.dim))
        return jnp.diag(vals, k=-1)

    def adjoint(self) -> Annihilator:
        return Annihilator(self.domain)


