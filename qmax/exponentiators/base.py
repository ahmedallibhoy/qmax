from __future__ import annotations

from typing import Union, Literal, TYPE_CHECKING
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState

if TYPE_CHECKING:
    from ..operator import Operator


type Order = Union[int, Literal[jnp.inf]]


# Maximum degree in the sweeps performed by the adapt methods of
# KrylovExponentiator and ChebyshevExponentiator.
N_MAX = 100


class AbstractExponentiator(eqx.Module):

    def adapt(
        self,
        op: Operator,
        hilbert_space: AbstractHilbertSpace,
        dt_max: ScalarLike) -> AbstractExponentiator:

        return self

    def __call__(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        return self.exp(op, dt, y)

    @abstractmethod
    def exp(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        pass

    @property
    @abstractmethod
    def order(self) -> Order:
        pass


class ExactExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        # TODO: better exception handling if op does not implement exp_action
        return op.exp_action(dt, y)

    @property
    def order(self) -> Order:
        return jnp.inf
