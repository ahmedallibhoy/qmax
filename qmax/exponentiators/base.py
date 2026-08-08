from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ScalarLike

from ..hilbert_space import AbstractHilbertSpace, AbstractState

if TYPE_CHECKING:
    from ..operator import Operator


# The order of accuracy of an exponentiator, or None when it is exact and so
# places no order limit on the solution.
type Order = Optional[int]


def min_order(*orders: Order) -> Order:
    finite = [order for order in orders if order is not None]
    return min(finite) if finite else None


class AbstractExponentiator(eqx.Module):

    def adapt(
        self,
        op: Operator,
        hilbert_space: AbstractHilbertSpace,
        dt_max: ScalarLike) -> AbstractExponentiator:

        return self

    def __call__(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return self.exp(op, h, y)

    @abstractmethod
    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        pass

    @property
    @abstractmethod
    def order(self) -> Order:
        pass


class ExactExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.exp_action(h, y)

    @property
    def order(self) -> Order:
        return None
