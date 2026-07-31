from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import ScalarLike

from ..hilbert_space import AbstractState
from .base import AbstractExponentiator, Order

if TYPE_CHECKING:
    from ..operator import Operator


class ForwardEuler(AbstractExponentiator):

    def exp(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        return y + dt * op.action(y)

    @property
    def order(self) -> Order:
        return 1


class ImplicitEuler(AbstractExponentiator):

    def exp(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y, scale=-dt, shift=1.0)

    @property
    def order(self) -> Order:
        return 1


class CrankNicolson(AbstractExponentiator):

    def exp(self, op: Operator, dt: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y + dt / 2 * op.action(y), scale=-dt / 2, shift=1.0)

    @property
    def order(self) -> Order:
        return 2
