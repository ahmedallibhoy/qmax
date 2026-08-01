from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import ScalarLike

from ..hilbert_space import AbstractState
from .base import AbstractExponentiator, Order

if TYPE_CHECKING:
    from ..operator import Operator


class ForwardEuler(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return y + h * op.action(y)

    @property
    def order(self) -> Order:
        return 1


class ImplicitEuler(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y, scale=-h, shift=1.0)

    @property
    def order(self) -> Order:
        return 1


class CrankNicolson(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y + h / 2 * op.action(y), scale=-h / 2, shift=1.0)

    @property
    def order(self) -> Order:
        return 2
