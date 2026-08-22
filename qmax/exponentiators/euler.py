from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxtyping import ScalarLike

from .._introspect import CountDict, Path, Field
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

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        return op.interface_count(parent_path, field).action


class ImplicitEuler(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y, scale=-h, shift=1.0)

    @property
    def order(self) -> Order:
        return 1

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        return op.interface_count(parent_path, field).solve


class CrankNicolson(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y + (h / 2) * op.action(y), scale=-h / 2, shift=1.0)

    @property
    def order(self) -> Order:
        return 2

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        i_count = op.interface_count(parent_path, field)
        return i_count.action + i_count.solve
