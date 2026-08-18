from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ScalarLike

from .._internal import _overrides
from ..hilbert_space import AbstractHilbertSpace, AbstractState

if TYPE_CHECKING:
    from ..operator import Operator

type Order = Optional[int]

def min_order(*orders: Order) -> Order:
    finite = [order for order in orders if order is not None]
    return min(finite) if finite else None


class NotExponentiableError(Exception):
    def __init__(self, reason: str, path: tuple[str, ...]=()):
        self.reason, self.path = reason, path
        super().__init__(reason if not path else f"{reason}\n  in {' -> '.join(path)}")

    def under(self, op) -> NotExponentiableError:
        return type(self)(self.reason, (type(op).__name__,) + self.path)


class AbstractExponentiator(eqx.Module):

    def adapt(
        self,
        op: Operator,
        hilbert_space: AbstractHilbertSpace,
        dt_max: ScalarLike) -> AbstractExponentiator:

        return self

    def __call__(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        self.check_exponentiable_tree(op)
        return self.exp(op, h, y)

    @abstractmethod
    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        pass

    def check_exponentiable_tree(self, op: Operator) -> None:
        """
        Checks `op` and everything beneath it in the expression tree, recording `op` in the path
        of any NotExponentiableError raised from below. This is the only place the path is
        extended, so every entry point -- Operator.exp, Operator.check_exponentiable_tree, or a
        direct call to this exponentiator -- annotates the root exactly once.
        """
        try:
            self.check_exponentiable(op)
        except NotExponentiableError as e:
            raise e.under(op) from None

    def check_exponentiable(self, op: Operator) -> None:
        """
        Checks whether this exponentiator can act on `op` itself, recursing into whichever
        children its exp() needs via their check_exponentiable_tree(). Overridden by
        subclasses; never call this directly, use check_exponentiable_tree.
        """
        pass

    @property
    @abstractmethod
    def order(self) -> Order:
        pass

    def effective_order(self, op: Operator) -> Order:
        return self.order


class ExactExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.exp_action(h, y)

    def check_exponentiable(self, op) -> None:
        from ..operator import Operator
        if not _overrides(type(op), "exp_action", Operator):
            raise NotExponentiableError(
                f"ExactExponentiator requires {type(op).__name__} to implement exp_action")

    @property
    def order(self) -> Order:
        return None


class ShiftScaleExponentiator(AbstractExponentiator):
    """
    Exponentiates op = shift * I + scale * A
    """

    def exp(self, op, h, y):
        return jnp.exp(h * op.shift) * op.op.exp(h * op.scale, y)

    def check_exponentiable(self, op: Operator):
        from ..operator import ShiftScaleOperator

        if not isinstance(op, ShiftScaleOperator):
            raise NotExponentiableError(
                f"{type(self).__name__} can only exponentiate operators of type ShiftScaleOperator "
                f"but received operator of type {type(op).__name__}"
            )

        op.op.check_exponentiable_tree()

    @property
    def order(self) -> Order:
        return None

    def effective_order(self, op):
        return op.op.exp_order


class NoExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        raise NotExponentiableError(f"Cannot exponentiate. Use .with_exponentiator to assign an exponentiator")

    def check_exponentiable(self, op: Operator):
        raise NotExponentiableError(f"Cannot exponentiate. Use .with_exponentiator to assign an exponentiator")

    @property
    def order(self) -> Order:
        raise NotExponentiableError(f"Cannot exponentiate. Use .with_exponentiator to assign an exponentiator")
