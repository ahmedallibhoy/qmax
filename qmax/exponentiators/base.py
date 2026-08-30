from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ScalarLike, Scalar

from .._introspect import CountDict, Path
from .._internal import _update_field
from ..hilbert_space import AbstractHilbertSpace, AbstractState

if TYPE_CHECKING:
    from ..operator import Operator


__all__ = [
    "Order",
    "min_order",
    "NotExponentiableError",
    "AbstractExponentiator",
    "DelegatingExponentiator",
    "ExactExponentiator",
    "ShiftScaleExponentiator",
    "NoExponentiator",
]

# TODO: fix redundant tree checks

type Order = Optional[int]

def min_order(*orders: Order) -> Order:
    finite = [order for order in orders if order is not None]
    return min(finite) if finite else None


class NotExponentiableError(Exception):
    def __init__(self, reason: str, path: Path=Path()):
        self.reason, self.path = reason, path
        super().__init__(f"{reason}\n  path: {path}")

    def from_path(self, path: Path=Path()) -> NotExponentiableError:
        #if self.path:
        #    return self
        return type(self)(self.reason, path)


class AbstractExponentiator(eqx.Module):

    # --------------------------------------------------------------------------------------------
    # Do not override
    # --------------------------------------------------------------------------------------------

    def adapt_tree(self, op: Operator, dt_max: ScalarLike) -> Operator:
        op = self.adapt_children(op, dt_max)
        return op.with_exponentiator(self.adapt(op, dt_max))

    def __call__(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        self.check_exponentiable_tree(op)
        return self.exp(op, h, y)

    def check_exponentiable_tree(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None):

        """
        Traverses the expression tree of op to ensure op is compatible with this 
        exponentiator. If not, raises NotExponentiableError reproducing the path 
        to the offending node. 
        """
        try:
            if not isinstance(op, self.operator_type):
                raise NotExponentiableError(
                    f"{type(self).__name__} can only exponentiate operators of type "
                    f"{self.operator_type.__name__} but received operator of type "
                    f"{type(op).__name__}"
                )
            self.check_exponentiable(op, parent_path, child_idx)
        except NotExponentiableError as e:
            if e.path:
                raise e from None
            raise e.from_path(op.path(parent_path, child_idx)) from None

    def can_exponentiate(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> bool:
        """
        Traverses the expression tree of op and returns True if this exponentiator 
        can be called on op without raising an error. 
        """
        try:
            self.check_exponentiable_tree(op, parent_path, child_idx)
            return True
        except NotExponentiableError:
            return False

    def tree_order(self, op: Operator) -> Order:
        """
        Traverses the expression tree of op to compute effective order of this 
        exponentiator when applied to op
        """
        return self.effective_order(op)

    def tree_count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:
        """
        Traverses the expression tree of op to count the number of calls to interface 
        methods of leaves per one call to self.exp(op, h, y), as a rough measure 
        of the computational effort of required by this exponentiatiator. 
        """
        return self.count(op, h, parent_path, child_idx)

    # --------------------------------------------------------------------------------------------
    # Should be overriden by subclasses if necessary
    # --------------------------------------------------------------------------------------------

    @abstractmethod
    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        pass

    @property
    @abstractmethod
    def order(self) -> Order:
        """
        The intrinsic order of the method itself, independent of any operator. None means the
        method contributes no truncation error of its own.
        """

    @abstractmethod
    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:
        """
        Counts number of calls to each interface of op required by one call 
        to self.exp(op, h, y) recursing into children of op, if necessary,  
        via op_child.exp_count for each child of op.
        """
        return CountDict()

    def adapt(
        self,
        op: Operator,
        dt_max: ScalarLike) -> AbstractExponentiator:
        """
        Given a max stepsize and an operator, returns an instance of type(self) with any
        parameters tuned to guarantee that reported order estimate is valid while reducing
        computational burden.

        May be overridden on exponentiators acting on leaf operators. 
        """
        return self

    def adapt_children(
        self, 
        op, 
        dt_max) -> Operator:
        """
        Returns an instance of type(op) whose children are adapted.

        Should be overridden on exponentiators acting on composite operators.
        """
        return op

    @property
    def operator_type(self) -> type[Operator]:
        """
        Type of operator this exponentiator is compatible with
        """
        from ..operator import Operator
        return Operator

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None):
        """
        Validates exponentiability by recursing into a composite operators tree via 
        op_child.check_exponentiable_tree for each child of op
        """
        pass

    def effective_order(self, op: Operator) -> Order:
        """
        The minimum of the intrinsic order and the orders of the exponentiators of each 
        child of op, queried via op_child.tree_order for each child of op
        """
        return self.order


class DelegatingExponentiator(AbstractExponentiator):
    """
    Base class for exponentiators that delegate to children of an operator, e.g. splitting
    exponentiators or exponentiators acting on tensor products
    """

    @abstractmethod
    def schedule(self, op: Operator) -> list[tuple[int, Scalar, int]]:
        """
        Schedule of (index, scale, mult) of op's children that this exponentiator 
        delegates to, i.e., (i, s, m) means that op.children[i].exp(scale * h, y)
        is called m times. 
        """
        pass 

    def count(self, op, h, parent_path=None, child_idx=None) -> CountDict:
        path = op.path(parent_path, child_idx)
        c = CountDict()
        for idx, scale, mult in self.schedule(op):
            c |= mult * op.children[idx].exp_count(scale * h, path, idx)
        return c

    def h_scales(self, op) -> list[Scalar]:
        """
        Returns list where h_scales[i] is the maximum scaling factor applied to op.children[i]
        """
        scales = [0.0] * len(op.children)
        for idx, coeff, _ in self.schedule(op):
            scales[idx] = max(scales[idx], abs(coeff))
        return scales

    def check_exponentiable(self, op, parent_path=None, child_idx=None) -> None:
        path = op.path(parent_path, child_idx)
        for idx in range(len(op.children)):
            op.children[idx].check_exponentiable_tree(path, idx)

    def adapt_children(self, op, dt_max) -> Operator:
        children = tuple(child.adapt(s * dt_max) for (child, s) in zip(op.children, self.h_scales(op)))
        return _update_field(op, "children", children)

    def effective_order(self, op: Operator) -> Order:
        """
        The minimum of the intrinsic order and the orders of the exponentiators of each 
        child of op, queried via op_child.tree_order for each child of op
        """
        return min_order(self.order, *(child.tree_order for child in op.children))


class ExactExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.exp_action(h, y)

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None):

        if not op.overrides_exp_action:
            raise NotExponentiableError(
                f"ExactExponentiator requires {type(op).__name__} to implement exp_action")

    @property
    def order(self) -> Order:
        return None

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        return op.interface_count(parent_path, child_idx).exp_action


class ShiftScaleExponentiator(DelegatingExponentiator):
    """
    Exponentiates op = shift * I + scale * A
    """

    def schedule(self, op: Operator) -> list[tuple[int, Scalar, int]]:
        # only the scale stretches the step; the shift contributes a phase
        return [(0, op.scale, 1)]

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        (A,) = op.children
        return jnp.exp(h * op.shift) * A._exp(h * op.scale, y)

    @property
    def operator_type(self) -> type:
        from ..operator import ShiftScaleOperator
        return ShiftScaleOperator

    @property
    def order(self) -> Order:
        return None


class NoExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        raise NotExponentiableError(
            f"Cannot exponentiate {type(op).__name__} since no exponentiator is assigned. "
            "Use .with_exponentiator to assign an exponentiator")

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None):
        
        raise NotExponentiableError(
            f"Cannot exponentiate {type(op).__name__} since no exponentiator is assigned. "
            "Use .with_exponentiator to assign an exponentiator")

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        raise NotExponentiableError(
            f"Cannot exponentiate {type(op).__name__} since no exponentiator is assigned. "
            "Use .with_exponentiator to assign an exponentiator")

    @property
    def order(self) -> Order:
        raise NotExponentiableError(f"Cannot compute order of NoExponentatiator")

