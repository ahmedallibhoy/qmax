from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ScalarLike

from .._introspect import CountDict, Path, Field, child_field
from .._internal import _update_field
from ..hilbert_space import AbstractHilbertSpace, AbstractState

if TYPE_CHECKING:
    from ..operator import Operator


type Order = Optional[int]

def min_order(*orders: Order) -> Order:
    finite = [order for order in orders if order is not None]
    return min(finite) if finite else None


class NotExponentiableError(Exception):
    def __init__(self, reason: str, path: Path=Path()):
        self.reason, self.path = reason, path
        super().__init__(f"{reason}\n  path: {path}")

    def from_path(self, path: Path=Path()) -> NotExponentiableError:
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
        parent_path: Path=Path(), 
        field: Field=Field()):

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
            self.check_exponentiable(op, parent_path, field)
        except NotExponentiableError as e:
            raise e.from_path(op.path(parent_path, field)) from None

    def can_exponentiate(
        self, 
        op: Operator, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> bool:
        """
        Traverses the expression tree of op and returns True if this exponentiator 
        can be called on op without raising an error. 
        """
        try:
            self.check_exponentiable_tree(op, parent_path, field)
            return True
        except NotExponentiableError:
            return False

    def tree_order(self, op: Operator) -> Order:
        """
        Traverses the expression tree of op to compute effective order of this 
        exponentiator when applied to op
        """
        self.check_exponentiable_tree(op)
        return self.effective_order(op)

    def tree_count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:
        """
        Traverses the expression tree of op to count the number of calls to interface 
        methods of leaves per one call to self.exp(op, h, y), as a rough measure 
        of the computational effort of required by this exponentiatiator. 
        """
        self.check_exponentiable_tree(op)
        return self.count(op, h, parent_path, field)

    # --------------------------------------------------------------------------------------------
    # Should be overriden by subclasses if necessary
    # --------------------------------------------------------------------------------------------

    @abstractmethod
    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        pass

    def schedule(self, op) -> tuple[tuple[Field, ScalarLike, int]]:
        return ()

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
        parent_path: Path=Path(), 
        field: Field=Field()):
        """
        Validates exponentiability by recursing into a composite operators tree via 
        op_child.check_exponentiable_tree for each child of op
        """
        pass

    @property
    @abstractmethod
    def order(self) -> Order:
        """
        The intrinsic order of the method itself, independent of any operator. None means the
        method contributes no truncation error of its own.
        """

    def effective_order(self, op: Operator) -> Order:
        """
        The minimum of the intrinsic order and the orders of the exponentiators of each 
        child of op, queried via op_child.tree_order for each child of op
        """
        return self.order

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:
        """
        Counts number of calls to each interface of op required by one call 
        to self.exp(op, h, y) recursing into children of op, if necessary,  
        via op_child.exp_count for each child of op.
        """
        return CountDict()


class ExactExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        return op.exp_action(h, y)

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Path=Path(), 
        field: Field=Field()):

        if not op.has_exact_exponential:
            raise NotExponentiableError(
                f"ExactExponentiator requires {type(op).__name__} to implement exp_action")

    @property
    def order(self) -> Order:
        return None

    def count(
        self, 
        op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        return op.interface_count(parent_path, field).exp_action


class ShiftScaleExponentiator(AbstractExponentiator):
    """
    Exponentiates op = shift * I + scale * A
    """

    def exp(self, op, h, y):
        (A,) = op.children
        return jnp.exp(h * op.shift) * A.exp(h * op.scale, y)

    def adapt_children(self, op: Operator, dt_max: ScalarLike) -> Operator:
        # only the scale stretches the step; the shift contributes a phase
        (A,) = op.children
        return _update_field(op, "children", (A.adapt(jnp.abs(op.scale) * dt_max),))

    @property
    def operator_type(self) -> type:
        from ..operator import ShiftScaleOperator
        return ShiftScaleOperator

    def check_exponentiable(
        self,
        op: Operator,
        parent_path: Path=Path(),
        field: Field=Field()):

        (A,) = op.children
        path = op.path(parent_path, field)
        A.check_exponentiable_tree(path, child_field(0))

    @property
    def order(self) -> Order:
        return None

    def effective_order(self, op: Operator):
        (A,) = op.children
        return A.tree_order

    def count(
        self,
        op: Operator,
        h: ScalarLike,
        parent_path: Path=Path(),
        field: Field=Field()) -> CountDict:

        (A,) = op.children
        path = op.path(parent_path, field)
        return A.exp_count(h * op.scale, path, child_field(0))


class NoExponentiator(AbstractExponentiator):

    def exp(self, op: Operator, h: ScalarLike, y: AbstractState) -> AbstractState:
        raise NotExponentiableError(
            f"Cannot exponentiate {type(op).__name__} since no exponentiator is assigned. "
            "Use .with_exponentiator to assign an exponentiator")

    def check_exponentiable(
        self, 
        op: Operator, 
        parent_path: Path=Path(), 
        field: Field=Field()):
        
        raise NotExponentiableError(
            f"Cannot exponentiate {type(op).__name__} since no exponentiator is assigned. "
            "Use .with_exponentiator to assign an exponentiator")

    @property
    def order(self) -> Order:
        return None
