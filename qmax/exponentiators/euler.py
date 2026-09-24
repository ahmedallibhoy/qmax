from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from jaxtyping import ScalarLike

from .._introspect import CountDict, Path
from ..hilbert_space import AbstractState
from .._types import ComplexScalarLike
from .base import AbstractExponentiator, Order

if TYPE_CHECKING:
    from ..operator import Operator


__all__ = ["ForwardEuler", "ImplicitEuler", "Cayley"]


class ForwardEuler(AbstractExponentiator):
    r"""
    Forward Euler method: $\exp(hA)y \approx (I + hA)y$. 

    !!! warning
        This method should be avoided since it is numerically unstable and not unitary preserving.
    """

    def exp(self, op: Operator, h: ComplexScalarLike, y: AbstractState) -> AbstractState:
        return y + h * op.action(y)

    @property
    def order(self) -> Order:
        return 1

    def count(
        self, 
        op: Operator, 
        h: ComplexScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        return op.interface_count(parent_path, child_idx).action


class ImplicitEuler(AbstractExponentiator):
    r"""
    Implicit Euler method: $\exp(hA)y \approx (I - hA)^{-1}y$. 

    !!! warning
        This method should be avoided since it is not unitary preserving.
    """

    def exp(self, op: Operator, h: ComplexScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y, scale=-h, shift=1.0)

    @property
    def order(self) -> Order:
        return 1

    def count(
        self, 
        op: Operator, 
        h: ComplexScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        return op.interface_count(parent_path, child_idx).solve


class Cayley(AbstractExponentiator):
    r"""
    Approximates the matrix exponential action via the Cayley transform: 
    $\exp(hA)y \approx (I - \frac{h}{2}A)^{-1}(I + \frac{h}{2}A)y$. 
    This equivalent to a half step of the forward Euler method, followed by a 
    half step of the implicit Euler method.

    !!! info 
        Since the method is implicit, this exponentiator is best paired with operators 
        that have an efficient `solve` override. 
    """

    def exp(self, op: Operator, h: ComplexScalarLike, y: AbstractState) -> AbstractState:
        return op.solve(y + (h / 2) * op.action(y), scale=-h / 2, shift=1.0)

    @property
    def order(self) -> Order:
        return 2

    def count(
        self, 
        op: Operator, 
        h: ComplexScalarLike, 
        parent_path: Optional[Path]=None, 
        child_idx: Optional[int]=None) -> CountDict:

        i_count = op.interface_count(parent_path, child_idx)
        return i_count.action | i_count.solve
