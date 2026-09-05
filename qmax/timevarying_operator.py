from __future__ import annotations

from typing import Optional
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import ArrayLike, ScalarLike

from ._internal import _update_field
from ._introspect import _rows
from .expression_tree import AbstractExpressionTree
from .hilbert_space import AbstractHilbertSpace
from .operator import Operator, AddOperator, IncompatibleDomainError
from .exponentiators import AbstractSplitMethod, Strang
from .control import AbstractControl, ConstantControl


class AbstractTimeVaryingOperator(AbstractExpressionTree):

    def __call__(self, t: ScalarLike) -> Operator:
        return self.evaluate(t)

    @abstractmethod
    def quadrature(self, t_quad: ArrayLike, weights: ArrayLike) -> Operator:
        pass

    def evaluate(self, t: ScalarLike) -> Operator:
        return self.quadrature(jnp.atleast_1d(t), jnp.ones(1))

    # --------------------------------------------------------------------------------------------
    # Operator Algebra
    # --------------------------------------------------------------------------------------------

    def __add__(self, other: Operator | AbstractTimeVaryingOperator) -> AbstractTimeVaryingOperator:
        self._check_compatible(other)

        if isinstance(other, Operator):
            other = ConstantTimeVaryingOperator(other)
        if not isinstance(other, AbstractTimeVaryingOperator):
            return NotImplemented

        return AddTimeVaryingOperator(self, other)

    def __radd__(self, other: Operator | AbstractTimeVaryingOperator) -> AbstractTimeVaryingOperator:
        self._check_compatible(other)
    
        if isinstance(other, Operator):
            other = ConstantTimeVaryingOperator(other)
        if not isinstance(other, AbstractTimeVaryingOperator):
            return NotImplemented

        return AddTimeVaryingOperator(other, self)

    def __sub__(self, other: Operator | AbstractTimeVaryingOperator) -> AbstractTimeVaryingOperator:
        return self + (-other)

    def __rsub__(self, other: Operator | AbstractTimeVaryingOperator) -> AbstractTimeVaryingOperator:
        return (-self) + other

    def __mul__(self, other: ScalarLike | AbstractControl) -> AbstractTimeVaryingOperator:
        if jnp.isscalar(other):
            other = ConstantControl(other)
        if not isinstance(other, AbstractControl):
            return NotImplemented

        return ScalarMulTimeVaryingOperator(self, other)

    def __rmul__(self, other: ScalarLike | AbstractControl) -> AbstractTimeVaryingOperator:
        if jnp.isscalar(other):
            other = ConstantControl(other)
        if not isinstance(other, AbstractControl):
            return NotImplemented

        return ScalarMulTimeVaryingOperator(self, other)

    def __neg__(self) -> AbstractTimeVaryingOperator:
        return -1.0 * self


class ConstantTimeVaryingOperator(AbstractTimeVaryingOperator):
    op: Operator

    def __init__(self, op: Operator, *, name: Optional[str]=None):
        self.op = op
        self.domain = op.domain
        self.name = name if name is not None else op.label

    def quadrature(self, t_quad: ArrayLike, weights: ArrayLike) -> Operator:
        return jnp.sum(weights) * self.op


class AddTimeVaryingOperator(AbstractTimeVaryingOperator):
    split_method: AbstractSplitMethod

    def __init__(
        self,
        A: AbstractTimeVaryingOperator,
        B: AbstractTimeVaryingOperator,
        *,
        split_method: AbstractSplitMethod=Strang(),
        name: Optional[str]=None):

        if A.domain != B.domain:
            raise IncompatibleDomainError(
                f"Cannot add operators on different domains: A={type(A).__name__} acts on {A.domain}, "
                f"but B={type(B).__name__} acts on {B.domain},"
            )

        self.children = (A, B)
        self.domain = A.domain
        self.split_method = split_method
        self.name = name if name is not None else f"({A.label} + {B.label})"

    def with_split_method(self, split_method: AbstractSplitMethod):
        return _update_field(self, "split_method", split_method)

    def quadrature(self, t_quad: ArrayLike, weights: ArrayLike) -> Operator:
        A, B = self.children
        return AddOperator(
            A.quadrature(t_quad, weights), B.quadrature(t_quad, weights),
            exponentiator=self.split_method)


class ScalarMulTimeVaryingOperator(AbstractTimeVaryingOperator):
    u: AbstractControl

    def __init__(
        self,
        A: AbstractTimeVaryingOperator,
        u: AbstractControl,
        *,
        name: Optional[str]=None):

        self.children = (A,)
        self.u = u
        self.domain = A.domain
        self.name = name if name is not None else f"{type(u).__name__} * {A.label}"

    def quadrature(self, t_quad: ArrayLike, weights: ArrayLike) -> Operator:
        (A,) = self.children
        return A.quadrature(t_quad, weights * jax.vmap(self.u)(t_quad))


