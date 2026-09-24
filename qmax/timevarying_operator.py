from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

from ._internal import _update_field
from ._types import ComplexArrayLike, RealScalarLike
from .control import AbstractControl, ConstantControl
from .exponentiators import AbstractSplitMethod, Strang
from .expression_tree import AbstractExpressionTree
from .hilbert_space import AbstractState
from .operator import AddOperator, IncompatibleDomainError, Operator


class AbstractTimeVaryingOperator[S: AbstractState[Any]](
    AbstractExpressionTree["AbstractTimeVaryingOperator", S]
):
    """
    A timevarying operator $H(t)$.
    """

    def __call__(self, t: ScalarLike) -> Operator:
        """
        Evaluates the timevarying operator at time `t`.

        Args:
            t (Scalar): Evaluation time

        Returns:
            the operator $H(t)$.
        """
        return self.evaluate(t)

    @abstractmethod
    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike) -> Operator:
        pass

    def evaluate(self, t: ScalarLike) -> Operator:
        return self.quadrature(jnp.atleast_1d(t), jnp.ones(1))

    # --------------------------------------------------------------------------------------------
    # Operator Algebra
    # --------------------------------------------------------------------------------------------

    def __add__(self, other: Operator | AbstractTimeVaryingOperator) -> AddTimeVaryingOperator[S]:
        self._check_compatible(other)

        if isinstance(other, Operator):
            other = ConstantTimeVaryingOperator(other)
        if not isinstance(other, AbstractTimeVaryingOperator):
            return NotImplemented

        return AddTimeVaryingOperator(self, other)

    def __radd__(self, other: Operator | AbstractTimeVaryingOperator) -> AddTimeVaryingOperator[S]:
        self._check_compatible(other)

        if isinstance(other, Operator):
            other = ConstantTimeVaryingOperator(other)
        if not isinstance(other, AbstractTimeVaryingOperator):
            return NotImplemented

        return AddTimeVaryingOperator(other, self)

    def __sub__(self, other: Operator | AbstractTimeVaryingOperator) -> AbstractTimeVaryingOperator:
        return self + (-other)

    def __rsub__(
        self, other: Operator | AbstractTimeVaryingOperator
    ) -> AbstractTimeVaryingOperator:
        return (-self) + other

    def __mul__(self, other: RealScalarLike | AbstractControl) -> AbstractTimeVaryingOperator:
        if isinstance(other, AbstractControl):
            pass
        elif jnp.isscalar(other):
            other = ConstantControl(other)
        else:
            return NotImplemented

        return ScalarMulTimeVaryingOperator(self, other)

    def __rmul__(self, other: RealScalarLike | AbstractControl) -> AbstractTimeVaryingOperator:
        if isinstance(other, AbstractControl):
            pass
        elif jnp.isscalar(other):
            other = ConstantControl(other)
        else:
            return NotImplemented

        return ScalarMulTimeVaryingOperator(self, other)

    def __neg__(self) -> AbstractTimeVaryingOperator:
        return -1.0 * self


class ConstantTimeVaryingOperator[S: AbstractState[Any]](AbstractTimeVaryingOperator[S]):
    op: Operator[S]

    def __init__(self, op: Operator, *, name: Optional[str] = None):
        self.op = op
        self.domain = op.domain
        self.name = name if name is not None else op.label

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike) -> Operator:
        return jnp.sum(weights) * self.op


class AddTimeVaryingOperator[S: AbstractState[Any]](AbstractTimeVaryingOperator[S]):
    split_method: AbstractSplitMethod

    def __init__(
        self,
        A: AbstractTimeVaryingOperator,
        B: AbstractTimeVaryingOperator,
        *,
        split_method: AbstractSplitMethod = Strang(),
        name: Optional[str] = None,
    ):

        if A.domain != B.domain:
            raise IncompatibleDomainError(
                f"Cannot add operators on different domains: "
                f"A={type(A).__name__} acts on {A.domain}, "
                f"but B={type(B).__name__} acts on {B.domain},"
            )

        self.children = (A, B)
        self.domain = A.domain
        self.split_method = split_method
        self.name = name if name is not None else f"({A.label} + {B.label})"

    def with_split_method(self, split_method: AbstractSplitMethod):
        return _update_field(self, "split_method", split_method)

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike) -> Operator:
        A, B = self.children
        return AddOperator(
            A.quadrature(t_quad, weights),
            B.quadrature(t_quad, weights),
            exponentiator=self.split_method,
        )


class ScalarMulTimeVaryingOperator[S: AbstractState[Any]](AbstractTimeVaryingOperator[S]):
    u: AbstractControl

    def __init__(
        self, A: AbstractTimeVaryingOperator, u: AbstractControl, *, name: Optional[str] = None
    ):

        self.children = (A,)
        self.u = u
        self.domain = A.domain
        self.name = name if name is not None else f"{type(u).__name__} * {A.label}"

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike) -> Operator:
        (A,) = self.children
        return A.quadrature(t_quad, weights * jax.vmap(self.u)(t_quad))
