from abc import abstractmethod
from typing import Any, Optional, Self, cast

import jax.numpy as jnp
from jaxtyping import Scalar, ScalarLike

from ._internal import _update_field
from ._types import ComplexArrayLike, ComplexScalarLike, RealScalarLike
from .exponentiators.split import AbstractSplitMethod, Strang
from .expression_tree import ExpressionTree, IncompatibleDomainError
from .hilbert_space import AbstractState
from .operator import AddOperator, Operator
from .timevarying_operator import AbstractTimeVaryingOperator, ConstantTimeVaryingOperator

type QuasiOperatorLike[S: AbstractState[Any]] = (
    Operator[S] | AbstractTimeVaryingOperator[S] | AbstractQuasiOperator[S]
)

# TODO: multiplication of AbstractQuasiOperator by controls


class AbstractQuasiOperator[S: AbstractState[Any]](ExpressionTree["AbstractQuasiOperator", S]):
    @abstractmethod
    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike, y: S) -> Operator[S]:
        pass

    def flow(self, h: ComplexScalarLike, y: S) -> S:
        raise NotImplementedError

    @property
    def has_flow(self) -> bool:
        return False

    def flow_order(self) -> Optional[int]:
        # TODO: complete order estimates of flows
        return None

    def evaluate(self, t: ScalarLike, y: S) -> Operator[S]:
        return self.quadrature(jnp.atleast_1d(t), jnp.ones(1), y)

    def __call__(self, t: ScalarLike, y: S) -> Operator[S]:
        return self.evaluate(t, y)

    def __add__(self, other: QuasiOperatorLike[S]) -> AbstractQuasiOperator[S]:
        if isinstance(other, Operator) or isinstance(other, AbstractTimeVaryingOperator):
            other = StateIndependentQuasiOperator(other)

        if not isinstance(other, AbstractQuasiOperator):
            return NotImplemented

        return AddQuasiOperator(self, other)

    def __radd__(self, other: QuasiOperatorLike[S]) -> AbstractQuasiOperator[S]:
        if isinstance(other, Operator) or isinstance(other, AbstractTimeVaryingOperator):
            other = StateIndependentQuasiOperator(other)

        if not isinstance(other, AbstractQuasiOperator):
            return NotImplemented

        return AddQuasiOperator(other, self)

    def __sub__(self, other: QuasiOperatorLike[S]) -> AbstractQuasiOperator[S]:
        return self + (-other)

    def __rsub__(self, other: QuasiOperatorLike[S]) -> AbstractQuasiOperator[S]:
        return (-self) + other

    def __mul__(self, other: RealScalarLike) -> AbstractQuasiOperator[S]:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulQuasiOperator(self, other)

    def __rmul__(self, other: RealScalarLike) -> AbstractQuasiOperator[S]:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulQuasiOperator(self, other)

    def __truediv__(self, other: RealScalarLike) -> AbstractQuasiOperator[S]:
        if not jnp.isscalar(other):
            return NotImplemented

        return ScalarMulQuasiOperator(self, 1.0 / other)

    def __neg__(self) -> AbstractQuasiOperator[S]:
        return -1.0 * self


class StateIndependentQuasiOperator[S: AbstractState[Any]](AbstractQuasiOperator[S]):
    t_op: AbstractTimeVaryingOperator

    def __init__(self, t_op: Operator[S] | AbstractTimeVaryingOperator[S]):
        if isinstance(t_op, Operator):
            t_op = ConstantTimeVaryingOperator(t_op)

        self.domain = t_op.domain
        self.t_op = t_op

    def flow(self, h: ComplexScalarLike, y: S) -> S:
        if not isinstance(self.t_op, ConstantTimeVaryingOperator):
            raise NotImplementedError

        return self.t_op.op.exp(h, y)

    @property
    def has_flow(self) -> bool:
        return isinstance(self.t_op, ConstantTimeVaryingOperator)

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike, y: S) -> Operator[S]:
        return self.t_op.quadrature(t_quad, weights)


class AddQuasiOperator[S: AbstractState[Any]](AbstractQuasiOperator[S]):
    split_method: AbstractSplitMethod

    def __init__(
        self,
        A: AbstractQuasiOperator[S],
        B: AbstractQuasiOperator[S],
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
        self.name = name

    def with_split_method(self, split_method: AbstractSplitMethod) -> Self:
        return cast(Self, _update_field(self, "split_method", split_method))

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike, y: S) -> Operator[S]:
        (A, B) = self.children

        return AddOperator(
            A.quadrature(t_quad, weights, y),
            B.quadrature(t_quad, weights, y),
            exponentiator=self.split_method,
        )

    def flow(self, h: ComplexScalarLike, y: S) -> S:
        return self.split_method.flow(self, h, y)

    @property
    def has_flow(self) -> bool:
        (A, B) = self.children
        return A.has_flow and B.has_flow


class ScalarMulQuasiOperator[S: AbstractState[Any]](AbstractQuasiOperator[S]):
    c: Scalar

    def __init__(
        self, A: AbstractQuasiOperator[S], c: RealScalarLike, *, name: Optional[str] = None
    ):

        self.children = (A,)
        self.c = jnp.asarray(c)
        self.domain = A.domain
        self.name = name

    def quadrature(self, t_quad: ComplexArrayLike, weights: ComplexArrayLike, y: S) -> Operator[S]:
        (A,) = self.children
        return A.quadrature(t_quad, self.c * weights, y)

    def flow(self, h: ComplexScalarLike, y: S) -> S:
        (A,) = self.children
        return A.flow(self.c * h, y)

    @property
    def has_flow(self) -> bool:
        (A,) = self.children
        return A.has_flow
