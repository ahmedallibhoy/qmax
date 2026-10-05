from functools import reduce
from typing import Any, ClassVar, Iterable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar

from ._introspect import CountDictKey, InterfaceCount
from ._types import ComplexScalarLike
from .exponentiators.base import AbstractExponentiator, DelegatingExponentiator, Order
from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import IncompatibleDomainError, Operator

__all__ = ["DirectProductState", "DirectProduct"]


class DirectProductState[H: DirectProduct[Any]](AbstractState[H]):
    def factor(self, idx: int) -> AbstractState:
        idx_start = sum(self.hilbert_space.dim_list[:idx])
        idx_end = idx_start + self.hilbert_space[idx].dim
        coeffs = self.coeffs[..., idx_start:idx_end]
        return self.hilbert_space[idx].from_coeffs(coeffs)


class DirectProduct[S: DirectProductState[Any]](AbstractHilbertSpace[S]):
    state_type: ClassVar[type] = DirectProductState
    spaces: tuple[AbstractHilbertSpace, ...]

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.spaces[idx]

    def __getitem__(self, idx: int) -> AbstractHilbertSpace:
        return self.factor(idx)

    @property
    def num_factors(self) -> int:
        return len(self.spaces)

    @property
    def dim_list(self) -> list[int]:
        return [space.dim for space in self.spaces]

    @property
    def dim(self) -> int:
        return sum(self.dim_list)

    def product_state(self, y_list: Iterable[AbstractState]) -> S:
        coeffs = jnp.concatenate([y.coeffs for y in y_list], axis=-1)
        return self.from_coeffs(coeffs)

    def innerp(self, y1: S, y2: S) -> Scalar:
        return jnp.sum(
            jnp.array([y1.factor(idx) @ y2.factor(idx) for idx in range(self.num_factors)])
        )

    def block_diagonal_operator(self, op_list: Iterable[Operator]) -> BlockDiagonalOperator[S]:
        return BlockDiagonalOperator(self, children=op_list)


class BlockDiagonalExponentiator[S: DirectProductState[Any]](
    DelegatingExponentiator["BlockDiagonalOperator[S]", S]
):
    def schedule(
        self, op: BlockDiagonalOperator[S]
    ) -> Sequence[tuple[int, ComplexScalarLike, int]]:

        return [(idx, 1, 1) for idx in range(op.domain.num_factors)]

    @property
    def operator_type(self) -> type[BlockDiagonalOperator[S]]:
        return BlockDiagonalOperator

    def exp(self, op: BlockDiagonalOperator[S], h: ComplexScalarLike, y: S) -> S:
        exp_ys = [child_op._exp(h, y.factor(idx)) for idx, child_op in enumerate(op.children)]
        return op.domain.product_state(exp_ys)

    @property
    def order(self) -> Order:
        return None


class AbstractDirectProductOperator[S: DirectProductState[Any]](Operator[S]):
    domain: DirectProduct[S] = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.domain, DirectProduct):
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on a tensor space but received "
                f"domain of type {type(self.domain).__name__}"
            )


class BlockDiagonalOperator[S: DirectProductState[Any]](AbstractDirectProductOperator[S]):
    exponentiator: AbstractExponentiator = eqx.field(
        default=BlockDiagonalExponentiator(), kw_only=True
    )

    def __check_init__(self):
        if len(self.children) != self.domain.num_factors:
            raise ValueError(
                f"Received {len(self.children)} operators but "
                f"{self.domain} has {self.domain.num_factors} factors"
            )

        for idx in range(self.domain.num_factors):
            if self.children[idx].domain != self.domain[idx]:
                raise IncompatibleDomainError(
                    f"Domain at index {idx} is {self.domain[idx]} but "
                    f"operand at index {idx} acts on {self.children[idx].domain}"
                )

    def action(self, y: S) -> S:
        y_list = [op.action(y.factor(idx)) for idx, op in enumerate(self.children)]
        return self.domain.product_state(y_list)

    def adj_action(self, y: S) -> S:
        y_list = [op.adj_action(y.factor(idx)) for idx, op in enumerate(self.children)]
        return self.domain.product_state(y_list)

    def _solve(self, b: S, scale: ComplexScalarLike = -1.0, shift: ComplexScalarLike = 0.0) -> S:
        y_list = [op._solve(b.factor(idx), scale, shift) for idx, op in enumerate(self.children)]
        return self.domain.product_state(y_list)

    @property
    def spectral_bounds(self) -> Array:
        bounds = jnp.stack([op.spectral_bounds for op in self.children])
        return jnp.array([jnp.min(bounds), jnp.max(bounds)])

    def to_matrix(self) -> Array:
        mat_list = [op.to_matrix() for op in self.children]
        return jax.scipy.linalg.block_diag(*mat_list)

    def adjoint(self) -> Operator[S]:
        return self.domain.block_diagonal_operator([op.adjoint() for op in self.children])

    def interface_count(
        self, parent_key: Optional[CountDictKey] = None, child_idx: Optional[int] = None
    ) -> InterfaceCount:

        key = self.count_key(parent_key, child_idx)
        c_list = [op.interface_count(key, idx) for idx, op in enumerate(self.children)]

        return InterfaceCount(
            action=reduce(lambda a, b: a | b, [c.action for c in c_list]),
            adj_action=reduce(lambda a, b: a | b, [c.adj_action for c in c_list]),
            solve=reduce(lambda a, b: a | b, [c.solve for c in c_list]),
            exp_action=None,
        )
