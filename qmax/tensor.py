from __future__ import annotations

import math
from abc import abstractmethod
from functools import reduce
from typing import Any, Callable, ClassVar, Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._introspect import Count, InterfaceCount, Path
from ._types import ComplexScalarLike
from .exponentiators.base import AbstractExponentiator, DelegatingExponentiator, Order
from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Identity, IncompatibleDomainError, Operator

__all__ = [
    "TensorProduct", 
    "TensorPower"
]

def apply_along_tensor(fn: Callable[[Array], Array], tensor: ArrayLike, axis: int) -> Array:
    r"""
    Given a tensor y \in V_1 ⊗ V_2 ⊗ ... ⊗ V_m and a linear map A: V_i -> V_i,
    this function applies I ⊗ ... ⊗ A ⊗ ... ⊗ I to y, or equivalently, applies
    the map A to each axis of y.

    Specifically, we have

        V_1 ⊗ V_2 ⊗ ... ⊗ V_m ≅ V_i x V_i x ... x V_i,

    where the direct product consists of n_1 x n_2 x n_{i-1} x n_{i+1} ... x n_m,
    independent copies of V_i, with n_j = dim V_j. Thus A may be lifted to the map
    blockdiag(A, A, ..., A) acting on y.

    The map blockdiag(A, A, ..., A) may be expressed in coordinates as follows. The
    tensor y is an array with shape (n_1, n_2, ..., n_m), whose element at the
    multi-index (k_1, k_2, ..., k_m) is y_{k_1, k_2, ..., k_m}. Thus y may be
    decomposed into a collection of n_1 x n_2 x n_{i - 1} x n_{i + 1} x ... x n_m
    vectors, having coordinates

        y_{k_1, ..., 1, ... k_m}
        y_{k_1, ..., 2, ... k_m}
        ...
        y_{k_1, ..., n_i, ..., k_m}

    for each multi-index (k_1, ..., k_{i-1}, k_{i+1}, ..., k_m). The block diagonal
    map is computed by vmapping A over all such multi-indices.
    """

    t = jnp.moveaxis(tensor, axis, 0)
    rest = t.shape[1:]
    out = jax.vmap(fn, in_axes=1, out_axes=1)(t.reshape(t.shape[0], -1))
    return jnp.moveaxis(out.reshape((out.shape[0],) + rest), 0, axis)


def apply_along_state(
    state_fn: Callable[[AbstractState], AbstractState], y: TensorState[Any], factor_idx: int
):
    """
    Same function as apply_along_tensor except acting on batched state vectors
    rather than raw coefficient arrays. This function extracts the coefficients,
    then delegates to apply_along_tensor, and repacks the result into a TensorState
    """

    num_factors = y.hilbert_space.num_factors
    space = y.hilbert_space[factor_idx]
    fn = lambda col: state_fn(space.from_coeffs(col)).coeffs
    axis = factor_idx % num_factors - num_factors
    return y.hilbert_space.from_tensor(apply_along_tensor(fn, y.tensor, axis))


class TensorState[H: AbstractTensorSpace[Any] = AbstractTensorSpace](AbstractState[H]):
    @property
    def tensor(self) -> Array:
        return self.coeffs.reshape(*self.coeffs.shape[:-1], *self.hilbert_space.dim_list)


class AbstractTensorSpace[S: TensorState[Any]](AbstractHilbertSpace[S]):
    state_type: ClassVar = TensorState

    @abstractmethod
    def factor(self, idx: int) -> AbstractHilbertSpace:
        pass

    def __getitem__(self, idx: int) -> AbstractHilbertSpace:
        return self.factor(idx)

    @property
    @abstractmethod
    def num_factors(self) -> int:
        pass

    @property
    @abstractmethod
    def dim_list(self) -> list[int]:
        pass

    @property
    def dim(self) -> int:
        return math.prod(self.dim_list)

    def from_tensor(self, tensor: ArrayLike) -> S:
        tensor = jnp.asarray(tensor)
        batch_shape = tensor.shape[: tensor.ndim - self.num_factors]
        return self.from_coeffs(tensor.reshape(*batch_shape, self.dim))

    def product_state(self, y_list: Iterable[AbstractState]) -> S:
        expanded = [
            y.coeffs.reshape(
                *y.coeffs.shape[:-1],
                *(1,) * factor_idx,
                y.coeffs.shape[-1],
                *(1,) * (self.num_factors - factor_idx - 1),
            )
            for factor_idx, y in enumerate(y_list)
        ]
        return self.from_tensor(reduce(lambda a, b: a * b, expanded))

    def lift(self, op: Operator, factor_idx: int) -> LiftOperator[S]:
        return LiftOperator(self, op, factor_idx)

    def kron_sum(self, op_list: Iterable[Operator]) -> KroneckerSum[S]:
        return KroneckerSum(self, children=op_list)

    def kron_prod(self, op_list: Iterable[Operator]) -> KroneckerProduct[S]:
        return KroneckerProduct(self, children=op_list)


class TensorProduct[S: TensorState[Any] = TensorState](AbstractTensorSpace[S]):
    spaces: tuple[AbstractHilbertSpace, ...]

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.spaces[idx]

    @property
    def num_factors(self) -> int:
        return len(self.spaces)

    @property
    def dim_list(self) -> list[int]:
        return [hs.dim for hs in self.spaces]


class TensorPower[S: TensorState[Any] = TensorState](AbstractTensorSpace[S]):
    factorspace: AbstractHilbertSpace
    power: int

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.factorspace

    @property
    def num_factors(self) -> int:
        return self.power

    @property
    def dim_list(self) -> list[int]:
        return [self.factorspace.dim for _ in range(self.power)]

    @property
    def dim(self) -> int:
        return self.factorspace.dim**self.power


class AbstractTensorOperator[S: TensorState[Any]](Operator[S]):
    domain: AbstractTensorSpace[S] = eqx.field(static=True)

    def _check_tensor_domain(self):
        if not isinstance(self.domain, AbstractTensorSpace):
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on a tensor space but received "
                f"domain of type {type(self.domain).__name__}"
            )


class LiftExp[S: TensorState[Any]](DelegatingExponentiator["LiftOperator[S]", S]):
    def schedule(self, op: LiftOperator[S]) -> list[tuple[int, ComplexScalarLike, int]]:
        # the lifted operator acts on each of the remaining subspaces in turn
        (A,) = op.children
        # return [(0, 1.0, op.domain.dim // A.domain.dim)]
        return [(0, 1.0, 1)]

    def exp(self, op: LiftOperator[S], h: ComplexScalarLike, y: S) -> S:

        (A,) = op.children
        return apply_along_state(lambda s: A._exp(h, s), y, op.factor_idx)

    @property
    def operator_type(self) -> type[LiftOperator[S]]:
        return LiftOperator

    @property
    def order(self) -> Order:
        return None


class LiftOperator[S: TensorState[Any]](AbstractTensorOperator[S]):
    factor_idx: int
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)

    def __init__(self, domain: AbstractTensorSpace[S], A: Operator, factor_idx: int):

        self.domain = domain
        self.children = (A,)

        if factor_idx < 0:
            factor_idx = factor_idx % self.num_factors

        self.factor_idx = factor_idx

    def __check_init__(self):
        self._check_tensor_domain()

        (A,) = self.children
        if A.domain != self.domain[self.factor_idx]:
            raise IncompatibleDomainError(
                f"Domain at index {self.factor_idx} is {self.domain[self.factor_idx]} but "
                f"operand acts on {A.domain}"
            )

    @property
    def num_factors(self):
        return self.domain.num_factors

    def action(self, y: S) -> S:
        (A,) = self.children
        return apply_along_state(lambda s: A.action(s), y, self.factor_idx)

    def adj_action(self, y: S) -> S:
        (A,) = self.children
        return apply_along_state(lambda s: A.adj_action(s), y, self.factor_idx)

    @property
    def spectral_bounds(self) -> Array:
        (A,) = self.children
        return A.spectral_bounds

    def _solve(self, b: S, scale: ComplexScalarLike = -1.0, shift: ComplexScalarLike = 0.0) -> S:
        (A,) = self.children
        return apply_along_state(lambda s: A._solve(s, scale, shift), b, self.factor_idx)

    def to_matrix(self) -> Array:
        (A,) = self.children
        mat_list = [
            jnp.eye(self.domain[idx].dim) if idx != self.factor_idx else A.to_matrix()
            for idx in range(self.num_factors)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def adjoint(self) -> Operator[S]:
        (A,) = self.children
        return LiftOperator(self.domain, A.adjoint(), self.factor_idx)

    @property
    def label(self) -> str:
        return f"{type(self).__name__}(idx={self.factor_idx})"

    def interface_count(
        self, parent_path: Optional[Path] = None, child_idx: Optional[int] = None
    ) -> InterfaceCount:
        (A,) = self.children
        path = self.path(parent_path, child_idx)
        c = A.interface_count(path, 0)
        # num = self.domain.dim // A.domain.dim
        num = 1

        return InterfaceCount(
            action=num * c.action,
            adj_action=num * c.adj_action,
            solve=num * c.solve,
            exp_action=self._exp_action_count(path),
        )


class KroneckerProductMixin[S: TensorState[Any]](AbstractTensorOperator[S]):
    def __check_init__(self):
        self._check_tensor_domain()

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

    def interface_count(
        self, parent_path: Optional[Path] = None, child_idx: Optional[int] = None
    ) -> InterfaceCount:
        path = self.path(parent_path, child_idx)
        # dim = self.domain.dim
        # scaled = [
        #    (dim // self.domain[idx].dim, op.interface_count(path, idx))
        #    for idx, op in enumerate(self.children)
        # ]

        scaled = [(1, op.interface_count(path, idx)) for idx, op in enumerate(self.children)]

        return InterfaceCount(
            action=reduce(lambda a, b: a | b, [num * c.action for num, c in scaled]),
            adj_action=reduce(lambda a, b: a | b, [num * c.adj_action for num, c in scaled]),
            solve={path: Count(solves=1)},
            exp_action=self._exp_action_count(path),
        )

    def adjoint(self) -> Operator[S]:
        return type(self)(self.domain, children=tuple(op.adjoint() for op in self.children))


class KroneckerSumExp[S: TensorState[Any]](DelegatingExponentiator["KroneckerSum[S]", S]):
    def schedule(self, op: KroneckerSum[S]) -> list[tuple[int, ComplexScalarLike, int]]:
        # each factor acts on every slice along its own axis
        # return [
        #    (idx, 1.0, op.domain.dim // op.domain[idx].dim)
        #    for idx in range(len(op.children))
        # ]

        return [(idx, 1.0, 1) for idx in range(len(op.children))]

    def exp(self, op: KroneckerSum[S], h: ComplexScalarLike, y: S) -> S:

        for factor_idx, child in enumerate(op.children):
            y = apply_along_state(lambda s, child=child: child._exp(h, s), y, factor_idx)

        return y

    @property
    def operator_type(self) -> type[KroneckerSum[S]]:
        return KroneckerSum

    @property
    def order(self) -> Order:
        return None


class KroneckerSum[S: TensorState[Any]](KroneckerProductMixin[S]):
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)

    def action(self, y: S) -> S:
        return reduce(
            lambda a, b: a + b,
            [
                apply_along_state(lambda s, op=op: op.action(s), y, factor_idx)
                for factor_idx, op in enumerate(self.children)
            ],
        )

    def adj_action(self, y: S) -> S:
        return reduce(
            lambda a, b: a + b,
            [
                apply_along_state(lambda s, op=op: op.adj_action(s), y, factor_idx)
                for factor_idx, op in enumerate(self.children)
            ],
        )

    @property
    def spectral_bounds(self) -> Array:
        bounds = [op.spectral_bounds for idx, op in enumerate(self.children)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self) -> Array:
        mat = jnp.zeros((self.domain.dim, self.domain.dim), dtype=complex)
        for factor_idx, op in enumerate(self.children):
            mat_list = [
                jnp.eye(self.domain[idx].dim) if idx != factor_idx else op.to_matrix()
                for idx in range(self.domain.num_factors)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat


class KroneckerProduct[S: TensorState[Any]](KroneckerProductMixin[S]):
    def action(self, y: S) -> S:
        for factor_idx, op in enumerate(self.children):
            if isinstance(op, Identity):
                continue
            y = apply_along_state(lambda s, op=op: op.action(s), y, factor_idx)

        return y

    def adj_action(self, y: S) -> S:
        for factor_idx, op in enumerate(self.children):
            if isinstance(op, Identity):
                continue
            y = apply_along_state(lambda s, op=op: op.adj_action(s), y, factor_idx)
        return y

    @property
    def spectral_bounds(self) -> Array:
        lo, hi = 1.0, 1.0
        for idx, op in enumerate(self.children):
            a, b = op.spectral_bounds
            corners = jnp.array([lo * a, lo * b, hi * a, hi * b])
            lo, hi = jnp.min(corners), jnp.max(corners)
        return jnp.array([lo, hi])

    def to_matrix(self) -> Array:
        mat_list = [op.to_matrix() for idx, op in enumerate(self.children)]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)
