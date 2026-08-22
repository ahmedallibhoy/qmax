from __future__ import annotations

from typing import Callable, ClassVar
from abc import abstractmethod
from functools import reduce

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike

from ._internal import _update_field
from ._introspect import (
    Count, CountDict, InterfaceCount, Path, Field, child_field
)
from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator, Identity, IncompatibleDomainError
from .exponentiators import Order, min_order, AbstractExponentiator, Count


def apply_along_tensor(
    fn: Callable[[ArrayLike], ArrayLike],
    tensor: ArrayLike,
    axis: int) -> Array:
    """Apply `fn` to each 1-D slice along `axis`, batched over every other axis.

    `axis` is a literal array axis and may be negative. `fn` may change the axis
    length; the output shape is inferred from `fn`'s output."""

    t = jnp.moveaxis(tensor, axis, 0)
    rest = t.shape[1:]
    out = jax.vmap(fn, in_axes=1, out_axes=1)(t.reshape(t.shape[0], -1))
    return jnp.moveaxis(out.reshape((out.shape[0],) + rest), 0, axis)


def apply_along_state(
    state_fn: Callable[[AbstractState], AbstractState],
    y: TensorState,
    factor_idx: int) -> TensorState:
    """Apply a subspace endomorphism `state_fn` along factor `factor_idx` of a
    tensor product state, wrapping the from_coeffs / .coeffs / from_tensor
    boilerplate.

    `factor_idx` selects a tensor factor, not an axis of `y.coeff_tensor`. The
    factor axes are the trailing ones, so leading batch axes -- time from a
    timestepper, an outer vmap over initial conditions, both at once -- pass
    through untouched, and `state_fn` only ever sees an unbatched subspace
    state. Callers therefore never need to know the batch rank of `y`."""

    num_factors = y.hilbert_space.num_factors
    space = y.hilbert_space[factor_idx]
    fn = lambda col: state_fn(space.from_coeffs(col)).coeffs
    # Count from the back, so the axis is independent of the batch rank.
    axis = factor_idx % num_factors - num_factors
    return y.hilbert_space.from_tensor(apply_along_tensor(fn, y.coeff_tensor, axis))


class TensorState(AbstractState):
    hilbert_space: AbstractTensorSpace = eqx.field(static=True)

    @property
    def coeff_tensor(self) -> Array:
        return self.coeffs.reshape(*self.coeffs.shape[:-1], *self.hilbert_space.dim_list)


class AbstractTensorSpace(AbstractHilbertSpace):
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

    def from_tensor(self, coeff_tensor: ArrayLike):
        coeff_tensor = jnp.asarray(coeff_tensor)
        batch_shape = coeff_tensor.shape[:coeff_tensor.ndim - self.num_factors]
        return self.from_coeffs(coeff_tensor.reshape(*batch_shape, self.dim))

    def product_state(self, y_list: tuple[AbstractState, ...]) -> TensorState:
        expanded = [
            y.coeffs.reshape(
                *y.coeffs.shape[:-1],
                *(1,) * factor_idx,
                y.coeffs.shape[-1],
                *(1,) * (self.num_factors - factor_idx - 1))
            for factor_idx, y in enumerate(y_list)
        ]
        return self.from_tensor(reduce(lambda a, b: a * b, expanded))

    def lift(self, op: Operator, factor_idx: int) -> LiftOperator:
        return LiftOperator(self, factor_idx, children=(op,))


class TensorProduct(AbstractTensorSpace):
    spaces: tuple[AbstractHilbertSpace, ...]

    def factor(self, idx: int) -> AbstractHilbertSpace:
        return self.spaces[idx] 

    @property
    def num_factors(self) -> int:
        return len(self.spaces)

    @property
    def dim_list(self) -> list[int]:
        return [hs.dim for hs in self.spaces]


class TensorPower(AbstractTensorSpace):
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
        return self.factorspace.dim ** self.power


class AbstractTensorOperator(Operator):

    def _check_tensor_domain(self):
        if not isinstance(self.domain, AbstractTensorSpace):
            raise IncompatibleDomainError(
                f"{type(self).__name__} acts on a tensor space but received "
                f"domain of type {type(self.domain).__name__}"
            )


class LiftExp(AbstractExponentiator):

    def exp(
        self, 
        lift_op: LiftOperator, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        (A,) = lift_op.children
        return apply_along_state(lambda s: A.exp(h, s), y, lift_op.factor_idx)

    def adapt_children(
        self,
        op: LiftOperator,
        dt_max: ScalarLike) -> Operator:

        (A,) = op.children
        return _update_field(op, "children", (A.adapt(dt_max),))

    @property
    def operator_type(self) -> type:
        return LiftOperator

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

    def effective_order(self, lift_op: LiftOperator) -> Order:
        (A,) = lift_op.children
        return A.tree_order

    def count(
        self,
        lift_op: Operator,
        h: ScalarLike,
        parent_path: Path=Path(),
        field: Field=Field()) -> CountDict:

        (A,) = lift_op.children
        num = lift_op.domain.dim // A.domain.dim
        path = lift_op.path(parent_path, field)
        return num * A.exp_count(h, path, child_field(0))


class LiftOperator(AbstractTensorOperator):
    factor_idx: int
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)

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

    def action(self, y: TensorState) -> TensorState:
        (A,) = self.children
        return apply_along_state(lambda s: A.action(s), y, self.factor_idx)

    def adj_action(self, y):
        (A,) = self.children
        return apply_along_state(lambda s: A.adj_action(s), y, self.factor_idx)

    @property
    def spectral_bounds(self) -> Array:
        (A,) = self.children
        return A.spectral_bounds

    def _solve(self, b: TensorState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> TensorState:
        (A,) = self.children
        return apply_along_state(lambda s: A._solve(s, scale, shift), b, self.factor_idx)

    def to_matrix(self) -> Array:
        (A,) = self.children
        mat_list = [
            jnp.eye(self.domain[idx].dim) if idx != self.factor_idx else A.to_matrix()
            for idx in range(self.num_factors)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def adjoint(self) -> Operator:
        (A,) = self.children
        return LiftOperator(self.domain, self.factor_idx, children=(A.adjoint(),))

    def label(self) -> str:
        return f"{type(self).__name__}(idx={self.factor_idx})"

    def interface_count(self, parent_path: Path=Path(), field: Field=Field()) -> InterfaceCount:
        (A,) = self.children
        path = self.path(parent_path, field)
        c = A.interface_count(path, child_field(0))
        num = self.domain.dim // A.domain.dim

        return InterfaceCount(
            action     = num * c.action,
            adj_action = num * c.adj_action,
            solve      = num * c.solve,
            exp_action = self._exp_action_count(path),
        )


class KroneckerProductMixin(AbstractTensorOperator):

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

    def interface_count(self, parent_path: Path=Path(), field: Field=Field()) -> InterfaceCount:
        path = self.path(parent_path, field)
        dim = self.domain.dim
        scaled = [
            (dim // self.domain[idx].dim, op.interface_count(path, child_field(idx)))
            for idx, op in enumerate(self.children)
        ]

        return InterfaceCount(
            action = reduce(lambda a, b: a + b, [num * c.action for num, c in scaled]),
            adj_action = reduce(lambda a, b: a + b, [num * c.adj_action for num, c in scaled]),
            solve = {path: Count(solves=1)},
            exp_action = self._exp_action_count(path),
        )

    def adjoint(self) -> Operator:
        return type(self)(self.domain, children=tuple(op.adjoint() for op in self.children))


class KroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        for factor_idx, op in enumerate(kron_op.children):
            y = apply_along_state(lambda s, op=op: op.exp(h, s), y, factor_idx)

        return y

    def adapt_children(
        self,
        op: KroneckerSum,
        dt_max: ScalarLike) -> Operator:

        return _update_field(op, "children", tuple(
            factor.adapt(dt_max) for factor in op.children))

    @property
    def operator_type(self) -> type:
        return KroneckerSum

    def check_exponentiable(
        self,
        op: Operator,
        parent_path: Path=Path(),
        field: Field=Field()):

        path = op.path(parent_path, field)
        for idx, factor in enumerate(op.children):
            factor.check_exponentiable_tree(path, child_field(idx))

    @property
    def order(self) -> Order:
        return None

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min_order(*[op.tree_order for op in kron_op.children])

    def count(
        self, 
        kron_op: Operator, 
        h: ScalarLike, 
        parent_path: Path=Path(), 
        field: Field=Field()) -> CountDict:

        path = kron_op.path(parent_path, field)
        c = CountDict()
        for idx, op in enumerate(kron_op.children):
            num = kron_op.domain.dim // kron_op.domain[idx].dim
            c += num * op.exp_count(h, path, child_field(idx))
        return c


class KroneckerSum(KroneckerProductMixin):
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)

    def action(self, y: TensorState) -> TensorState:
        return reduce(lambda a, b: a + b, [
            apply_along_state(lambda s, op=op: op.action(s), y, factor_idx)
            for factor_idx, op in enumerate(self.children)
        ])

    def adj_action(self, y):
        return reduce(lambda a, b: a + b, [
            apply_along_state(lambda s, op=op: op.adj_action(s), y, factor_idx)
            for factor_idx, op in enumerate(self.children)
        ])

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


class KroneckerProduct(KroneckerProductMixin):

    def action(self, y: TensorState) -> TensorState:
        for factor_idx, op in enumerate(self.children):
            if isinstance(op, Identity):
                continue
            y = apply_along_state(lambda s, op=op: op.action(s), y, factor_idx)

        return y

    def adj_action(self, y):
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
        mat_list = [
            op.to_matrix() for idx, op in enumerate(self.children)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def adjoint(self) -> Operator:
        return KroneckerProduct(self.domain, tuple(op.adjoint() for op in self.children))
