from __future__ import annotations

from typing import Callable, ClassVar, Union
from abc import abstractmethod
from functools import reduce
import itertools

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator, Identity
from .exponentiators import Order, min_order, AbstractExponentiator, ExactExponentiator, TruncatedTaylorExponentiator, NotExponentiableError


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

    @property
    @abstractmethod
    def structure(self):
        pass

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


class TensorProduct(AbstractTensorSpace):
    spaces: tuple[AbstractHilbertSpace, ...]

    @property
    def structure(self):
        return tuple(s.structure for s in self.spaces)

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

    @property
    def structure(self):
        return tuple(self.factorspace.structure for _ in range(self.power))

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
    """
    """


class LiftExp(AbstractExponentiator):

    def exp(
        self, 
        lift_op: LiftOperator, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        return apply_along_state(lambda s: lift_op.op.exp(h, s), y, lift_op.factor_idx)

    def check_exponentiable(self, op: Operator):
        if not isinstance(op, LiftOperator):
            raise NotExponentiableError(
                f"{type(self).__name__} can only exponentiate operators of type LiftOperator "
                f"but received operator of type {type(op).__name__}"
            )

        op.op.check_exponentiable_tree()

    @property
    def order(self) -> Order:
        return None

    def effective_order(self, lift_op: LiftOperator) -> Order:
        return lift_op.op.exp_order


class LiftOperator(AbstractTensorOperator):
    op: Operator
    factor_idx: int
    num_factors: int
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(self.op.domain if i == self.factor_idx else AbstractHilbertSpace for i in range(self.num_factors))

    def action(self, y: TensorState) -> TensorState:
        return apply_along_state(lambda s: self.op(s), y, self.factor_idx)

    def adj_action(self, y):
        return apply_along_state(lambda s: self.op.adj_action(s), y, self.factor_idx)

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        return self.op.spectral_bounds(hilbert_space[self.factor_idx])

    def solve(self, b: TensorState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> TensorState:
        return apply_along_state(lambda s: self.op.solve(s, scale, shift), b, self.factor_idx)

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat_list = [
            jnp.eye(hilbert_space[idx].dim) if idx != self.factor_idx else self.op.to_matrix(hilbert_space[idx])
            for idx in range(hilbert_space.num_factors)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def adjoint(self) -> Operator:
        return LiftOperator(self.op.adjoint(), self.factor_idx, self.num_factors)

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "op": self.op.to_dict(h_scale)
        }


class KroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        h: ScalarLike, 
        y: TensorState) -> TensorState:
        
        for factor_idx, op in enumerate(kron_op.ops):
            y = apply_along_state(lambda s, op=op: op.exp(h, s), y, factor_idx)

        return y

    def check_exponentiable(self, op: Operator):
        if not isinstance(op, KroneckerSum):
            raise NotExponentiableError(
                f"{type(self).__name__} can only exponentiate operators of type KroneckerSum "
                f"but received operator of type {type(op).__name__}"
            )

        for factor in op.ops:
            factor.check_exponentiable_tree()

    @property
    def order(self) -> Order:
        return None

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min_order(*[op.exp_order for op in kron_op.ops])


class KroneckerSum(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        return reduce(lambda a, b: a + b, [
            apply_along_state(lambda s, op=op: op(s), y, factor_idx)
            for factor_idx, op in enumerate(self.ops)
        ])

    def adj_action(self, y):
        return reduce(lambda a, b: a + b, [
            apply_along_state(lambda s, op=op: op.adj_action(s), y, factor_idx)
            for factor_idx, op in enumerate(self.ops)
        ])

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        bounds = [op.spectral_bounds(hilbert_space[idx]) for idx, op in enumerate(self.ops)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat = jnp.zeros((hilbert_space.dim, hilbert_space.dim), dtype=complex)
        for factor_idx, op in enumerate(self.ops):
            mat_list = [
                jnp.eye(hilbert_space[idx].dim) if idx != factor_idx else op.to_matrix(hilbert_space[idx])
                for idx in range(hilbert_space.num_factors)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat

    def adjoint(self) -> Operator:
        return KroneckerSum(tuple(op.adjoint() for op in self.ops))

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "ops": [op.to_dict(h_scale) for op in self.ops]
        }


class KroneckerProduct(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=TruncatedTaylorExponentiator(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorState) -> TensorState:
        for factor_idx, op in enumerate(self.ops):
            if isinstance(op, Identity):
                continue
            y = apply_along_state(lambda s, op=op: op(s), y, factor_idx)

        return y

    def adj_action(self, y):
        for factor_idx, op in enumerate(self.ops):
            if isinstance(op, Identity):
                continue
            y = apply_along_state(lambda s, op=op: op.adj_action(s), y, factor_idx)
        return y

    def spectral_bounds(self, hilbert_space: AbstractTensorSpace) -> Array:
        lo, hi = 1.0, 1.0
        for idx, op in enumerate(self.ops):
            a, b = op.spectral_bounds(hilbert_space[idx])
            corners = jnp.array([lo * a, lo * b, hi * a, hi * b])
            lo, hi = jnp.min(corners), jnp.max(corners)
        return jnp.array([lo, hi])

    def to_matrix(self, hilbert_space: AbstractTensorSpace) -> Array:
        mat_list = [
            op.to_matrix(hilbert_space[idx]) for idx, op in enumerate(self.ops)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def adjoint(self) -> Operator:
        return KroneckerProduct(tuple(op.adjoint() for op in self.ops))

    def to_dict(self, h_scale: int=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "ops": None
        }

