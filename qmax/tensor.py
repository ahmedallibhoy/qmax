from __future__ import annotations

from typing import Callable, ClassVar, Union
from functools import reduce
import itertools

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike, ScalarLike

from .hilbert_space import AbstractHilbertSpace, AbstractState
from .operator import Operator
from .exponentiators import Order, AbstractExponentiator, ExactExponentiator, KrylovExponentiator


class TensorProductState(AbstractState):
    hilbert_space: TensorProduct = eqx.field(static=True)

    @property
    def coeff_tensor(self) -> Array:
        return self.coeffs.reshape(*self.coeffs.shape[:-1], *self.hilbert_space.dim_list)

    @property
    def spaces(self) -> list[AbstractHilbertSpace]:
        return self.hilbert_space.spaces


class TensorProduct(AbstractHilbertSpace):
    state_type: ClassVar = TensorProductState
    spaces: tuple[AbstractHilbertSpace]

    @property
    def structure(self):
        return tuple(s.structure for s in self.spaces)

    @property
    def dim(self) -> int:
        return math.prod(self.dim_list)

    @property
    def dim_list(self) -> list[int]:
        return [h.dim for h in self.spaces]

    def from_tensor(self, coeff_tensor: ArrayLike):
        return self.from_coeffs(coeff_tensor.reshape((self.dim,)))

    def product_state(self, y_list: tuple[AbstractState]) -> TensorProductState:
        coeffs = reduce(lambda a, b: jnp.outer(a, b).flatten(), [y.coeffs for y in y_list])
        return self.from_coeffs(coeffs)


def apply_along_tensor_axis(
    fn: Callable[[ArrayLike], ArrayLike],
    tensor: ArrayLike,
    axis: int) -> Array:
    """Apply `fn` to each 1-D slice along `axis`, batched over the rest. `fn` may
    change the axis length; the output shape is inferred from `fn`'s output."""

    t = jnp.moveaxis(tensor, axis, 0)
    rest = t.shape[1:]
    out = jax.vmap(fn, in_axes=1, out_axes=1)(t.reshape(t.shape[0], -1))
    return jnp.moveaxis(out.reshape((out.shape[0],) + rest), 0, axis)


def apply_along_state_axis(
    state_fn: Callable[[AbstractState], AbstractState],
    y: TensorProductState,
    axis: int) -> TensorProductState:
    """Apply a subspace endomorphism `state_fn` along `axis` of a tensor product
    state, wrapping the from_coeffs / .coeffs / from_tensor boilerplate."""

    space = y.spaces[axis]
    fn = lambda col: state_fn(space.from_coeffs(col)).coeffs
    return y.hilbert_space.from_tensor(apply_along_tensor_axis(fn, y.coeff_tensor, axis))


class AbstractTensorOperator(Operator):
    """
    """


class LiftExp(AbstractExponentiator):

    def exp(
        self, 
        lift_op: LiftOperator, 
        h: ScalarLike, 
        y: TensorProductState) -> TensorProductState:
        
        return apply_along_state_axis(lambda s: lift_op.op.exp(h, s), y, lift_op.idx)

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, lift_op: LiftOperator) -> Order:
        return lift_op.op.exponentiator.order


class LiftOperator(AbstractTensorOperator):
    op: Operator
    idx: int
    num_factors: int
    exponentiator: AbstractExponentiator = eqx.field(default=LiftExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(self.op.domain if i == self.idx else AbstractHilbertSpace for i in range(self.num_factors))

    def action(self, y: TensorProductState) -> TensorProductState:
        return apply_along_state_axis(lambda s: self.op(s), y, self.idx)

    def exp_action(self, h: ScalarLike, y: TensorProductState) -> TensorProductState:
        return apply_along_state_axis(lambda s: self.op.exp_action(h, s), y, self.idx)

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        return self.op.spectral_bounds(hilbert_space.spaces[self.idx])

    def solve(self, b: TensorProductState, scale: ScalarLike=-1.0, shift: ScalarLike=0.0) -> TensorProductState:
        return apply_along_state_axis(lambda s: self.op.solve(s, scale, shift), b, self.idx)

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat_list = [
            jnp.eye(hs.dim) if jdx != self.idx else self.op.to_matrix(hs) 
            for jdx, hs in enumerate(hilbert_space.spaces)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "op": self.op.to_dict(h_scale)
        }


class KroneckerSumExp(AbstractExponentiator):

    def exp(
        self, 
        kron_op: KroneckerSum, 
        h: ScalarLike, 
        y: TensorProductState) -> TensorProductState:
        
        for idx, op in enumerate(kron_op.ops):
            y = apply_along_state_axis(lambda s, op=op: op.exp(h, s), y, idx)

        return y

    @property
    def order(self) -> Order:
        return jnp.inf

    def effective_order(self, kron_op: KroneckerSum) -> Order:
        return min([op.exponentiator.order for op in kron_op.ops])


class KroneckerSum(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KroneckerSumExp(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorProductState) -> TensorProductState:
        y_list = [
            apply_along_state_axis(lambda s, op=op: op(s), y, idx) 
            for idx, op in enumerate(self.ops)
        ]

        return reduce(lambda a, b: a + b, y_list)

    def exp_action(self, h: ScalarLike, y: TensorProductState) -> TensorProductState:
        for idx, op in enumerate(self.ops):
            y = apply_along_state_axis(lambda s, op=op: op.exp_action(h, s), y, idx)

        return y

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        bounds = [op.spectral_bounds(space) for op, space in zip(self.ops, hilbert_space.spaces)]
        return jnp.sum(jnp.array(bounds), axis=0)

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat = jnp.zeros((hilbert_space.dim, hilbert_space.dim), dtype=complex)
        for idx, op in enumerate(self.ops):
            mat_list = [
                jnp.eye(hs.dim) if jdx != idx else op.to_matrix(hs) 
                for jdx, hs in enumerate(hilbert_space.spaces)
            ]
            mat += reduce(lambda a, b: jnp.kron(a, b), mat_list)
        return mat

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": [op.to_dict(h_scale) for op in self.ops]
        }


class KroneckerProduct(AbstractTensorOperator):
    ops: tuple[Operator, ...]
    exponentiator: AbstractExponentiator = eqx.field(default=KrylovExponentiator(), kw_only=True)

    @property
    def domain(self):
        return tuple(op.domain for op in self.ops)

    def action(self, y: TensorProductState) -> TensorProductState:
        for idx, op in enumerate(self.ops):
            y = apply_along_state_axis(lambda s, op=op: op(s), y, idx)

        return y

    def spectral_bounds(self, hilbert_space: TensorProduct) -> Array:
        prods = jnp.array([1.0])
        for op, space in zip(self.ops, hilbert_space.spaces):
            lo, hi = op.spectral_bounds(space)
            prods = jnp.concatenate([prods * lo, prods * hi])   
        return jnp.array([jnp.min(prods), jnp.max(prods)])

    def to_matrix(self, hilbert_space: TensorProduct) -> Array:
        mat_list = [
            op.to_matrix(hs) for op, hs in zip(self.ops, hilbert_space.spaces)
        ]
        return reduce(lambda a, b: jnp.kron(a, b), mat_list)

    def to_dict(self, h_scale=1.0) -> dict:
        return {
            "class": type(self).__name__,
            "obj": self,
            "h_scale": h_scale,
            "exp_delegated": False,
            "ops": None
        }


class TensorPower(TensorProduct):
    state_type: ClassVar = TensorProductState
    subspace: AbstractHilbertSpace
    power: int

    @property
    def dim(self) -> int:
        return self.subspace.dim ** self.power

    @property
    def dim_list(self) -> list[int]:
        return [self.subspace.dim for h in range(self.power)]

